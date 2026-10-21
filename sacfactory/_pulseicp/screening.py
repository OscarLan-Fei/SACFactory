"""Optional labeled screening heads with batch-protected fit and calibration."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .composition import Composition, component_features, as_composition
from .data import PreparedData
from .ensemble import feature_names, model_references
from .features import TrainingScaler
from .references import digest
from .schema import Recipe
from .transfer import protected_split

TARGETS = ('icp_pass', 'xrd_pass')
SCHEMA = 'composition-screening-logistic-v1'


def label(value):
    if value is None or str(value).strip().upper() in {'', 'NAN', '<NA>', 'NONE', 'UNKNOWN', 'NOT MEASURED'}:
        return None
    text = str(value).strip().upper()
    if text in {'PASS', 'TRUE', '1', '1.0'}:
        return 1.
    if text in {'FAIL', 'FALSE', '0', '0.0'}:
        return 0.
    raise ValueError('Use PASS/FAIL, 1/0 or blank for an unmeasured screening endpoint.')


def screening_features(composition, physics, thermodynamic_features=True):
    """One molarity-weighted vector per physical specimen, independent of results."""
    c = as_composition(composition)
    weights = np.asarray([c.concentrations_mM[e] for e in c.elements], dtype=float)
    weights /= weights.sum()
    return weights @ np.asarray([component_features(c, e, physics, thermodynamic_features) for e in c.elements])


def prepare_screening(frame, physics, origin, target, thermodynamic_features=True):
    if target not in TARGETS or origin not in {'SIMULATED', 'EXPERIMENTAL'}:
        raise ValueError('Choose an explicit screening endpoint and data origin.')
    if target not in frame:
        raise ValueError(f'Add the optional {target} column with actual sample-level labels.')
    records, audits, xx, yy = [], [], [], []
    groups = {}
    for raw in frame.to_dict('records'):
        groups.setdefault((str(raw.get('origin', '')), str(raw.get('sample_id', ''))), []).append(raw)
    for (_, ident), rows in groups.items():
        reason = ''
        try:
            if rows[0].get('origin') != origin:
                raise ValueError('Different or undeclared origin; never pooled into screening training.')
            if not ident.strip() or ident.upper() in {'NAN', 'NONE', '<NA>'}:
                raise ValueError('A nonblank sample ID is required.')
            for key in ('batch_id', 'protocol_id'):
                values = {str(r.get(key, '')).strip() for r in rows}
                if len(values) != 1 or next(iter(values)).upper() in {'', 'NAN', 'NONE', '<NA>', 'UNKNOWN', 'UNASSIGNED'}:
                    raise ValueError('A consistent actual '+key+' is required.')
            if len({str(r.get('result_reported_at')) for r in rows}) != 1:
                raise ValueError('Mixture rows must share result reporting time.')
            components = [Recipe.from_row(r) for r in rows]
            if len({r.element for r in components}) != len(components):
                raise ValueError('Duplicate sample-element rows cannot be overwritten.')
            first = components[0]
            shared = lambda r: (r.temperature_C, r.pulse_count, r.pan_mass_g, r.volume_mL, r.carbonization_loss_fraction)
            if any(shared(r) != shared(first) for r in components):
                raise ValueError('Mixture rows need consistent process and material conditions.')
            labels = {v for r in rows if (v := label(r.get(target))) is not None}
            if len(labels) != 1:
                raise ValueError('Record one consistent sample-level label; blank or conflicting endpoints stay audited.')
            c = Composition({r.element: r.concentration_mM for r in components}, *shared(first)).validate()
            value = labels.pop()
            vector = screening_features(c, physics, thermodynamic_features)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            reason = str(exc)
        audits.append({'sample_id': ident, 'origin': rows[0].get('origin'), 'target': target,
                       'eligible': not reason, 'exclusion_reason': reason})
        if not reason:
            records.append({'raw': rows[0].copy(), 'recipe': first, 'composition': c})
            xx.append(vector); yy.append(value)
    y = np.asarray(yy, dtype=float)
    return PreparedData(records, pd.DataFrame(audits),
        np.asarray(xx, dtype=np.float32).reshape(-1, len(feature_names({'thermodynamic_features':thermodynamic_features}))), y, y.copy())


def _fit_logistic(x, y, penalty=.01):
    torch.set_num_threads(1)
    xx, yy = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    coef = torch.zeros(xx.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(math.log(float(yy.mean())/(1-float(yy.mean()))), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([coef, bias], max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(xx @ coef+bias, yy) + .5*penalty*(coef**2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    result = {'coefficient': coef.detach().numpy().tolist(), 'intercept': float(bias.detach())}
    if not np.isfinite(result['coefficient']+[result['intercept']]).all():
        raise ValueError('Nonfinite screening fit; no checkpoint was saved.')
    return result


def _probability(x, parameters):
    logits = np.asarray(x) @ np.asarray(parameters['coefficient']) + parameters['intercept']
    return 1/(1+np.exp(-np.clip(logits, -700., 700.)))


def classification_metrics(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    p = np.clip(probability, 1e-12, 1-1e-12)
    positive, negative = int(y.sum()), int(len(y)-y.sum())
    auc = balanced = ap = None
    if positive and negative:
        rank = pd.Series(p).rank(method='average').to_numpy()
        auc = float((rank[y==1].sum()-positive*(positive+1)/2)/(positive*negative))
        balanced = float(.5*((p[y==1]>=.5).mean()+(p[y==0]<.5).mean()))
        tp = fp = recall = area = 0.
        for threshold in sorted(set(p), reverse=True):
            block = y[p==threshold]
            tp += block.sum(); fp += len(block)-block.sum()
            new_recall = tp/positive
            area += (new_recall-recall)*tp/(tp+fp)
            recall = new_recall
        ap = float(area)
    return {'n_specimens': len(y), 'n_pass': positive, 'n_fail': negative,
        'brier_score': float(np.mean((probability-y)**2)),
        'log_loss': float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
        'balanced_accuracy': balanced, 'auroc': auc, 'average_precision': ap}


class ScreeningModel:
    def __init__(self, metadata):
        self.metadata = metadata
        self.scaler = TrainingScaler.from_dict(metadata['scaler'])

    def predict(self, compositions):
        compositions = [as_composition(c) for c in compositions]
        if not compositions:
            return pd.DataFrame()
        raw = np.asarray([screening_features(c, self.metadata['physics'],
            self.metadata['settings']['thermodynamic_features']) for c in compositions])
        p = _probability(self.scaler.transform(raw), self.metadata['parameters'])
        if self.metadata['calibration']['status'] == 'PLATT_HOLDOUT':
            logit = np.log(np.clip(p, 1e-12, 1-1e-12)/(1-np.clip(p, 1e-12, 1-1e-12)))[:,None]
            p = _probability(logit, self.metadata['calibration']['parameters'])
        outside = self.scaler.outside(raw)
        rows = []
        target = self.metadata['target']
        for i,c in enumerate(compositions):
            unsupported = bool(outside[i] or c.key not in self.metadata['compositions'] or
                               any(e not in self.metadata['elements'] for e in c.elements))
            calibrated = self.metadata['calibration']['status']=='PLATT_HOLDOUT' and not unsupported
            rows.append({'composition_index': i, f'predicted_{target}_probability': float(p[i]),
                f'{target}_model_id': self.metadata['model_id'], f'{target}_model_origin': self.metadata['origin'],
                f'{target}_protocol_id': self.metadata['protocol'],
                f'{target}_probability_status': 'SOURCE_HOLDOUT_CALIBRATION' if calibrated else 'PROVISIONAL_UNCALIBRATED',
                f'{target}_outside_training_support': unsupported,
                f'{target}_uncertainty_status': 'NOT_ESTIMATED'})
        return pd.DataFrame(rows)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'screening.json').write_text(json.dumps(self.metadata, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    @classmethod
    def load(cls, directory):
        meta = json.loads((Path(directory)/'screening.json').read_text(encoding='utf-8'))
        ident = meta.pop('model_id')
        if ident != digest(meta)[:20]:
            raise ValueError('Screening checkpoint failed its integrity check.')
        meta['model_id'] = ident
        if meta['schema'] != SCHEMA or meta['feature_names'] != feature_names(meta['settings']):
            raise ValueError('Screening feature schema changed; retrain.')
        if meta['references'] != model_references(meta['settings']):
            raise ValueError('Screening reference snapshots changed; retrain.')
        return cls(meta)


def train_screening(frame, origin, physics, output, target='xrd_pass', split='Batch holdout',
                    seed=20261018, held_element=None, source_elements=None, thermodynamic_features=True):
    settings = {'thermodynamic_features':bool(thermodynamic_features)}
    data = prepare_screening(frame, physics, origin, target, thermodynamic_features)
    if not data.records:
        raise ValueError('No eligible labeled screening specimens; inspect the endpoint audit.')
    if len({r['raw']['protocol_id'] for r in data.records}) != 1:
        raise ValueError('Filter screening labels to one comparable protocol before training.')
    (train, cal, test), partitions = protected_split(data, seed, split, held_element, source_elements)
    if min(np.sum(data.w[train]==0), np.sum(data.w[train]==1)) < 4:
        raise ValueError('Screening fitting needs at least four PASS and four FAIL training specimens.')
    scaler = TrainingScaler().fit(data.x[train])
    parameters = _fit_logistic(scaler.transform(data.x[train]), data.w[train])
    calibration = {'status': 'INSUFFICIENT_CLASS_BALANCED_CALIBRATION', 'parameters': None}
    cal_batches = {str(data.records[i]['raw']['batch_id']) for i in cal}
    if len(cal_batches) >= 2 and len(cal) >= 20 and min(np.sum(data.w[cal]==0), np.sum(data.w[cal]==1)) >= 5:
        base = _probability(scaler.transform(data.x[cal]), parameters)
        logit = np.log(np.clip(base, 1e-12, 1-1e-12)/(1-np.clip(base, 1e-12, 1-1e-12)))[:,None]
        calibration = {'status': 'PLATT_HOLDOUT', 'parameters': _fit_logistic(logit, data.w[cal], .01)}
    records = [data.records[i] for i in train]
    metadata = {'schema': SCHEMA, 'target': target, 'origin': origin,
        'protocol': records[0]['raw']['protocol_id'], 'physics': physics,
        'settings': settings, 'feature_names': feature_names(settings), 'references': model_references(settings),
        'parameters': parameters, 'scaler': scaler.to_dict(), 'calibration': calibration,
        'elements': sorted({e for r in records for e in r['composition'].elements}),
        'compositions': sorted({r['composition'].key for r in records}),
        'partitions': partitions, 'split': split, 'seed': seed,
        'fit_fingerprint': digest([{'sample_id':r['raw']['sample_id'], 'batch_id':r['raw']['batch_id'],
            'composition':r['composition'].concentrations_json, 'x':data.x[i].tolist(), 'label':float(data.w[i])}
            for i,r in zip(train,records)]),
        'calibration_fingerprint': digest([{'sample_id':data.records[i]['raw']['sample_id'],
            'x':data.x[i].tolist(), 'label':float(data.w[i])} for i in cal]),
        'assessment': 'Synthetic software evidence only' if origin=='SIMULATED' else 'Held-out screening labels; not proof of atomic dispersion'}
    metadata['model_id'] = digest(metadata)[:20]
    model = ScreeningModel(metadata)
    predictions = model.predict([data.records[i]['composition'] for i in test])
    predictions['sample_id'] = [data.records[i]['raw']['sample_id'] for i in test]
    predictions['batch_id'] = [data.records[i]['raw']['batch_id'] for i in test]
    predictions['observed_'+target] = data.w[test]
    summary = classification_metrics(data.w[test], predictions['predicted_'+target+'_probability'])
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
    destination = Path(output)/run_id
    destination.mkdir(parents=True, exist_ok=False)
    model.save(destination)
    predictions.to_csv(destination/'test_predictions.csv', index=False)
    data.audit.to_csv(destination/'eligibility.csv', index=False)
    (destination/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    pointer = Path(output)/('latest_'+target+'.json')
    temporary = pointer.with_suffix('.tmp')
    temporary.write_text(json.dumps({'run_id':run_id, 'target':target}), encoding='utf-8')
    temporary.replace(pointer)
    return destination, model, predictions, summary
