"""Rotating whole-batch ICP validation with one held-out prediction per target."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from ._pulseicp.data import prepare
from .accelerated_ensemble import FAMILIES, ResidualEnsemble
from ._pulseicp.references import digest, provenance
from ._pulseicp.uncertainty import metrics
from ._pulseicp.validation import predict_records

PROTOCOL = 'Rotating batch cross-validation'
VERSION = 'whole-batch-out-of-fold-v1'


def batch_groups(batches, per_fold, seed):
    """Disjoint test groups use only batch IDs and a prespecified seed."""
    batches = sorted(set(map(str, batches)))
    if isinstance(per_fold, bool) or not isinstance(per_fold, (int, np.integer)):
        raise ValueError('Batches per test fold must be an integer.')
    if len(batches) < 3 or not 1 <= per_fold <= len(batches) - 2:
        raise ValueError('Need at least three actual batches and two training batches in every fold.')
    np.random.default_rng(int(seed)).shuffle(batches)
    return [batches[start:start+per_fold] for start in range(0, len(batches), per_fold)]


def dataset_fingerprint(frame):
    return digest(frame.astype(object).where(pd.notna(frame), None).to_dict('records'))


def run_cross_validation(frame, origin, physics, settings, output, family=FAMILIES[0],
                         batches_per_fold=1, context=None, progress=None):
    """Refit the selected algorithm inside each outer fold; never refit on test W.

    No calibration outcomes enter these fits. Intervals are raw model quantiles.
    Reports have their own pointer and never replace a deployment checkpoint.
    """
    settings, physics = copy.deepcopy(settings), copy.deepcopy(physics)
    if family not in FAMILIES:
        raise ValueError('Select Gaussian residual or Conditional diffusion.')
    data = prepare(frame, physics, origin, settings.get('thermodynamic_features', True))
    if not data.records:
        raise ValueError('No eligible quantified ICP records. Inspect the eligibility audit.')
    protocols = {str(r['raw']['protocol_id']) for r in data.records}
    if len(protocols) != 1:
        raise ValueError('Select one comparable synthesis protocol.')
    batch_ids = np.asarray([str(r['raw']['batch_id']) for r in data.records])
    groups = batch_groups(batch_ids, batches_per_fold, settings['seed'])
    run_id = 'cv_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    destination = Path(output) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    info = {'version': VERSION, 'run_id': run_id, 'origin': origin, 'protocol': next(iter(protocols)),
        'split_protocol': PROTOCOL, 'family': family, 'settings': settings, 'physics': physics,
        'batches_per_test_fold': int(batches_per_fold), 'dataset_fingerprint': dataset_fingerprint(frame),
        'reference_provenance': provenance(), 'sacfactory_process_key': context,
        'eligible_n': len({r['raw']['sample_id'] for r in data.records}),
        'eligible_target_rows': len(data.records), 'recorded_n': int(frame.sample_id.nunique()),
        'partitions': [], 'skipped_folds': [],
        'prediction_kind': 'BATCH_OUT_OF_FOLD', 'deployment_checkpoint_updated': False,
        'interval_status': 'PROVISIONAL_UNCALIBRATED',
        'assessment': 'Synthetic implementation evidence only' if origin == 'SIMULATED' else 'Retrospective whole-batch cross-validation',
        'note': 'One fixed algorithm and settings; different fitted weights in each fold. Each evaluated sample-element pair occurs once. Whole test batches are absent from training, preprocessing and bootstraps. Repeated recipes may occur in other batches; this is new-batch validation, not unseen-recipe or chronological validation. Model/settings/group-size selection on these results is development, not an additional independent test. Intervals have no separate empirical calibration.'}
    predicted, all_draws, per_fold, skipped_by_id = [], [], [], {}
    try:
        for fold, held_batches in enumerate(groups, 1):
            test = np.flatnonzero(np.isin(batch_ids, held_batches))
            train = np.flatnonzero(~np.isin(batch_ids, held_batches))
            train_ids = sorted({str(data.records[i]['raw']['sample_id']) for i in train})
            test_ids = sorted({str(data.records[i]['raw']['sample_id']) for i in test})
            train_batches = sorted(set(batch_ids[train]))
            assert not set(train_ids) & set(test_ids)
            assert not set(train_batches) & set(held_batches)
            partition = {'fold': fold, 'train_batch_ids': train_batches, 'test_batch_ids': held_batches,
                'train_sample_ids': train_ids, 'test_sample_ids': test_ids,
                'training_specimens': len(train_ids), 'test_specimens': len(test_ids)}
            info['partitions'].append(partition)
            if len(train_ids) < 16 or len(train_batches) < 2:
                reason = 'Need sixteen eligible training specimens and two training batches.'
                info['skipped_folds'].append({'fold': fold, 'reason': reason})
                skipped_by_id.update(dict.fromkeys(test_ids, reason))
                if progress:
                    progress(fold, len(groups))
                continue
            local_settings = {**settings, 'seed': int(settings['seed']) + fold * 1009}
            model = ResidualEnsemble(family, local_settings, physics).fit(data, train)
            partition.update(model_id=model.model_id, training_fingerprint=model.training_fingerprint,
                training_seed=local_settings['seed'], bootstrap=model.bootstrap,
                scaler=model.scaler.to_dict())
            points, draws = predict_records(model, data, test, origin, next(iter(protocols)))
            points['sample_id'] = [data.records[i]['raw']['sample_id'] for i in test]
            points['batch_id'] = [data.records[i]['raw']['batch_id'] for i in test]
            points['observed_w_wt_percent'] = data.w[test]
            points['fold'] = fold
            points['training_specimens'] = len(train_ids)
            points['prediction_kind'] = 'BATCH_OUT_OF_FOLD'
            predicted.append(points)
            all_draws.append(draws)
            per_fold.append({'fold': fold, 'test_batches': ', '.join(held_batches),
                'training_specimens': len(train_ids), 'model': family, **metrics(points, draws)})
            if progress:
                progress(fold, len(groups))
        if predicted:
            combined = pd.concat(predicted, ignore_index=True)
            if combined.duplicated(['sample_id', 'element']).any():
                raise RuntimeError('A target received multiple cross-validation predictions.')
            comparisons = [{'model': family, **metrics(combined, np.concatenate(all_draws, axis=0))}]
        else:
            combined = pd.DataFrame(columns=['sample_id', 'element', 'model'])
            comparisons = []
        seen = set(zip(combined.sample_id, combined.element))
        audit = data.audit.copy()
        audit['evaluated'] = [(r.sample_id, r.element) in seen for r in audit.itertuples()]
        audit['coverage_reason'] = ['Predicted once in held-out batch' if row.evaluated else
            row.exclusion_reason if not row.eligible else skipped_by_id.get(str(row.sample_id), 'No evaluable fold')
            for row in audit.itertuples()]
        info.update(status='COMPLETE' if len(seen) == len(data.records) else 'PARTIAL_COVERAGE',
            evaluated_specimens_n=int(combined.sample_id.nunique()), evaluated_target_rows=len(seen),
            coverage_fraction=len(seen)/len(data.records), comparisons=comparisons, fold_metrics=per_fold)
        combined.to_csv(destination/'test_predictions.csv', index=False)
        pd.DataFrame(comparisons).to_csv(destination/'comparison.csv', index=False)
        pd.DataFrame(per_fold).to_csv(destination/'fold_metrics.csv', index=False)
        audit.to_csv(destination/'eligibility.csv', index=False)
        info['artifacts_sha256'] = {name:hashlib.sha256((destination/name).read_bytes()).hexdigest()
            for name in ('test_predictions.csv','comparison.csv','fold_metrics.csv','eligibility.csv')}
        (destination/'run.json').write_text(json.dumps(info, indent=2, allow_nan=False), encoding='utf-8')
        # Cross-validation reports cannot become production model weights.
        pointer = Path(output)/'latest_cv.json'
        temporary = Path(output)/f'.latest_cv_{uuid4().hex}.tmp'
        temporary.write_text(json.dumps({'run_id':run_id, 'origin':origin}), encoding='utf-8')
        temporary.replace(pointer)
        return destination, combined, pd.DataFrame(comparisons), info
    except Exception:
        info['status'] = 'FAILED_NOT_PROMOTED'
        (destination/'run.json').write_text(json.dumps(info, indent=2, allow_nan=False), encoding='utf-8')
        raise


def load_matching_report(output, frame, origin, physics, settings, family, batches_per_fold, context):
    """Reload the last report only when its data, source scope and settings match."""
    output = Path(output)
    pointer = output/'latest_cv.json'
    if not pointer.exists():
        return None
    reference = json.loads(pointer.read_text(encoding='utf-8'))
    run_id = reference['run_id']
    if Path(run_id).name != run_id or not run_id.startswith('cv_'):
        raise ValueError('Invalid cross-validation report reference.')
    destination = output/run_id
    info = json.loads((destination/'run.json').read_text(encoding='utf-8'))
    if (info.get('status') not in {'COMPLETE', 'PARTIAL_COVERAGE'} or info.get('origin') != origin or
        info.get('version') != VERSION or info.get('dataset_fingerprint') != dataset_fingerprint(frame) or
        digest(info.get('settings')) != digest(settings) or digest(info.get('physics')) != digest(physics) or
        info.get('family') != family or info.get('batches_per_test_fold') != batches_per_fold or
        digest(info.get('sacfactory_process_key')) != digest(context) or
        digest(info.get('reference_provenance')) != digest(provenance())):
        return None
    hashes = info.get('artifacts_sha256', {})
    for name in ('test_predictions.csv','comparison.csv','fold_metrics.csv','eligibility.csv'):
        if name not in hashes or hashlib.sha256((destination/name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError('Cross-validation report integrity changed; rerun validation.')
    points = pd.read_csv(destination/'test_predictions.csv', keep_default_na=False,
        dtype={'sample_id':str, 'element':str, 'batch_id':str, 'model_id':str})
    comparison = pd.DataFrame(info['comparisons'])
    return destination, points, comparison, info
