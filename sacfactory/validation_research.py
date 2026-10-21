"""Protected chronological, unseen-element and paired transfer evaluation.

Replays use current corrected observations, not reconstructed historical revisions.
Only saved pre-result forecasts establish prospective predictions.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .adaptive import AdaptivePredictor, eligible_records, scores
from .composition import transfer_feature, composition_name
from .domain import recipe_key, screen


COLUMNS = ['sample_id', 'record_uid', 'batch_id', 'composition', 'element_count', 'label',
           'p_screen_pass', 'p_target_only', 'fold', 'result_reported_at']


def fitting_ready(records, methods):
    return len(records) >= 12 and all(len({recipe_key(s) for s in records if screen(s, methods)['label'] == label}) >= 3 for label in (0, 1))


def train_predict(train, test, methods, iterations=8, batch_effect=True):
    model = AdaptivePredictor(iterations, batch_effect=batch_effect)
    x = np.array([transfer_feature(s) for s in train])
    model.fit(x, np.array([screen(s, methods)['label'] for s in train]), np.array([s['batch_id'] for s in train]))
    p = model.predict_proba(np.array([transfer_feature(s) for s in test]))[:, 1]
    provenance = {'train_sample_ids': [s['sample_id'] for s in train],
        'train_record_uids': [s.get('record_uid') for s in train],
        'train_batch_ids': sorted({s['batch_id'] for s in train}),
        'test_batch_ids': sorted({s['batch_id'] for s in test}),
        'kernel_fit_batch_ids': sorted({train[i]['batch_id'] for i in model.fit_indices}),
        'calibration_batch_ids': sorted({train[i]['batch_id'] for i in model.calibration_indices}),
        'learning': model.gp.diagnostics, 'calibration_status': model.calibration_status}
    return p, provenance


def rows_for(test, methods, probabilities, fold, target_only=None):
    return [{'sample_id': s['sample_id'], 'record_uid': s.get('record_uid'), 'batch_id': s['batch_id'],
        'composition': composition_name(s), 'element_count': len(s['elements']), 'label': screen(s, methods)['label'],
        'p_screen_pass': float(p), 'p_target_only': float(target_only[i]) if target_only is not None else None,
        'fold': fold, 'result_reported_at': s.get('result_reported_at')}
        for i, (s, p) in enumerate(zip(test, probabilities))]


def summarize(frame, column='p_screen_pass'):
    data = frame.dropna(subset=[column, 'label'])
    if data.empty:
        return None
    return {**scores(data['label'].to_numpy(dtype=int), data[column].to_numpy()),
            'independent_batches': int(data['batch_id'].nunique())}


def forward_validation(samples, methods, context, origin, iterations=8, max_folds=3,
                       novel_recipes=False, batch_effect=True):
    records, exclusions = eligible_records(samples, methods, context, origin, True)
    table = pd.DataFrame([{'index': i, 'batch_id': s['batch_id'], 'time': pd.to_datetime(s.get('result_reported_at'), utc=True, errors='coerce')}
                          for i, s in enumerate(records)], columns=['index', 'batch_id', 'time'])
    groups, missing = [], []
    for batch, frame in table.groupby('batch_id'):
        if frame['time'].isna().any():
            missing.append(batch)
        else:
            groups.append((batch, frame['time'].min(), frame['time'].max()))
    groups.sort(key=lambda item: (item[1], item[2], item[0]))
    candidates, skipped = [], []
    for batch, start, end in groups:
        earlier = {name for name, _, completion in groups if completion < start}
        train = [s for s in records if s['batch_id'] in earlier]
        test = [s for s in records if s['batch_id'] == batch]
        if novel_recipes:
            keys = {recipe_key(s) for s in test}
            train = [s for s in train if recipe_key(s) not in keys]
        if not fitting_ready(train, methods):
            skipped.append({'batch_id': batch, 'reason': 'Earlier completed batches lack fitting diversity'})
        else:
            candidates.append((batch, start, end, train, test))
    max_folds = max(1, min(12, int(max_folds)))
    selected = np.unique(np.linspace(0, len(candidates)-1, min(max_folds, len(candidates)), dtype=int)) if candidates else []
    rows, folds = [], []
    for index in selected:
        batch, start, end, train, test = candidates[index]
        p, provenance = train_predict(train, test, methods, iterations, batch_effect)
        provenance.update(test_batch_start=start.isoformat(), test_batch_completed=end.isoformat(),
                          latest_train_result=max(pd.to_datetime(s['result_reported_at'], utc=True) for s in train).isoformat())
        folds.append(provenance)
        rows.extend(rows_for(test, methods, p, batch))
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame, {'protocol': 'CHRONOLOGICAL_CORRECTED_DATA_REPLAY', 'novel_recipe_purge': bool(novel_recipes),
        'eligible_n': len(records), 'evaluated_n': len(frame), 'candidate_folds': len(candidates), 'evaluated_folds': len(folds),
        'fold_budget': max_folds, 'batches_missing_timestamps': missing, 'skipped': skipped, 'exclusions': exclusions,
        'metrics': summarize(frame), 'folds': folds,
        'note': 'Current corrected data replay ordered by result availability; not a saved prospective forecast or a reconstruction of historical revisions.'}


def unseen_element_validation(samples, methods, context, origin, element, iterations=8, batch_effect=True):
    records, exclusions = eligible_records(samples, methods, context, origin, True)
    test = [s for s in records if element in s['elements']]
    test_batches = {s['batch_id'] for s in test}
    train = [s for s in records if element not in s['elements'] and s['batch_id'] not in test_batches]
    info = {'protocol': 'STRICT_UNSEEN_ELEMENT', 'held_out_element': element, 'eligible_n': len(records),
            'test_n': len(test), 'source_n': len(train), 'exclusions': exclusions, 'folds': [], 'metrics': None}
    if not test or not fitting_ready(train, methods):
        info['reason'] = 'Need held-out target outcomes and diverse source conditions after removing every target-containing sample and shared batches.'
        return pd.DataFrame(columns=COLUMNS), info
    p, provenance = train_predict(train, test, methods, iterations, batch_effect)
    provenance['train_elements'] = sorted({e for s in train for e in s['elements']})
    provenance['held_out_element'] = element
    frame = pd.DataFrame(rows_for(test, methods, p, element), columns=COLUMNS)
    info.update(evaluated_n=len(frame), folds=[provenance], metrics=summarize(frame),
                reason='Every training composition excludes the held-out element. Shared actual batches are purged. Retrospective transfer test, not chronological forecasting.')
    return frame, info


def paired_effect(frame, draws=1000, seed=42):
    data = frame.dropna(subset=['label', 'p_target_only', 'p_screen_pass']).copy()
    if data.empty:
        return {'status': 'UNAVAILABLE', 'n': 0}
    data['delta'] = (data['label']-data['p_target_only'])**2 - (data['label']-data['p_screen_pass'])**2
    group = data.groupby('batch_id')['delta'].agg(['sum', 'count'])
    result = {'n': len(data), 'independent_batches': len(group), 'brier_improvement': float(data['delta'].mean()),
              'lower_95': None, 'upper_95': None, 'status': 'INSUFFICIENT_BATCHES', 'bootstrap_draws': draws}
    if len(group) >= 3:
        pick = np.random.default_rng(seed).integers(0, len(group), (draws, len(group)))
        values = group['sum'].to_numpy()[pick].sum(axis=1) / group['count'].to_numpy()[pick].sum(axis=1)
        low, high = np.quantile(values, [.025, .975])
        result.update(lower_95=float(low), upper_95=float(high),
                      status='NEGATIVE_TRANSFER_SIGNAL' if high < 0 else 'POSITIVE_TRANSFER_SIGNAL' if low > 0 else 'INCONCLUSIVE')
    result['note'] = 'Sample-weighted paired Brier difference; actual-batch cluster bootstrap. Exploratory comparison, no automatic model replacement.'
    return result


def negative_transfer_validation(samples, methods, context, origin, iterations=8, max_folds=3, batch_effect=True):
    records, exclusions = eligible_records(samples, methods, context, origin, True)
    target = [s for s in records if set(s['elements']) == set(context['elements'])]
    rows, folds, skipped = [], [], []
    groups = np.array([s['batch_id'] for s in target])
    info = {'protocol': 'PAIRED_ADAPTIVE_TRANSFER', 'target_n': len(target), 'source_pool_n': len(records), 'exclusions': exclusions}
    if len(set(groups)) >= 3:
        for fold, (_, indices) in enumerate(GroupKFold(n_splits=min(max(2, int(max_folds)), len(set(groups)), 6)).split(target, groups=groups)):
            test = [target[i] for i in indices]
            test_batches = {s['batch_id'] for s in test}
            pooled = [s for s in records if s['batch_id'] not in test_batches]
            local = [s for s in pooled if set(s['elements']) == set(context['elements'])]
            if not fitting_ready(local, methods) or not fitting_ready(pooled, methods):
                skipped.append({'fold': fold, 'reason': 'Target-only or pooled fit lacks recipe/class diversity'})
                continue
            p, provenance = train_predict(pooled, test, methods, iterations, batch_effect)
            q, local_provenance = train_predict(local, test, methods, iterations, batch_effect)
            folds.append({'transfer': provenance, 'target_only': local_provenance})
            rows.extend(rows_for(test, methods, p, fold, q))
    frame = pd.DataFrame(rows, columns=COLUMNS)
    info.update(evaluated_n=len(frame), folds=folds, skipped=skipped,
                transfer_metrics=summarize(frame), target_only_metrics=summarize(frame, 'p_target_only'), effect=paired_effect(frame),
                note='Both adaptive pipelines refit on the same outer target test batches; source batches linked to the test are removed. Calibration and learning stay inside each arm. Retrospective comparison.')
    return frame, info
