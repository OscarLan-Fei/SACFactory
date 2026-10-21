"""Outcome maps, repeatability and auditable prediction evaluation.

All calculations use recorded screening labels. No instrument traces are read.
Model surfaces are conditional slices, never evidence of atomic dispersion.
"""
import copy
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import GroupKFold

from .analysis import experiment_frame, numeric, wilson_interval
from .composition import composition_name, process_key, transfer_feature, transfer_classifier
from .domain import ELEMENTS, RULE_VERSION, cohort_key, recipe_key, screen
from .manual import actual_batch_recorded
from .planning import classifier, feature, fit_model, independent_groups
from .queueing import controlled_recipe_key


def analysis_records(samples, methods):
    frame = experiment_frame(samples, {s['sample_id']: screen(s, methods) for s in samples})
    frame['composition'] = [composition_name(s) for s in samples]
    frame['total_concentration_mM'] = [sum(s['concentrations_mM'].values()) if all(numeric(v) is not None for v in s['concentrations_mM'].values()) else None for s in samples]
    return frame


def readiness(samples, methods, context, origin, transfer=False):
    """Mirror model selection without fitting, with per-record explanations."""
    rows = []
    for sample in samples:
        reasons = []
        result = screen(sample, methods)
        if sample.get('origin') != origin:
            reasons.append('Different data namespace')
        if (process_key(sample) if transfer else cohort_key(sample)) != (process_key(context) if transfer else cohort_key(context)):
            reasons.append('Different process or composition')
        if any(str(sample.get(k) or 'UNKNOWN').strip().upper() in {'UNKNOWN', 'UNASSIGNED'} or str(sample.get(k)).strip().upper().startswith(('FILL_IN_', 'REPLACE_')) for k in ('batch_id', 'precursor_family', 'support_batch', 'protocol_id')):
            reasons.append('Unresolved process metadata')
        if sample.get('pulse_count') is None or any(numeric(sample['concentrations_mM'].get(e)) is None or numeric(sample['concentrations_mM'][e]) <= 0 for e in sample['elements']) or (sample.get('pulse_limit_C') is None and sample.get('temperature_C') is None):
            reasons.append('Incomplete synthesis conditions')
        if result['review_conflicts']:
            reasons.append('Conflicting judgments')
        if result['label'] is None:
            reasons.append('Unresolved outcome or invalid run')
        rows.append({'sample_id': sample['sample_id'], 'batch_id': sample['batch_id'], 'composition': composition_name(sample), 'status': result['status'], 'eligible': not reasons, 'reason': '; '.join(reasons) or 'Eligible'})
    return pd.DataFrame(rows, columns=['sample_id', 'batch_id', 'composition', 'status', 'eligible', 'reason'])


def gate_summary(samples, methods, context, origin, transfer=False):
    detail = readiness(samples, methods, context, origin, transfer)
    ids = set(detail.loc[detail['eligible'], 'sample_id'])
    chosen = [s for s in samples if s['sample_id'] in ids]
    counts = Counter(screen(s, methods)['label'] for s in chosen)
    recipes = {label: len({recipe_key(s) for s in chosen if screen(s, methods)['label'] == label}) for label in (0, 1)}
    batches = len({s['batch_id'] for s in chosen})
    minimum_batches = 8 if origin == 'REAL' else 3
    return detail, {'eligible': len(chosen), 'batches': batches, 'passes': counts[1], 'failures': counts[0], 'pass_recipes': recipes[1], 'fail_recipes': recipes[0], 'minimum_batches': minimum_batches, 'ready': len(chosen) >= 12 and batches >= minimum_batches and min(recipes.values()) >= 3}


def repeatability(frame):
    """Exact recipes; separate between-batch recurrence from same-batch repeats."""
    rows = []
    for recipe, group in frame.dropna(subset=['recipe']).groupby('recipe', sort=False):
        resolved = group.dropna(subset=['label'])
        named = resolved.loc[~resolved['batch_id'].isin(['UNKNOWN', 'UNASSIGNED']) & ~resolved['batch_id'].str.startswith(('FILL_IN_', 'REPLACE_'))]
        passes = int((resolved['label'] == 1).sum())
        low, high = wilson_interval(passes, len(resolved))
        batch_means = named.groupby('batch_id')['label'].mean()
        first = group.iloc[0]
        rows.append({'recipe': recipe, 'example_sample': first['sample_id'], 'composition': first['composition'] if 'composition' in group else '', 'records': len(group), 'resolved': len(resolved), 'independent_batches': len(batch_means), 'same_batch_extra_records': len(named) - len(batch_means), 'passed': passes, 'burned_out': int(((resolved['status'] == 'BURNED_OUT')).sum()), 'pass_rate': passes / len(resolved) if len(resolved) else None, 'lower_95': low, 'upper_95': high, 'equal_batch_pass_rate': batch_means.mean() if len(batch_means) else None, 'concentrations_mM': '; '.join(f'{c[:-3]}: {first[c]:g}' for c in group if c.endswith('_mM') and pd.notna(first[c])), 'pulse_count': first['pulse_count'], 'pulse_limit_C': first['pulse_limit_C'], 'temperature_C': first['temperature_C']})
    return pd.DataFrame(rows)


def failure_breakdown(frame):
    """Keep invalid runs and conflicts out of chemical failure denominators."""
    data = frame.copy()
    data['category'] = np.where(data['conflict'], 'CONFLICT', np.where(data['status'] == 'RETEST', data['sample_condition'], data['status']))
    return data.groupby(['batch_id', 'category'], observed=True).size().reset_index(name='records')


def composition_coverage(samples, methods):
    frame = analysis_records(samples, methods)
    elements = []
    for element in ELEMENTS:
        indices = [i for i, s in enumerate(samples) if element in s['elements']]
        rows = frame.iloc[indices]
        elements.append({'element': element, 'records': len(rows), 'resolved': int(rows['label'].notna().sum()), 'passed': int((rows['label'] == 1).sum()), 'single_element_records': int((rows['element_count'] == 1).sum()), 'mixture_records': int((rows['element_count'] > 1).sum())})
    pairs = Counter(pair for s, label in zip(samples, frame['label']) if pd.notna(label) for pair in combinations(sorted(s['elements'], key=ELEMENTS.index), 2))
    pair_frame = pd.DataFrame([{'element_1': a, 'element_2': b, 'resolved_records': count} for (a, b), count in pairs.items()], columns=['element_1', 'element_2', 'resolved_records'])
    stages = frame.groupby('element_count').agg(records=('sample_id', 'size'), resolved=('label', 'count'), passed=('label', 'sum'), compositions=('composition', 'nunique')).reset_index()
    return pd.DataFrame(elements), pair_frame, stages


def conditional_surface(samples, methods, context, origin, element, pulse, resolution=12, engine='baseline'):
    """Fit the existing exact-composition model and slice observed input ranges.

    Other element concentrations stay at the explicitly returned anchor recipe.
    Predictive entropy measures ambiguity, not calibrated epistemic uncertainty.
    """
    state = fit_model(samples, methods, context, origin, transfer=False, engine=engine)
    if state.model is None:
        return pd.DataFrame(), state.info, None
    training = state.training
    concentrations = [s['concentrations_mM'][element] for s in training]
    anchor = copy.deepcopy(context)
    anchor['concentrations_mM'] = {e: float(np.median([s['concentrations_mM'][e] for s in training])) for e in context['elements']}
    anchor['pulse_count'] = int(pulse)
    controlled = context.get('pulse_limit_C') is not None
    temperatures = [s['pulse_limit_C'] for s in training] if controlled else [context['temperature_C']]
    points = []
    for concentration in np.geomspace(min(concentrations), max(concentrations), resolution) if max(concentrations) > min(concentrations) else [concentrations[0]]:
        for temperature in np.linspace(min(temperatures), max(temperatures), resolution) if max(temperatures) > min(temperatures) else [temperatures[0]]:
            point = copy.deepcopy(anchor)
            point['concentrations_mM'][element] = float(concentration)
            if controlled:
                point['pulse_limit_C'] = float(temperature)
            points.append(point)
    encode = transfer_feature if engine == 'adaptive' else feature
    x = np.array([encode(s) for s in points])
    xt = np.array([encode(s) for s in training])
    probabilities = state.model.predict_proba(x)[:, 1]
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    entropy = -(clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped))
    # Kernel-normalized nearest distance shows coverage, not a confidence bound.
    if engine == 'adaptive':
        kernel = state.model.gp.kernel
        similarity = kernel(x, xt) / np.sqrt(kernel.diag(x)[:, None] * kernel.diag(xt)[None, :])
        distances = np.sqrt(np.maximum(2 * (1 - similarity.max(axis=1)), 0))
    else:
        scales = np.array([.65] * len(context['elements']) + [1.] * (xt.shape[1] - len(context['elements'])))
        distances = np.sqrt((((x[:, None, :] - xt[None, :, :]) / scales) ** 2).sum(axis=2)).min(axis=1)
    frame = pd.DataFrame({'concentration_mM': [p['concentrations_mM'][element] for p in points], 'temperature_C': [p['pulse_limit_C'] if controlled else p['temperature_C'] for p in points], 'pulse_count': pulse, 'p_screen_pass': probabilities, 'predictive_entropy_bits': entropy, 'nearest_training_distance': distances})
    if engine == 'adaptive':
        details = state.model.summary(x)
        frame['bald_bits'] = details['bald']
        frame['posterior_interval_width'] = details['high'] - details['low']
    return frame, state.info, anchor


def transfer_comparison(samples, methods, context, origin):
    """Paired target-only/transfer folds with identical test rows and purges.

    Same fixed transfer kernel in both arms isolates addition of source records.
    Lower-stage evaluation uses no equal or higher-stage training compositions.
    """
    eligible = readiness(samples, methods, context, origin, transfer=True)
    ids = set(eligible.loc[eligible['eligible'], 'sample_id'])
    pool = [s for s in samples if s['sample_id'] in ids]
    if not pool:
        return pd.DataFrame(), {'reason': 'No eligible source records.'}
    groups = independent_groups(pool)
    target = np.array([cohort_key(s) == cohort_key(context) for s in pool])
    test_indices = np.flatnonzero(target)
    if len(set(groups[test_indices])) < 3:
        return pd.DataFrame(), {'reason': 'At least three connected target validation groups are required. Repeated recipes can connect several batches.'}
    y = np.array([screen(s, methods)['label'] for s in pool])
    x = np.array([transfer_feature(s) for s in pool])
    rows, skipped = [], 0
    splitter = GroupKFold(n_splits=min(3, len(set(groups[test_indices]))))
    def ready(indices):
        chosen = [pool[i] for i in indices]
        minimum = 8 if origin == 'REAL' else 3
        return len(chosen) >= 12 and len({s['batch_id'] for s in chosen}) >= minimum and all(len({recipe_key(pool[i]) for i in indices if y[i] == label}) >= 3 for label in (0, 1))
    for fold, (_, local_test) in enumerate(splitter.split(x[test_indices], y[test_indices], groups[test_indices]), 1):
        test = test_indices[local_test]
        allowed = ~np.isin(groups, groups[test])
        target_train = np.flatnonzero(allowed & target)
        transfer_train = np.flatnonzero(allowed)
        if not (ready(target_train) and ready(transfer_train)):
            skipped += 1
            continue
        p_target = transfer_classifier().fit(x[target_train], y[target_train]).predict_proba(x[test])[:, 1]
        p_transfer = transfer_classifier().fit(x[transfer_train], y[transfer_train]).predict_proba(x[test])[:, 1]
        lower = np.flatnonzero(allowed & np.array([len(s['elements']) < len(context['elements']) for s in pool]))
        p_lower = transfer_classifier().fit(x[lower], y[lower]).predict_proba(x[test])[:, 1] if ready(lower) else [None] * len(test)
        for i, pt, pm, pl in zip(test, p_target, p_transfer, p_lower):
            rows.append({'sample_id': pool[i]['sample_id'], 'batch_id': pool[i]['batch_id'], 'fold': fold, 'label': int(y[i]), 'p_target_only': float(pt), 'p_transfer': float(pm), 'p_lower_stage_only': None if pl is None else float(pl), 'target_train_n': len(target_train), 'transfer_train_n': len(transfer_train), 'lower_stage_train_n': len(lower), 'target_train_ids': ';'.join(pool[j]['sample_id'] for j in target_train), 'transfer_train_ids': ';'.join(pool[j]['sample_id'] for j in transfer_train)})
    return pd.DataFrame(rows), {'target_records': len(test_indices), 'evaluated': len(rows), 'skipped_folds': skipped, 'reason': 'Paired retrospective validation. Connected synthesis batches and repeated recipes never cross training/test. Not a chronological deployment test.'}


def probability_metrics(frame, probability='p_screen_pass'):
    data = frame.dropna(subset=['label', probability])
    if data.empty:
        return {'n': 0, 'brier': None, 'balanced_accuracy': None, 'false_pass_count': 0}
    y, p = data['label'].to_numpy(), data[probability].to_numpy()
    return {'n': len(data), 'brier': float(brier_score_loss(y, p)), 'balanced_accuracy': float(balanced_accuracy_score(y, p >= .5)) if len(set(y)) == 2 else None, 'false_pass_count': int(((p >= .5) & (y == 0)).sum())}


def prediction_records(samples, methods, recommendations=()):
    """Match historical estimates to current results; reject stale or leaked rows.

    Prospective eligibility requires captured training provenance, pre-result
    timestamp and a new synthesis batch. Historical imports without this audit
    information remain visible but do not enter probability calibration.
    """
    tasks = {r['recommendation_id']: r for r in recommendations}
    rows = []
    for s in samples:
        proposal = s.get('manual_prediction') or s.get('proposal') or tasks.get(s.get('recommendation_id'), {})
        if not proposal:
            continue
        result = screen(s, methods)
        probability = numeric(proposal.get('p_screen_pass'))
        reasons = []
        if probability is None or not 0 <= probability <= 1:
            reasons.append('No valid saved probability')
        if proposal.get('prediction_stale') or s.get('reported_recipe_deviation'):
            reasons.append('Changed or stale recipe')
        recipe = proposal.get('recipe')
        if not recipe or recipe.get('origin') != s.get('origin') or controlled_recipe_key(recipe) != controlled_recipe_key(s):
            reasons.append('Recipe or namespace mismatch')
        if result['label'] is None or result['review_conflicts']:
            reasons.append('Unresolved or conflicting outcome')
        if not actual_batch_recorded(s):
            reasons.append('Actual synthesis batch missing')
        capture = proposal.get('evaluation_capture') or {}
        if not capture or not proposal.get('model_id') or proposal.get('model_id') == 'NO_MODEL':
            reasons.append('Training provenance unavailable')
        if capture.get('rule_version') != RULE_VERSION:
            reasons.append('Screening rule changed or unrecorded')
        if s['batch_id'] in capture.get('training_batch_ids', []):
            reasons.append('Batch already used for training')
        for ref in capture.get('training_records', []):
            if (ref.get('record_uid') and ref.get('record_uid') == s.get('record_uid')) or (not ref.get('record_uid') and ref.get('sample_id') == s['sample_id']):
                reasons.append('Sample already used for training')
                break
        created = pd.to_datetime(proposal.get('created'), utc=True, errors='coerce')
        reported = pd.to_datetime(s.get('result_reported_at'), utc=True, errors='coerce')
        if pd.isna(created) or pd.isna(reported) or created >= reported:
            reasons.append('Pre-result timing unverified')
        from .planning import fingerprint
        rows.append({'sample_id': s['sample_id'], 'batch_id': s['batch_id'], 'composition': composition_name(s), 'element_count': len(s['elements']), 'status': result['status'], 'label': result['label'] if not result['review_conflicts'] else None, 'p_screen_pass': probability, 'p_target_only': numeric(proposal.get('p_target_only')), 'eligible': not reasons, 'exclusion_reason': '; '.join(reasons), 'model_id': proposal.get('model_id'), 'prediction_created': proposal.get('created'), 'result_reported_at': s.get('result_reported_at'), 'policy': proposal.get('policy') or 'CLASSIFICATION_ACTIVE_LEARNING',
                     'recipe_identity': fingerprint(controlled_recipe_key(s)), 'process_identity': fingerprint(process_key(s))})
    return pd.DataFrame(rows, columns=['sample_id', 'batch_id', 'composition', 'element_count', 'status', 'label', 'p_screen_pass', 'p_target_only', 'eligible', 'exclusion_reason', 'model_id', 'prediction_created', 'result_reported_at', 'policy', 'recipe_identity', 'process_identity'])


def calibration_table(frame, bins=5):
    data = frame.loc[frame['eligible']].dropna(subset=['label', 'p_screen_pass']).copy()
    data['probability_bin'] = pd.cut(data['p_screen_pass'], np.linspace(0, 1, bins + 1), include_lowest=True).astype(str)
    rows = []
    for name, group in data.groupby('probability_bin', sort=True):
        low, high = wilson_interval(int(group['label'].sum()), len(group))
        rows.append({'probability_bin': name, 'n': len(group), 'independent_batches': group['batch_id'].nunique(), 'mean_prediction': group['p_screen_pass'].mean(), 'observed_pass_rate': group['label'].mean(), 'lower_95': low, 'upper_95': high})
    return pd.DataFrame(rows, columns=['probability_bin', 'n', 'independent_batches', 'mean_prediction', 'observed_pass_rate', 'lower_95', 'upper_95'])


def learning_progress(samples, methods):
    """Chronological progress requires actual result timestamps, never lexical IDs."""
    frame = analysis_records(samples, methods)
    frame['reported_at'] = pd.to_datetime([s.get('result_reported_at') for s in samples], utc=True, errors='coerce')
    frame = frame.dropna(subset=['reported_at', 'label'])
    frame = frame.loc[~frame['batch_id'].isin(['UNKNOWN', 'UNASSIGNED']) & ~frame['batch_id'].str.startswith(('FILL_IN_', 'REPLACE_'))]
    if frame.empty:
        return pd.DataFrame()
    completion = frame.groupby('batch_id')['reported_at'].max().sort_values()
    rows, seen, n, passed, burned = [], set(), 0, 0, 0
    for iteration, (batch, time) in enumerate(completion.items(), 1):
        group = frame.loc[frame['batch_id'] == batch]
        n += len(group)
        passed += int((group['label'] == 1).sum())
        burned += int((group['status'] == 'BURNED_OUT').sum())
        seen.update(group.loc[group['label'] == 1, 'composition'])
        rows.append({'iteration': iteration, 'batch_id': batch, 'completed_at': time.isoformat(), 'cumulative_resolved_experiments': n, 'cumulative_passes': passed, 'cumulative_burnouts': burned, 'successful_compositions': len(seen), 'batch_resolved': len(group), 'batch_pass_rate': group['label'].mean()})
    return pd.DataFrame(rows)


def interaction_rates(frame, x, y, bins=4):
    """Observed two-factor slices with explicit sample denominators."""
    data = frame.dropna(subset=[x, y, 'label']).copy()
    if data.empty or x == y:
        return pd.DataFrame()
    for variable, name in ((x, 'x_range'), (y, 'y_range')):
        order_column = name.replace('_range', '_order')
        if data[variable].nunique() <= bins:
            data[name] = data[variable].map(lambda v: f'{v:g}')
            data[order_column] = data[variable]
        else:
            ranges = pd.qcut(data[variable], bins, duplicates='drop', precision=12)
            labels = {interval: f'{max(interval.left, data[variable].min()):.12g} to {interval.right:.12g}' for interval in ranges.cat.categories}
            data[name] = ranges.map(labels).astype(str)
            data[order_column] = ranges.map({interval: interval.mid for interval in ranges.cat.categories}).astype(float)
    result = data.groupby(['x_range', 'y_range'], observed=True).agg(n=('label', 'size'), passed=('label', 'sum'), independent_batches=('batch_id', 'nunique'), x_order=('x_order', 'min'), y_order=('y_order', 'min')).reset_index()
    result['pass_rate'] = result['passed'] / result['n']
    result['failed'] = result['n'] - result['passed']
    from .analysis import wilson_interval
    intervals = [wilson_interval(p, n) for p, n in zip(result['passed'], result['n'])]
    result['lower_95'] = [v[0] for v in intervals]
    result['upper_95'] = [v[1] for v in intervals]
    result['cell_label'] = [f'{p/n:.0%} (n={n})' for p, n in zip(result['passed'], result['n'])]
    return result


def strategy_comparison(frame, guided_batches, baseline_batches, baseline_name):
    """Descriptive comparator from disjoint batches within one process cohort."""
    guided, baseline = set(guided_batches), set(baseline_batches)
    if guided & baseline:
        raise ValueError('A synthesis batch cannot belong to both strategies.')
    if frame['cohort'].nunique() != 1:
        raise ValueError('Strategy comparison requires one comparable composition and process.')
    rows = []
    for name, batches in (('Model-guided', guided), (baseline_name, baseline)):
        group = frame.loc[frame['batch_id'].isin(batches)]
        labeled = group.dropna(subset=['label'])
        passed = int((labeled['label'] == 1).sum())
        low, high = wilson_interval(passed, len(labeled))
        rows.append({'strategy': name, 'recorded_samples': len(group), 'resolved': len(labeled), 'synthesis_batches': labeled['batch_id'].nunique(), 'passed': passed, 'burned_out': int((labeled['status'] == 'BURNED_OUT').sum()), 'pass_rate': passed / len(labeled) if len(labeled) else None, 'lower_95': low, 'upper_95': high})
    return pd.DataFrame(rows)
