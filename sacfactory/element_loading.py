"""Strict cross-element loading validation using frozen elemental references."""
import numpy as np
import pandas as pd

from .chemical_descriptors import descriptor_matrix, reference
from .composition import process_key
from .domain import ELEMENTS, recipe_key
from .loading import loading_records, LoadingRegressor, regression_metrics, PREDICTION_COLUMNS
from .planning import fingerprint
from .precursors import precursor_features, base_profile


def single_element_features(sample, profile):
    element = sample['elements'][0]
    if len(sample['elements']) != 1:
        raise ValueError('Cross-element loading phase uses single-element samples only.')
    conditions = [np.log(sample['concentrations_mM'][element] / .05),
                  (sample['pulse_count'] - 1) / 2., (sample['pulse_limit_C'] - 1000) / 200.]
    values = conditions + descriptor_matrix(profile)[ELEMENTS.index(element)].tolist()
    base = base_profile(profile)
    if base in {'chemistry_ion', 'chemistry_precursor', 'chemistry_state'}:
        values += precursor_features(sample, include_state=base == 'chemistry_state',
                                     include_types=base != 'chemistry_ion', for_training=True)
    if profile in {'chemistry_thermal', 'chemistry_loss', 'chemistry_mixing', 'chemistry_pulseicp'}:
        from .physics_features import thermal_recipe_features
        from .preparation import model_features as preparation_features
        values += thermal_recipe_features(sample)
        values += preparation_features(sample)
    if profile == 'chemistry_loss':
        from .physics_features import loss_features
        values += loss_features(sample)
    if profile in {'chemistry_mixing', 'chemistry_pulseicp'}:
        from .mixing import feature_vector, component_features
        values += feature_vector(sample) + component_features(sample, element)
    if profile == 'chemistry_pulseicp':
        from .pulseicp_bridge import recipe_features
        values += recipe_features(sample)
    return values


def cross_element_loading(samples, origin, target, context, sources=(), profile='chemistry_ion', family='Gaussian process', physics=None):
    singles = [s for s in samples if s.get('origin') == origin and len(s['elements']) == 1 and
               (not sources or s['elements'][0] in set(sources) | {target})]
    records, audits = [], []
    for element in dict.fromkeys(s['elements'][0] for s in singles):
        valid, audit = loading_records(singles, origin, element, profile)
        audits.append(audit)
        records.extend((s, y, single_element_features(s, profile)) for s, y, _ in valid if process_key(s) == process_key(context))
    from .physics_features import prepare_loading_priors, compare_loading_predictions, physics_manifest
    records, prior_by_id, physics_audit = prepare_loading_priors(records, target, physics, per_member=True)
    test = [(s, y, x) for s, y, x in records if s['elements'] == [target]]
    batches = {s['batch_id'] for s in samples if s.get('origin') == origin and target in s['elements']}
    train = [(s, y, x) for s, y, x in records if s['elements'] != [target] and s['batch_id'] not in batches]
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    info = {'protocol': 'STRICT_CROSS_ELEMENT_LOADING', 'origin': origin, 'target': target, 'profile': profile,
        'model': family, 'training_n': len(train), 'test_n': len(test),
        'train_elements': sorted({s['elements'][0] for s, _, _ in train}),
        'train_sample_ids': [s['sample_id'] for s, _, _ in train], 'test_sample_ids': [s['sample_id'] for s, _, _ in test],
        'train_batch_ids': sorted({s['batch_id'] for s, _, _ in train}), 'test_batch_ids': sorted(batches),
        'dataset_fingerprint': fingerprint(samples), 'descriptor_snapshot_sha256': reference()['snapshot_sha256'],
        'note': 'No target-element or mixture training rows. All shared target batches are purged. Features use controls and frozen source-backed element/state references; constant source descriptors are retained so unseen chemistry is not erased. One source element cannot identify cross-element response effects. Extrapolation and conditional GP uncertainty are not guaranteed reliability.',
        'physics': physics_manifest(profile), 'physical_residual': physics_audit, 'metrics': None}
    if not test or len(train) < 8 or len({s['batch_id'] for s, _, _ in train}) < 2 or len({recipe_key(s) for s, _, _ in train}) < 3:
        info['reason'] = 'Need target measured W and 8 source quantified samples across 2 batches and 3 recipes.'
        return pd.DataFrame(columns=PREDICTION_COLUMNS), audit, info
    model = LoadingRegressor(family, preserve_constant=True).fit([x for _, _, x in train], [y for _, y, _ in train])
    direct, low, high, outside = model.predict([x for _, _, x in test])
    predicted = direct
    if physics is not None:
        residual = LoadingRegressor(family, preserve_constant=True).fit([x for _, _, x in train],
            [y for _, y, _ in train], [prior_by_id[s['sample_id']] for s, _, _ in train])
        predicted, low, high, outside = residual.predict([x for _, _, x in test], [prior_by_id[s['sample_id']] for s, _, _ in test])
    rows = []
    for (s, y, _), p, lo, hi, out in zip(test, predicted, low, high, outside):
        rows.append({'sample_id': s['sample_id'], 'batch_id': s['batch_id'], 'element': target, 'composition': target,
            'element_count': 1, 'fold': 1, 'observed_w_wt_percent': y, 'predicted_w_wt_percent': float(p),
            'lower_95_w_wt_percent': float(lo) if np.isfinite(lo) else None,
            'upper_95_w_wt_percent': float(hi) if np.isfinite(hi) else None,
            'training_n': len(train), 'outside_training_range': bool(out)})
    if physics is not None:
        for row, baseline in zip(rows, direct):
            row.update(direct_predicted_w_wt_percent=float(baseline), physical_prior_w_wt_percent=prior_by_id[row['sample_id']])
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS +
        (['direct_predicted_w_wt_percent', 'physical_prior_w_wt_percent'] if physics is not None else []))
    info['paired_comparison'] = compare_loading_predictions(frame)
    info['metrics'] = regression_metrics(frame)
    return frame, audit, info


def combined_element_loading(samples, origin, targets, sources=(), profile='chemistry_ion', family='Gaussian process', physics=None):
    """One held-out-element fit per compatible context; no invented empty targets."""
    targets = list(dict.fromkeys(targets))
    if not targets or set(targets) - set(ELEMENTS):
        raise ValueError('Select at least one valid element.')
    points, statuses, provenance = [], [], []
    for element in targets:
        contexts = {}
        for sample in samples:
            if sample.get('origin') == origin and sample['elements'] == [element]:
                contexts.setdefault(process_key(sample), sample)
        if not contexts:
            statuses.append({'element': element, 'process_context': '', 'plotted_samples': 0,
                'status': 'No single-element target records', 'mae_wt_percent': None, 'rmse_wt_percent': None, 'r_squared': None})
        for key, context in contexts.items():
            frame, _, info = cross_element_loading(samples, origin, element, context, sources, profile, family, physics)
            identity = fingerprint(key)
            provenance.append({'element': element, 'process_context': identity, **info})
            if not frame.empty:
                points.append(frame.assign(process_context=identity))
            statuses.append({'element': element, 'process_context': identity, 'plotted_samples': len(frame),
                'status': 'Evaluated' if len(frame) else info['reason'],
                **{name: (info['metrics'] or {}).get(name) for name in ('mae_wt_percent', 'rmse_wt_percent', 'r_squared')}})
    combined = pd.concat(points, ignore_index=True) if points else pd.DataFrame(columns=PREDICTION_COLUMNS + ['process_context'])
    report = pd.DataFrame(statuses)
    info = {'protocol': 'COMBINED_STRICT_CROSS_ELEMENT_LOADING', 'origin': origin, 'selected_elements': targets,
            'source_filter': list(sources), 'model': family, 'profile': profile,
            'metrics': regression_metrics(combined), 'fits': provenance,
            'dataset_fingerprint': fingerprint(samples),
            'physical_residual_assumptions': physics,
            'note': 'Each point predicts one element in one held-out single-element sample. Entire target elements and their shared batches are excluded from that fit. Process contexts stay separate. Pooled metrics are sample-weighted; per-element/context errors must also be inspected. Missing targets do not become synthetic points.'}
    from .physics_features import compare_loading_predictions
    info['paired_comparison'] = compare_loading_predictions(combined)
    return combined, report, info
