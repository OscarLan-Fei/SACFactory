"""Element-specific ICP mass-content regression with protected batch validation.

This auxiliary target never changes screening labels or acquisition priorities.
Only quantified mass percentages train regressors; censoring stays in the audit.
"""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .chemical_descriptors import model_features, reference
from .composition import process_key, transfer_feature, composition_name
from .domain import ELEMENTS, INVALID_RUN_CONDITIONS, parse_measurement, recipe_key
from .manual import actual_batch_recorded
from .planning import fingerprint
from .precursors import documented, state_reference
from .units import normalize_assay

MODEL_VERSION = 'icp-loading-regression-v2-physical-residual'
MODELS = ['Gaussian process', 'Random forest', 'Ridge baseline']
PROTOCOLS = ['Batch holdout', 'New recipe holdout', 'Chronological replay', 'Single to multiple elements']
PREDICTION_COLUMNS = ['sample_id', 'batch_id', 'element', 'composition', 'element_count', 'fold',
                      'observed_w_wt_percent', 'predicted_w_wt_percent', 'lower_95_w_wt_percent',
                      'upper_95_w_wt_percent', 'training_n', 'outside_training_range']


def controls(sample, profile, target_element=None):
    """Use only pre-synthesis conditions and recorded, source-gated chemistry."""
    if sample.get('pulse_limit_C') is None or not np.isfinite(float(sample['pulse_limit_C'])):
        raise ValueError('Controlled cutoff temperature missing')
    if float(sample.get('pulse_count') or 0) < 1 or not float(sample['pulse_count']).is_integer():
        raise ValueError('Positive integer pulse count required')
    x = transfer_feature(sample, descriptor_profile=profile, for_training=True)
    x = model_features([x], profile)[0]
    if profile in {'chemistry_mixing', 'chemistry_pulseicp'} and target_element is not None:
        from .mixing import component_features
        x = np.r_[x, component_features(sample, target_element)]
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite synthesis conditions')
    return x


def loading_records(samples, origin, element, profile='chemistry_ion'):
    if origin not in {'REAL', 'DEMO'} or element not in ELEMENTS:
        raise ValueError('Select a valid data space and element.')
    audit, eligible = [], []
    for sample in samples:
        if sample.get('origin') != origin or element not in sample['elements']:
            continue
        value, upper, reason, x = None, None, '', None
        try:
            assays = [normalize_assay(a) for a in sample.get('icp', []) if a.get('element') == element]
            if len(assays) != 1:
                raise ValueError('One unambiguous element measurement required')
            assay = assays[0]
            reading = parse_measurement(assay.get('value', ''))
            upper = reading.get('upper')
            if assay.get('flag', 'MEASURED') != 'MEASURED' or reading['kind'] != 'measured':
                raise ValueError('Unquantified or censored ICP; not imputed')
            value = reading['value']
            if not 0 <= value <= 100:
                raise ValueError('Mass percentage outside 0 to 100')
            if sample.get('sample_condition') in INVALID_RUN_CONDITIONS | {'BURNED_OUT'}:
                raise ValueError('No valid recovered synthesis specimen')
            if not actual_batch_recorded(sample):
                raise ValueError('Actual synthesis batch missing')
            if any(not documented(sample.get(k)) for k in ('precursor_family', 'support_batch', 'protocol_id')):
                raise ValueError('Process context missing')
            x = controls(sample, profile, element)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            reason = str(exc)
        audit.append({'sample_id': sample['sample_id'], 'batch_id': sample.get('batch_id'), 'element': element,
                      'w_wt_percent': value, 'censor_upper_w_wt_percent': upper,
                      'eligible': not reason, 'exclusion_reason': reason})
        if not reason:
            eligible.append((sample, value, x))
    return eligible, pd.DataFrame(audit, columns=['sample_id', 'batch_id', 'element', 'w_wt_percent',
                         'censor_upper_w_wt_percent', 'eligible', 'exclusion_reason'])


class LoadingRegressor:
    def __init__(self, family='Gaussian process', preserve_constant=False):
        if family not in MODELS:
            raise ValueError('Select a supported loading model.')
        self.family = family
        self.preserve_constant = preserve_constant

    def fit(self, x, y, prior=None):
        x, y = np.asarray(x), np.asarray(y)
        # All transformations and active columns are learned inside training only.
        self.active = np.ones(x.shape[1], dtype=bool) if self.preserve_constant else np.ptp(x, axis=0) > 1e-12
        self.scaler = StandardScaler().fit(x[:, self.active])
        scaled = self.scaler.transform(x[:, self.active])
        self.minimum, self.maximum = x.min(axis=0), x.max(axis=0)
        if self.family == 'Gaussian process':
            kernel = RBF(np.sqrt(scaled.shape[1]), length_scale_bounds='fixed') + WhiteKernel(.1, noise_level_bounds='fixed')
            self.model = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True, random_state=42)
        elif self.family == 'Random forest':
            self.model = RandomForestRegressor(n_estimators=120, min_samples_leaf=3, max_depth=10, random_state=42, n_jobs=1)
        else:
            self.model = Ridge(alpha=10.)
        self.residual = prior is not None
        offset = self.offset(prior, len(y)) if self.residual else np.zeros(len(y))
        self.model.fit(scaled, np.log1p(y) - offset)
        return self

    @staticmethod
    def offset(prior, count):
        values = np.asarray(prior, dtype=float)
        if values.shape != (count,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
            raise ValueError('A valid pre-synthesis physical prior is required for every row.')
        return np.log1p(values)

    def predict(self, x, prior=None):
        x = np.asarray(x)
        offset = self.offset(prior, len(x)) if self.residual else np.zeros(len(x))
        scaled = self.scaler.transform(x[:, self.active])
        if self.family == 'Gaussian process':
            mean, sd = self.model.predict(scaled, return_std=True)
            mean += offset
            low, high = np.clip(np.expm1(mean - 1.96 * sd), 0, 100), np.clip(np.expm1(mean + 1.96 * sd), 0, 100)
        else:
            mean = self.model.predict(scaled) + offset
            low, high = np.full(len(x), np.nan), np.full(len(x), np.nan)
        return np.clip(np.expm1(mean), 0, 100), low, high, np.any((x < self.minimum - 1e-9) | (x > self.maximum + 1e-9), axis=1)


def regression_metrics(frame):
    if frame.empty:
        return {'n': 0, 'mae_wt_percent': None, 'rmse_wt_percent': None, 'r_squared': None, 'interval_coverage': None}
    y, p = frame.observed_w_wt_percent, frame.predicted_w_wt_percent
    intervals = frame.dropna(subset=['lower_95_w_wt_percent', 'upper_95_w_wt_percent'])
    return {'n': len(frame), 'mae_wt_percent': float(mean_absolute_error(y, p)),
            'rmse_wt_percent': float(np.sqrt(mean_squared_error(y, p))),
            'r_squared': float(r2_score(y, p)) if len(y) >= 2 and np.ptp(y) > 1e-12 else None,
            'interval_coverage': float(((intervals.observed_w_wt_percent >= intervals.lower_95_w_wt_percent) &
                (intervals.observed_w_wt_percent <= intervals.upper_95_w_wt_percent)).mean()) if len(intervals) else None}


def loading_validation(samples, origin, element, context, family='Gaussian process',
                       profile='chemistry_ion', protocol='Batch holdout', max_folds=3, physics=None):
    if protocol not in PROTOCOLS:
        raise ValueError('Select a supported loading validation protocol.')
    max_folds = max(2, min(6, int(max_folds)))
    records, audit = loading_records(samples, origin, element, profile)
    selected = [(s, y, x) for s, y, x in records if process_key(s) == process_key(context)]
    mismatched = {s['sample_id'] for s, _, _ in records if process_key(s) != process_key(context)}
    if not audit.empty:
        audit.loc[audit.sample_id.isin(mismatched), ['eligible', 'exclusion_reason']] = [False, 'Different process context']
    from .physics_features import prepare_loading_priors, compare_loading_predictions, physics_manifest
    selected, prior_by_id, physics_audit = prepare_loading_priors(selected, element, physics)
    splits, omitted = [], []
    if len({s['batch_id'] for s, _, _ in selected}) >= 3:
        batches = np.array([s['batch_id'] for s, _, _ in selected])
        if protocol == 'Single to multiple elements':
            test = [i for i, (s, _, _) in enumerate(selected) if len(s['elements']) > 1]
            test_batches = {batches[i] for i in test}
            train = [i for i, (s, _, _) in enumerate(selected) if len(s['elements']) == 1 and batches[i] not in test_batches]
            if test:
                splits = [(train, test)]
        elif protocol == 'Chronological replay':
            times = {}
            for batch in set(batches):
                dates = pd.to_datetime([s.get('result_reported_at') for s, _, _ in selected if s['batch_id'] == batch], utc=True, errors='coerce')
                if dates.isna().any():
                    omitted.append(batch)
                else:
                    times[batch] = (dates.min(), dates.max())
            for batch in sorted(times, key=lambda b: times[b][0]):
                train = [i for i, b in enumerate(batches) if b in times and times[b][1] < times[batch][0]]
                splits.append((train, list(np.flatnonzero(batches == batch))))
            if len(splits) > max_folds:
                splits = [splits[i] for i in np.unique(np.linspace(0, len(splits)-1, max_folds, dtype=int))]
        else:
            splits = list(GroupKFold(n_splits=min(max_folds, len(set(batches)))).split(selected, groups=batches))
    rows, folds, skipped = [], [], []
    for fold, (train, test) in enumerate(splits, 1):
        if protocol == 'New recipe holdout':
            recipes = {recipe_key(selected[i][0]) for i in test}
            train = [i for i in train if recipe_key(selected[i][0]) not in recipes]
        source = [selected[i] for i in train]
        if len(source) < 8 or len({s['batch_id'] for s, _, _ in source}) < 2 or len({recipe_key(s) for s, _, _ in source}) < 3:
            skipped.append({'fold': fold, 'reason': 'Training needs 8 quantified samples, 2 batches and 3 recipes'})
            continue
        if not np.any(np.ptp(np.asarray([x for _, _, x in source]), axis=0) > 1e-12):
            skipped.append({'fold': fold, 'reason': 'Training controls and enabled descriptors are constant'})
            continue
        model = LoadingRegressor(family).fit([x for _, _, x in source], [y for _, y, _ in source])
        target = [selected[i] for i in test]
        direct, low, high, outside = model.predict([x for _, _, x in target])
        pred = direct
        if physics is not None:
            train_prior = [prior_by_id[s['sample_id']] for s, _, _ in source]
            test_prior = [prior_by_id[s['sample_id']] for s, _, _ in target]
            residual = LoadingRegressor(family).fit([x for _, _, x in source], [y for _, y, _ in source], train_prior)
            pred, low, high, outside = residual.predict([x for _, _, x in target], test_prior)
        folds.append({'fold': fold, 'train_sample_ids': [s['sample_id'] for s, _, _ in source],
            'test_sample_ids': [s['sample_id'] for s, _, _ in target],
            'train_batch_ids': sorted({s['batch_id'] for s, _, _ in source}),
            'test_batch_ids': sorted({s['batch_id'] for s, _, _ in target})})
        for (s, y, _), p, lo, hi, out in zip(target, pred, low, high, outside):
            rows.append({'sample_id': s['sample_id'], 'batch_id': s['batch_id'], 'element': element,
                'composition': composition_name(s), 'element_count': len(s['elements']), 'fold': fold,
                'observed_w_wt_percent': y, 'predicted_w_wt_percent': float(p),
                'lower_95_w_wt_percent': float(lo) if np.isfinite(lo) else None,
                'upper_95_w_wt_percent': float(hi) if np.isfinite(hi) else None,
                'training_n': len(source), 'outside_training_range': bool(out)})
        if physics is not None:
            for row, baseline in zip(rows[-len(target):], direct):
                row.update(direct_predicted_w_wt_percent=float(baseline),
                           physical_prior_w_wt_percent=prior_by_id[row['sample_id']])
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS +
        (['direct_predicted_w_wt_percent', 'physical_prior_w_wt_percent'] if physics is not None else []))
    stages = [{'element_count': int(n), **regression_metrics(part)} for n, part in predictions.groupby('element_count')]
    info = {'version': MODEL_VERSION, 'origin': origin, 'element': element, 'model': family, 'profile': profile,
        'protocol': protocol, 'eligible_n': len(selected), 'metrics': regression_metrics(predictions), 'stage_metrics': stages,
        'folds': folds, 'skipped_folds': skipped, 'batches_missing_timestamps': omitted,
        'exclusions': dict(Counter(audit.loc[~audit.eligible, 'exclusion_reason'])) if not audit.empty else {},
        'dataset_fingerprint': fingerprint(samples), 'descriptor_snapshot_sha256': reference()['snapshot_sha256'],
        'state_snapshot_sha256': state_reference()['snapshot_sha256'],
        'interval_note': 'GP conditional 95% observation intervals in log1p(W), transformed and physically clipped; fixed kernel/noise, no hyperparameter uncertainty or guaranteed coverage. Other models have no interval.',
        'physics': physics_manifest(profile), 'physical_residual': physics_audit,
        'paired_comparison': compare_loading_predictions(predictions),
        'target_note': 'Quantified element mass content, not single-atom fraction or recovery yield. Censored results are audited but excluded; evaluation is conditional on quantification. No loading maximization or retrospective ICP input features.'}
    return predictions, audit, info
