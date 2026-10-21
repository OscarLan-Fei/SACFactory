"""Protected tabular challengers, beta calibration and descriptor ablations."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .adaptive import eligible_records, AdaptivePredictor, scores
from .chemical_descriptors import model_features, reference
from .composition import composition_name, transfer_feature
from .domain import ELEMENTS, screen
from .reliable_transfer import ReliabilityEnsemble, ready


class BetaCalibrator:
    """Monotone beta probability map; fitted only on reserved calibration batches."""
    def fit(self, probability, y):
        p = np.clip(np.asarray(probability), 1e-8, 1-1e-8)
        design = np.column_stack([np.log(p), np.log1p(-p), np.ones(len(p))])
        initial = np.array([1., -1., 0.])
        def objective(theta):
            q = np.clip(expit(design @ theta), 1e-8, 1-1e-8)
            return float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q)) + .02*np.sum((theta-initial)**2))
        result = minimize(objective, initial, method='L-BFGS-B', bounds=[(0., 8.), (-8., 0.), (-5., 5.)])
        self.theta = result.x if result.success and result.fun <= objective(initial) else initial
        self.diagnostics = {'initial_objective': objective(initial), 'final_objective': objective(self.theta),
                            'parameters': self.theta.tolist(), 'optimizer_converged': bool(result.success)}
        return self

    def transform(self, probability):
        p = np.clip(np.asarray(probability), 1e-8, 1-1e-8)
        return expit(self.theta[0]*np.log(p)+self.theta[1]*np.log1p(-p)+self.theta[2])


class TabularCandidate:
    def __init__(self, family='logistic', profile='chemistry', calibration=False):
        self.family, self.profile, self.calibration = family, profile, calibration

    def fit(self, x, y, groups):
        x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
        fit, holdout = np.arange(len(y)), np.array([], dtype=int)
        if self.calibration and len(y) >= 80 and len(set(groups)) >= 8:
            train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(x, y, groups))
            if len(test) >= 24 and min(np.bincount(y[test], minlength=2)) >= 6 and min(np.bincount(y[train], minlength=2)) >= 3:
                fit, holdout = train, test
        if self.family == 'logistic':
            self.model = make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=500, random_state=42))
        elif self.family == 'forest':
            self.model = RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=1)
        elif self.family == 'boosting':
            self.model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, min_samples_leaf=5, l2_regularization=2., random_state=42)
        else:
            raise ValueError('Select logistic, forest or boosting candidate.')
        self.model.fit(model_features(x[fit], self.profile), y[fit])
        self.calibrator = None
        if len(holdout):
            self.calibrator = BetaCalibrator().fit(self.model.predict_proba(model_features(x[holdout], self.profile))[:, 1], y[holdout])
        self.calibration_members = x[holdout, :len(ELEMENTS)].copy()
        self.provenance = {'fit_batch_ids': sorted(set(groups[fit])), 'calibration_batch_ids': sorted(set(groups[holdout])),
            'calibration_status': 'BETA_HOLDOUT_FIT' if self.calibrator is not None else 'UNCALIBRATED',
            'calibration': self.calibrator.diagnostics if self.calibrator else None}
        return self

    def predict_proba(self, x):
        p = self.model.predict_proba(model_features(x, self.profile))[:, 1]
        if self.calibrator is not None:
            mask = np.any(np.all(np.asarray(x)[:, None, :len(ELEMENTS)] == self.calibration_members[None, :, :], axis=2), axis=1)
            p[mask] = self.calibrator.transform(p[mask])
        return np.column_stack([1-p, p])


def reliability_benchmark(samples, methods, context, origin, iterations=4, max_folds=3,
                          protocol='target_batch', element=None, batch_effect=True):
    records, exclusions = eligible_records(samples, methods, context, origin, True)
    target = [s for s in records if composition_name(s) == composition_name(context)]
    splits, missing = [], []
    if protocol == 'unseen_element':
        test = [s for s in records if element in s['elements']]
        batches = {s['batch_id'] for s in test}
        splits = [([s for s in records if element not in s['elements'] and s['batch_id'] not in batches], test)] if test else []
    elif protocol == 'forward':
        times = {}
        for batch in sorted({s['batch_id'] for s in records}):
            dates = pd.to_datetime([s.get('result_reported_at') for s in records if s['batch_id'] == batch], utc=True, errors='coerce')
            if dates.isna().any():
                missing.append(batch)
            else:
                times[batch] = (dates.min(), dates.max())
        for batch in sorted({s['batch_id'] for s in target if s['batch_id'] in times}, key=lambda b: times[b][0]):
            start = times[batch][0]
            train = [s for s in records if s['batch_id'] in times and times[s['batch_id']][1] < start]
            test = [s for s in target if s['batch_id'] == batch]
            if ready(train, methods):
                splits.append((train, test))
        if len(splits) > max_folds:
            splits = [splits[i] for i in np.unique(np.linspace(0, len(splits)-1, max_folds, dtype=int))]
    elif protocol == 'target_batch':
        groups = [s['batch_id'] for s in target]
        if len(set(groups)) >= 3:
            for _, test_indices in GroupKFold(n_splits=min(max(2, max_folds), len(set(groups)), 6)).split(target, groups=groups):
                test = [target[i] for i in test_indices]
                batches = {s['batch_id'] for s in test}
                splits.append(([s for s in records if s['batch_id'] not in batches], test))
    else:
        raise ValueError('Select target_batch, forward or unseen_element benchmark.')
    predictions, provenance, skipped = [], [], []
    names = ['Adaptive GP / periodic', 'Reliability / periodic', 'Reliability / chemistry',
             'Logistic / chemistry', 'Random forest / chemistry', 'Boosting + beta calibration / chemistry']
    for fold, (train, test) in enumerate(splits):
        if not ready(train, methods):
            skipped.append({'fold': fold, 'reason': 'Source training diversity is insufficient'})
            continue
        x = np.array([transfer_feature(s) for s in train]); y = np.array([screen(s, methods)['label'] for s in train])
        groups = [s['batch_id'] for s in train]
        xt = np.array([transfer_feature(s) for s in test])
        pooled = AdaptivePredictor(iterations, batch_effect=batch_effect).fit(x, y, groups)
        periodic = ReliabilityEnsemble().fit(train, methods, context, pooled, iterations, batch_effect)
        chemical_gp = AdaptivePredictor(iterations, batch_effect=batch_effect, descriptor_profile='chemistry').fit(x, y, groups)
        chemistry = ReliabilityEnsemble().fit(train, methods, context, chemical_gp, iterations, batch_effect, 'chemistry')
        models = [pooled, periodic, chemistry, *[TabularCandidate(family, calibration=family == 'boosting').fit(x, y, groups)
                   for family in ('logistic', 'forest', 'boosting')]]
        provenance.append({'fold': fold, 'train_batch_ids': sorted(set(groups)),
            'test_batch_ids': sorted({s['batch_id'] for s in test}), 'train_sample_ids': [s['sample_id'] for s in train],
            'test_sample_ids': [s['sample_id'] for s in test],
            'latest_train_result': max(str(s.get('result_reported_at') or '') for s in train),
            'first_test_result': min(str(s.get('result_reported_at') or '') for s in test),
            'weight_fits': [periodic.weight_diagnostics, chemistry.weight_diagnostics],
            'tabular_models': [model.provenance for model in models[3:]]})
        for name, model in zip(names, models):
            p = model.predict_proba(xt)[:, 1]
            predictions.extend({'model': name, 'sample_id': s['sample_id'], 'batch_id': s['batch_id'],
                'composition': composition_name(s), 'label': screen(s, methods)['label'], 'p_screen_pass': float(value),
                'fold': fold} for s, value in zip(test, p))
    frame = pd.DataFrame(predictions, columns=['model', 'sample_id', 'batch_id', 'composition', 'label', 'p_screen_pass', 'fold'])
    comparison = [{'model': name, **scores(part.label.to_numpy(), part.p_screen_pass.to_numpy())}
                  for name, part in frame.groupby('model')] if not frame.empty else []
    info = {'protocol': protocol, 'held_out_element': element if protocol == 'unseen_element' else None,
        'comparison': comparison, 'folds': provenance, 'skipped': skipped, 'exclusions': exclusions,
        'batches_missing_timestamps': missing, 'descriptor_snapshot_sha256': reference()['snapshot_sha256'],
        'note': 'Identical outer test rows; weights and calibration refit inside each training fold. Exploratory model/descriptor comparison, no automatic promotion. Forward mode replays corrected data, not historical revisions.'}
    return frame, info
