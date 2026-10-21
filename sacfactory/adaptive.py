"""Learned hierarchical Bernoulli GP with protected calibration and validation.

Laplace inference and regularized empirical-Bayes kernel learning are explicit.
Posterior intervals condition on fitted hyperparameters, not full Bayes.
"""
from collections import Counter
from itertools import combinations
import math

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .composition import descriptors, transfer_feature, composition_name, process_key
from .domain import ELEMENTS, RULE_VERSION, cohort_key, recipe_key, screen


VERSION = 'adaptive-hierarchical-laplace-v4-representative-expert'
INITIAL = np.log([.45, .20, .35, .65, .9, 1., 1., 1., 1.])
NAMES = ['member_amplitude', 'pair_amplitude', 'composition_amplitude', 'concentration_scale',
         'total_scale', 'cardinality_scale', 'pulse_scale', 'cutoff_scale', 'descriptor_scale']
POSITIONS = np.array([descriptors(e) for e in ELEMENTS])
NODES, NODE_WEIGHTS = hermgauss(24)


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class HierarchicalKernel:
    """PSD learned member/pair/composition covariance times process covariance."""
    def __init__(self, theta=None, positions=None):
        self.theta = INITIAL.copy() if theta is None else np.asarray(theta, dtype=float).copy()
        self.positions = POSITIONS if positions is None else np.asarray(positions)

    def __call__(self, x, y=None):
        x = np.atleast_2d(x)
        y = x if y is None else np.atleast_2d(y)
        n = len(ELEMENTS)
        a, b = x[:, :n], y[:, :n]
        w = np.exp(self.theta)
        atom = .5 * np.exp(-.5 * cdist(self.positions, self.positions, 'sqeuclidean') / w[8] ** 2) + .5 * np.eye(n)
        norms_a = np.sqrt(np.sum((a @ atom) * a, axis=1))
        norms_b = np.sqrt(np.sum((b @ atom) * b, axis=1))
        members = (a @ atom @ b.T) / (norms_a[:, None] * norms_b[None, :])
        na, nb, overlap = a.sum(axis=1), b.sum(axis=1), a @ b.T
        denominator = np.sqrt((na * (na - 1) / 2)[:, None] * (nb * (nb - 1) / 2)[None, :])
        pairs = np.divide(overlap * (overlap - 1) / 2, denominator, out=np.zeros_like(overlap), where=denominator > 0)
        exact = ((overlap == na[:, None]) & (overlap == nb[None, :])).astype(float)
        concentrations = np.exp(-.5 * cdist(x[:, n:2*n], y[:, n:2*n], 'sqeuclidean') / w[3] ** 2)
        process_end = min(x.shape[1], 2*n+4)
        scales = w[4:4 + process_end - 2*n]
        distance = np.sqrt(np.maximum(cdist(x[:, 2*n:process_end] / scales, y[:, 2*n:process_end] / scales, 'sqeuclidean'), 0))
        # Matern 5/2 process covariance permits less smooth response windows.
        process = (1 + np.sqrt(5) * distance + 5 * distance ** 2 / 3) * np.exp(-np.sqrt(5) * distance)
        precursor = np.exp(-.5 * cdist(x[:, process_end:], y[:, process_end:], 'sqeuclidean') / w[8]**2) if x.shape[1] > process_end else 1.
        return concentrations * process * precursor * (w[0] * members + w[1] * pairs + w[2] * exact)

    def diag(self, x):
        w = np.exp(self.theta)
        return w[0] + w[2] + w[1] * (np.asarray(x)[:, :len(ELEMENTS)].sum(axis=1) > 1)

    def parameters(self):
        return dict(zip(NAMES, map(float, np.exp(self.theta))))


def laplace_mode(kernel, x, y, max_iter=60, batch_covariance=None):
    k = kernel(x) + np.eye(len(x)) * 1e-7
    if batch_covariance is not None:
        k = k + batch_covariance
    f = np.zeros(len(x))
    converged = False
    for iteration in range(max_iter):
        pi = expit(f)
        root = np.sqrt(np.maximum(pi * (1 - pi), 1e-10))
        factor = cholesky(np.eye(len(x)) + root[:, None] * k * root[None, :], lower=True)
        b = root ** 2 * f + y - pi
        a = b - root * cho_solve((factor, True), root * (k @ b))
        proposed = k @ a
        if np.max(np.abs(proposed - f)) < 1e-7:
            converged = True
            f = proposed
            break
        f = proposed
    pi = expit(f)
    root = np.sqrt(np.maximum(pi * (1 - pi), 1e-10))
    factor = cholesky(np.eye(len(x)) + root[:, None] * k * root[None, :], lower=True)
    alpha = y - pi
    evidence = -.5 * alpha @ f + y @ f - np.logaddexp(0, f).sum() - np.log(np.diag(factor)).sum()
    return {'alpha': alpha, 'root': root, 'factor': factor, 'evidence': float(evidence),
            'converged': converged, 'iterations': iteration + 1}


class LearnedGP:
    def __init__(self, max_iterations=18, seed=42, descriptor_profile='periodic'):
        self.max_iterations, self.seed = max_iterations, seed
        self.descriptor_profile = descriptor_profile

    def fit(self, x, y, groups=None, batch_effect=True):
        from .chemical_descriptors import descriptor_matrix
        positions = descriptor_matrix(self.descriptor_profile)
        self.x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(np.unique(y)) != 2:
            raise ValueError('The learned classifier requires both outcome classes.')
        groups = np.asarray(groups if groups is not None else [str(i) for i in range(len(y))])
        if len(groups) != len(y):
            raise ValueError('Batch identifiers must align with training rows.')
        _, recipes = np.unique(self.x, axis=0, return_inverse=True)
        repeated = sum(len(set(groups[recipes == i])) >= 2 for i in set(recipes))
        self.batch_enabled = bool(batch_effect and len(set(groups)) >= 4 and repeated >= 2)
        initial = np.append(INITIAL, np.log(.25)) if self.batch_enabled else INITIAL
        # Hyperparameter budget uses a declared stratified subset; final inference uses all fit rows.
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(y))
        if len(indices) > 128:
            parts = [rng.choice(np.flatnonzero(y == label), min(64, int((y == label).sum())), replace=False) for label in (0, 1)]
            indices = np.sort(np.concatenate(parts))
        hx, hy = self.x[indices], y[indices]
        same_batch = (groups[indices, None] == groups[None, indices]).astype(float)
        bounds = [(math.log(.025), math.log(5.))] * 3 + [(math.log(.15), math.log(5.))] * 5 + [(math.log(.25), math.log(4.))]
        if self.batch_enabled:
            bounds.append((math.log(.001), math.log(3.)))

        def loss(theta):
            try:
                covariance = np.exp(theta[9]) * same_batch if self.batch_enabled else None
                result = laplace_mode(HierarchicalKernel(theta[:9], positions), hx, hy, batch_covariance=covariance)
                if not result['converged']:
                    return 1e6
                return -result['evidence'] + .35 * np.sum((theta - initial) ** 2)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                return 1e6

        initial_loss = loss(initial)
        runs = []
        for start in (initial, np.clip(initial + rng.normal(0, .18, len(initial)), *np.array(bounds).T)):
            optimized = minimize(loss, start, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': self.max_iterations, 'maxfun': self.max_iterations * 18, 'ftol': 1e-6})
            runs.append(optimized)
        best = min(runs, key=lambda r: r.fun)
        theta = best.x if np.isfinite(best.fun) and best.fun <= initial_loss else initial
        self.kernel = HierarchicalKernel(theta[:9], positions)
        self.batch_variance = float(np.exp(theta[9])) if self.batch_enabled else 0.
        covariance = self.batch_variance * (groups[:, None] == groups[None, :]) if self.batch_enabled else None
        self.posterior = laplace_mode(self.kernel, self.x, y, batch_covariance=covariance)
        if not self.posterior['converged']:
            raise ValueError('The Laplace posterior did not converge. Use the baseline or revise the training pool.')
        self.diagnostics = {'parameters': self.kernel.parameters(), 'initial_parameters': HierarchicalKernel().parameters(),
                            'initial_regularized_loss': float(initial_loss), 'final_regularized_loss': float(loss(theta)),
                            'optimization_rows': len(indices), 'fit_rows': len(y), 'optimizer_converged': bool(best.success),
                            'optimizer_message': str(best.message), 'laplace_iterations': self.posterior['iterations'],
                            'hyperparameter_restarts': 2, 'regularization_strength': .35}
        self.diagnostics['descriptor_profile'] = self.descriptor_profile
        self.diagnostics['batch_effect'] = {'enabled': self.batch_enabled, 'variance': self.batch_variance,
            'independent_batches': len(set(groups)), 'cross_batch_repeated_recipes': int(repeated),
            'note': 'Shared random intercept; new-batch uncertainty is integrated. Requires four batches and two cross-batch repeated recipes.'}
        if self.batch_enabled:
            names = sorted(set(groups))
            cross = self.batch_variance * (groups[:, None] == np.array(names)[None, :])
            v = solve_triangular(self.posterior['factor'], self.posterior['root'][:, None] * cross, lower=True)
            means = cross.T @ self.posterior['alpha']
            variances = np.maximum(self.batch_variance - (v*v).sum(axis=0), 0)
            self.diagnostics['batch_effect']['offsets'] = [{'batch_id': str(name), 'latent_mean': float(mean),
                'conditional_sd': float(np.sqrt(variance))} for name, mean, variance in zip(names, means, variances)]
        return self

    def latent(self, x, joint=False, prediction_groups=None):
        if prediction_groups is not None and len(prediction_groups) != len(x):
            raise ValueError('Prediction batch identifiers must align with recipe rows.')
        cross = self.kernel(self.x, x)
        v = solve_triangular(self.posterior['factor'], self.posterior['root'][:, None] * cross, lower=True)
        mean = cross.T @ self.posterior['alpha']
        if joint:
            covariance = self.kernel(x) - v.T @ v
            # All proposals are interpreted as one new synthesis batch.
            batches = np.ones_like(covariance) if prediction_groups is None else np.equal.outer(prediction_groups, prediction_groups)
            covariance = covariance + self.batch_variance*batches
            covariance = (covariance + covariance.T) / 2
            return mean, covariance
        return mean, np.maximum(self.kernel.diag(x) + self.batch_variance - np.sum(v * v, axis=0), 1e-10)


def marginal_probability(mean, variance, slope=1., intercept=0.):
    f = mean[:, None] + np.sqrt(2 * np.maximum(variance, 0))[:, None] * NODES
    return np.clip(expit(slope * f + intercept) @ NODE_WEIGHTS / np.sqrt(np.pi), 1e-9, 1 - 1e-9)


class AdaptivePredictor:
    def __init__(self, iterations=18, seed=42, batch_effect=True, descriptor_profile='periodic'):
        self.iterations, self.seed, self.batch_effect = iterations, seed, batch_effect
        self.descriptor_profile = descriptor_profile

    def fit(self, x, y, groups):
        x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
        fit_indices, calibration_indices = np.arange(len(y)), np.array([], dtype=int)
        if len(y) >= 80 and len(set(groups)) >= 8:
            train, holdout = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=self.seed).split(x, y, groups))
            if min(np.bincount(y[train], minlength=2)) >= 3 and min(np.bincount(y[holdout], minlength=2)) >= 6 and len(holdout) >= 24:
                fit_indices, calibration_indices = train, holdout
        self.fit_indices, self.calibration_indices = fit_indices, calibration_indices
        # Dense GP inference has cubic cost. A declared representative expert
        # budget does not remove records from the full-data tree challenger or
        # reliability/count experts. Calibration is excluded before selection.
        gp_indices=fit_indices
        if len(fit_indices)>256:
            rng=np.random.default_rng(self.seed)
            strata={}
            for i in fit_indices:
                strata.setdefault((tuple(x[i,:len(ELEMENTS)]),int(y[i])),[]).append(int(i))
            selected=[int(rng.choice(values)) for values in strata.values()]
            if len(selected)>256:
                by_label=[np.array([i for i in selected if y[i]==label],dtype=int) for label in (0,1)]
                counts=[min(128,len(part)) for part in by_label]
                spare=256-sum(counts)
                for j in range(2):
                    extra=min(spare,len(by_label[j])-counts[j]); counts[j]+=extra; spare-=extra
                selected=[int(i) for part,count in zip(by_label,counts) for i in rng.choice(part,count,replace=False)]
            for batch in rng.permutation(np.unique(groups[fit_indices])):
                if len(selected)>=256:
                    break
                if not any(groups[i]==batch for i in selected):
                    selected.append(int(rng.choice(fit_indices[groups[fit_indices]==batch])))
            remaining=np.setdiff1d(fit_indices,selected)
            if len(selected)<256:
                selected.extend(map(int,rng.choice(remaining,256-len(selected),replace=False)))
            gp_indices=np.array(sorted(selected),dtype=int)
        self.gp_fit_indices=gp_indices
        self.gp = LearnedGP(self.iterations, self.seed, self.descriptor_profile).fit(x[gp_indices], y[gp_indices], groups[gp_indices], self.batch_effect)
        self.slope, self.intercept = 1., 0.
        self.calibration_status = 'UNCALIBRATED'
        self.calibration_diagnostics = {}
        if len(calibration_indices):
            mean, variance = self.gp.latent(x[calibration_indices])
            observed = y[calibration_indices]

            def objective(theta):
                p = marginal_probability(mean, variance, np.exp(theta[0]), theta[1])
                return log_loss(observed, p, labels=[0, 1]) + .015 * (theta @ theta)

            result = minimize(objective, np.zeros(2), method='L-BFGS-B', bounds=[(-2., 1.6), (-3., 3.)])
            if result.success and result.fun <= objective(np.zeros(2)):
                self.slope, self.intercept = float(np.exp(result.x[0])), float(result.x[1])
                self.calibration_status = 'BATCH_HOLDOUT_SIGMOID_FIT'
            self.calibration_diagnostics = {'n': len(observed), 'batch_ids': sorted(set(groups[calibration_indices])),
                                            'slope': self.slope, 'intercept': self.intercept,
                                            'note': 'Calibration rows are excluded from GP fitting. Fit quality is not independent calibration validation.'}
        n = len(ELEMENTS)
        self.calibration_members = x[calibration_indices, :n].copy()
        self.calibration_chemistry = x[calibration_indices, 2*n+4:].copy()
        self.challenger = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, min_samples_leaf=5,
                                                        l2_regularization=2., random_state=self.seed).fit(x[fit_indices], y[fit_indices])
        return self

    def calibration_mask(self, x):
        if not len(self.calibration_members) or self.calibration_status == 'UNCALIBRATED':
            return np.zeros(len(x), dtype=bool)
        matches = np.all(np.asarray(x)[:, None, :len(ELEMENTS)] == self.calibration_members[None, :, :], axis=2)
        if self.calibration_chemistry.shape[1]:
            matches &= np.all(np.asarray(x)[:, None, 2*len(ELEMENTS)+4:] == self.calibration_chemistry[None, :, :], axis=2)
        return np.any(matches, axis=1)

    def predict_proba(self, x):
        x = np.asarray(x)
        mean, variance = self.gp.latent(x)
        raw = marginal_probability(mean, variance)
        mask = self.calibration_mask(x)
        raw[mask] = marginal_probability(mean[mask], variance[mask], self.slope, self.intercept)
        return np.column_stack([1 - raw, raw])

    def probability_draws(self, x, draws=256, joint=False, seed=42, prediction_groups=None):
        x = np.asarray(x)
        mean, spread = self.gp.latent(x, joint, prediction_groups)
        rng = np.random.default_rng(seed)
        if joint:
            eigenvalues, vectors = np.linalg.eigh(spread)
            if eigenvalues.min() < -1e-6:
                raise ValueError('The computed posterior covariance is not positive semidefinite.')
            f = mean + rng.standard_normal((draws, len(x))) @ (vectors * np.sqrt(np.maximum(eigenvalues, 0))).T
        else:
            f = mean + rng.standard_normal((draws, len(x))) * np.sqrt(spread)
        mask = self.calibration_mask(x)
        f[:, mask] = self.slope * f[:, mask] + self.intercept
        return expit(f)

    def summary(self, x, seed=42):
        p = self.predict_proba(x)[:, 1]
        draws = self.probability_draws(x, seed=seed)
        return {'p': p, 'low': np.quantile(draws, .05, axis=0), 'high': np.quantile(draws, .95, axis=0),
                'bald': np.maximum(entropy(draws.mean(axis=0)) - entropy(draws).mean(axis=0), 0),
                'disagreement': np.abs(p - self.challenger.predict_proba(x)[:, 1]),
                'calibrated': self.calibration_mask(x)}


def eligible_records(samples, methods, context, origin, transfer, descriptor_profile='periodic'):
    from .precursors import base_profile
    descriptor_profile = base_profile(descriptor_profile)
    chosen, exclusions = [], Counter()
    for s in samples:
        reason = None
        if s.get('origin') != origin:
            reason = 'Different data namespace'
        elif ((process_key(s)[1:] != process_key(context)[1:] if descriptor_profile in {'chemistry_ion', 'chemistry_precursor', 'chemistry_state'} else process_key(s) != process_key(context)) if transfer else cohort_key(s) != cohort_key(context)):
            reason = 'Different composition or process cohort'
        elif transfer and descriptor_profile in {'chemistry_ion', 'chemistry_precursor', 'chemistry_state'} and s.get('precursor_family') != context.get('precursor_family'):
            from .precursors import precursor_coverage, documented
            chemistry = precursor_coverage(s, for_training=True) + precursor_coverage(context)
            unknown = any(not documented(row['source']) or row['oxidation_state'] is None for row in chemistry) if descriptor_profile == 'chemistry_ion' else any(row['type_used_by_model'] == 'UNKNOWN' for row in chemistry)
            if unknown:
                reason = 'Cross-system transfer needs source-backed per-element chemistry for the selected profile'
        if reason is None and any(str(s.get(k) or 'UNKNOWN').upper() in {'UNKNOWN', 'UNASSIGNED', ''} or str(s.get(k)).upper().startswith(('FILL_IN_', 'REPLACE_')) for k in ('precursor_family', 'support_batch', 'protocol_id')):
            reason = 'Unresolved process metadata'
        elif reason is None and (s.get('pulse_limit_C') is None or s.get('pulse_count') is None or any(s['concentrations_mM'].get(e) is None for e in s['elements'])):
            reason = 'Incomplete controlled synthesis conditions'
        elif reason is None:
            result = screen(s, methods)
            if result['review_conflicts'] or result['label'] is None:
                reason = 'Unresolved or conflicting screening label'
        if reason:
            exclusions[reason] += 1
        else:
            chosen.append(s)
    return chosen, dict(exclusions)


def scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    predicted_pass = p >= .5
    return {'n': len(y), 'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'balanced_accuracy': float(balanced_accuracy_score(y, predicted_pass)),
            'false_pass_count': int(np.sum(predicted_pass & (y == 0))), 'predicted_pass_n': int(predicted_pass.sum()),
            'false_pass_fraction': float(np.sum(predicted_pass & (y == 0)) / predicted_pass.sum()) if predicted_pass.any() else None}


def protected_validation(x, y, records, iterations, batch_effect=True, descriptor_profile='periodic'):
    """Outer batch holdouts contain neither kernel-fit nor calibration rows."""
    groups = np.array([s['batch_id'] for s in records])
    if len(set(groups)) < 3:
        return None, [], []
    p, challenger, baseline = [np.full(len(y), np.nan) for _ in range(3)]
    provenance = []
    from .composition import transfer_classifier
    for train, test in GroupKFold(n_splits=min(3, len(set(groups)))).split(x, y, groups):
        if len(train) < 12 or min(np.bincount(y[train], minlength=2)) < 3:
            continue
        model = AdaptivePredictor(iterations, batch_effect=batch_effect, descriptor_profile=descriptor_profile).fit(x[train], y[train], groups[train])
        p[test] = model.predict_proba(x[test])[:, 1]
        challenger[test] = model.challenger.predict_proba(x[test])[:, 1]
        baseline_x = x[:, :2*len(ELEMENTS)+4]
        baseline[test] = transfer_classifier().fit(baseline_x[train], y[train]).predict_proba(baseline_x[test])[:, 1]
        provenance.append({'train_batch_ids': sorted(set(groups[train])), 'test_batch_ids': sorted(set(groups[test])),
                           'calibration_batch_ids': sorted(set(groups[train][model.calibration_indices])),
                           'kernel_fit_batch_ids': sorted(set(groups[train][model.fit_indices]))})
    valid = np.isfinite(p)
    if not valid.any() or len(set(y[valid])) != 2:
        return None, [], provenance
    result = {**scores(y[valid], p[valid]), 'total_n': len(y), 'n_groups': len(set(groups[valid])),
              'full_coverage': bool(valid.all()), 'type': 'Outer synthesis-batch holdout; inner batch calibration; not chronological or novel-recipe validation'}
    comparison = [{'model': name, **scores(y[valid], values[valid])} for name, values in (
        ('Learned hierarchical GP', p), ('Fixed hierarchical GP', baseline), ('Uncalibrated tree challenger', challenger))]
    return result, comparison, provenance


def transfer_checks(x, y, records, iterations, batch_effect=True, descriptor_profile='periodic'):
    """Fit every transfer fold independently; purge actual shared batch IDs."""
    names = np.array([composition_name(s) for s in records])
    stages = np.array([len(s['elements']) for s in records])
    batches = np.array([s['batch_id'] for s in records])
    results = []
    for kind, keys in (('composition', sorted(set(names))[:3]), ('expansion', sorted(set(stages))[1:4])):
        predictions = np.full(len(y), np.nan)
        provenance = []
        for key in keys:
            test = np.flatnonzero(names == key if kind == 'composition' else stages == key)
            train = np.flatnonzero((names != key if kind == 'composition' else stages < key) & ~np.isin(batches, batches[test]))
            if len(train) < 12 or min(np.bincount(y[train], minlength=2)) < 3:
                continue
            model = AdaptivePredictor(iterations, batch_effect=batch_effect, descriptor_profile=descriptor_profile).fit(x[train], y[train], batches[train])
            predictions[test] = model.predict_proba(x[test])[:, 1]
            provenance.append({'held_out': str(key), 'train_compositions': sorted(set(names[train])),
                               'train_stages': sorted(map(int, set(stages[train]))), 'test_stages': sorted(map(int, set(stages[test]))),
                               'train_batch_ids': sorted(set(batches[train])), 'test_batch_ids': sorted(set(batches[test]))})
        valid = np.isfinite(predictions)
        if valid.any() and len(set(y[valid])) == 2:
            results.append({**scores(y[valid], predictions[valid]), 'total_n': len(y), 'full_coverage': bool(valid.all()),
                            'n_groups': len(set(batches[valid])), 'type': kind + ' holdout with shared-batch purge', 'folds': provenance})
        else:
            results.append(None)
    return results


def fit_adaptive(samples, methods, context, origin, transfer=True, settings=None):
    from .planning import ModelState, fingerprint
    from .precursors import base_profile
    from .physics_features import physics_manifest
    settings = settings or {}
    profile = settings.get('descriptor_profile', 'periodic')
    chosen, exclusions = eligible_records(samples, methods, context, origin, transfer, profile)
    y = np.array([screen(s, methods)['label'] for s in chosen], dtype=int)
    counts = Counter(y)
    target = composition_name(context)
    compositions = Counter(composition_name(s) for s in chosen)
    occurrences = Counter(e for s in chosen for e in s['elements'])
    stages = Counter(len(s['elements']) for s in chosen)
    target_pairs = set(combinations(sorted(context['elements']), 2))
    seen_pairs = {pair for s in chosen for pair in combinations(sorted(s['elements']), 2)}
    info = {'mode': 'ADAPTIVE_DESIGN_ONLY', 'engine': 'adaptive', 'feature_kind': 'COMPOSITION_TRANSFER',
            'model_version': VERSION, 'origin': origin, 'n_labeled': len(chosen), 'n_pass': counts[1], 'n_fail': counts[0],
            'n_batches': len({s['batch_id'] for s in chosen}), 'n_recipes': len({recipe_key(s) for s in chosen}),
            'minimum_batches': 0, 'target_labeled_n': compositions[target], 'unseen_target': compositions[target] == 0,
            'target_elements': context['elements'], 'target_element': '+'.join(context['elements']), 'source_elements': dict(occurrences),
            'source_compositions': dict(compositions), 'source_stages': dict(stages), 'highest_source_stage': max(stages, default=0),
            'unseen_elements': [e for e in context['elements'] if not occurrences[e]], 'target_pair_count': len(target_pairs),
            'unobserved_pair_count': len(target_pairs - seen_pairs), 'successful_source_ids': [s['sample_id'] for s, label in zip(chosen, y) if label],
            'exclusions': exclusions, 'validation': None, 'transfer_validation': None, 'expansion_validation': None,
            'validation_note': 'Independent batch evaluation unavailable.', 'cohort': list(cohort_key(context)),
            'features': ['membership', 'masked_log_concentrations', 'total_concentration', 'cardinality', 'pulses', 'controlled_cutoff'],
            'kernel_components': {'shared_members': .45, 'pair_residual': .20, 'exact_composition_residual': .35},
            'reported_label_count': sum(screen(s, methods)['icp_basis'] == 'USER_REPORTED' or screen(s, methods)['xrd_basis'] == 'USER_REPORTED' for s in chosen),
            'screening_rule_version': RULE_VERSION, 'stem_role': 'Optional confirmation only',
            'transfer_validation_note': 'New-composition performance is unvalidated; batch holdout is not transfer validation.',
            'expansion_validation_note': 'Higher-stage performance requires prospective or stage-held-out target outcomes.',
            'probability_note': 'Approximate Laplace probabilities; estimated hyperparameters; 90% conditional model intervals, not guaranteed coverage.'}
    info['physics'] = physics_manifest(profile)
    from .manual import actual_batch_recorded
    info['unknown_batch_samples'] = sum(not actual_batch_recorded(s) for s in chosen)
    info['n_batches'] = len({s['batch_id'] for s in chosen if actual_batch_recorded(s)})
    info['batch_provenance_note'] = ('Unknown-batch samples may fit the response model. Unknown batch IDs are not '
        'evidence of independent syntheses and remain grouped in validation. No independent-batch reliability claim follows.')
    if profile in {'chemistry_mixing', 'chemistry_pulseicp'}:
        info['mixing_training_n'] = sum(len(s['elements']) > 1 for s in chosen)
        info['mixing_note'] = 'Mixing features enter the learned covariance and challengers. Single-element data have zero mixing H/S by definition and cannot identify mixture effects; these are not labels of atomic dispersion.'
    diversity = [len({recipe_key(s) for s, label in zip(chosen, y) if label == c}) for c in (0, 1)]
    info['fitting_gate'] = {'required_samples': 12, 'eligible_samples': len(chosen),
        'required_distinct_recipes_per_class': 3, 'pass_recipes': diversity[1], 'fail_recipes': diversity[0],
        'met': len(chosen) >= 12 and min(diversity) >= 3}
    if profile != 'periodic':
        from .chemical_descriptors import reference
        from .precursors import state_reference
        info.update(descriptor_profile=profile, descriptor_snapshot_sha256=reference()['snapshot_sha256'],
                    state_descriptor_snapshot_sha256=state_reference()['snapshot_sha256'])
    if len(chosen) < 12 or min(diversity) < 3:
        info['reason'] = (f"Design only: {len(chosen)}/12 eligible samples; {diversity[1]}/3 distinct PASS recipes; "
            f"{diversity[0]}/3 distinct FAIL recipes. These are fitting gates, not a reliability guarantee. "
            'Experiment count is controlled separately by allowed ranges, deduplication and requested batch size.')
        return ModelState(None, info, chosen)
    x = np.array([transfer_feature(s, profile, for_training=True) for s in chosen])
    iterations = int(settings.get('learning_iterations', 18))
    if not 2 <= iterations <= 60:
        raise ValueError('Learning iterations must be between 2 and 60.')
    batch_effect = bool(settings.get('batch_effect', False))
    model = AdaptivePredictor(iterations, batch_effect=batch_effect, descriptor_profile=settings.get('descriptor_profile', 'periodic')).fit(x, y, [s['batch_id'] for s in chosen])
    model.methods = methods
    diagnostics_deferred=bool(settings.get('skip_validation') or len(chosen)>256)
    validation, comparison, provenance = protected_validation(x, y, chosen, iterations, batch_effect, profile) if not diagnostics_deferred else (None, [], [])
    transfer_validation, expansion_validation = transfer_checks(x, y, chosen, iterations, batch_effect, profile) if transfer and not diagnostics_deferred else (None, None)
    info.update(mode='ADAPTIVE_HIERARCHICAL_GP', reason='Learned hierarchical Bernoulli GP; regularized evidence learning, batch-held-out calibration and joint information acquisition.',
                validation=validation, model_comparison=comparison, validation_provenance=provenance,
                learning=model.gp.diagnostics, calibration_status=model.calibration_status,
                calibration=model.calibration_diagnostics, gp_fit_n=len(model.gp_fit_indices), calibration_n=len(model.calibration_indices),
                challenger_fit_n=len(model.fit_indices), gp_representative_budget=256,
                validation_deferred=diagnostics_deferred,
                gp_budget_note='Dense GP expert uses at most 256 training representatives selected after calibration exclusion; composition/outcome strata and actual batches are covered where capacity permits. The tree challenger uses every noncalibration training record. This is a bounded expert approximation, not a full-data GP posterior.',
                kernel_fit_records=[chosen[i].get('record_uid') or chosen[i]['sample_id'] for i in model.gp_fit_indices],
                calibration_records=[chosen[i].get('record_uid') or chosen[i]['sample_id'] for i in model.calibration_indices])
    info.update(transfer_validation=transfer_validation, expansion_validation=expansion_validation,
                transfer_validation_note='At most three held-out compositions; every fold refits learning/calibration on source rows only. Retrospective coverage is not prospective transfer certification.',
                expansion_validation_note='At most three higher stages; training includes lower stages only and purges shared synthesis batches. Report actual evaluated coverage.')
    params = model.gp.kernel.parameters()
    info['kernel_components'] = dict(zip(('shared_members', 'pair_residual', 'exact_composition_residual'), (params[k] for k in NAMES[:3])))
    info['model_id'] = fingerprint({'samples': chosen, 'labels': y.tolist(), 'methods': methods, 'settings': settings, 'version': VERSION, 'rule': RULE_VERSION,
                                   'descriptor_snapshot': info.get('descriptor_snapshot_sha256'), 'state_descriptor_snapshot': info.get('state_descriptor_snapshot_sha256'), 'physics': info['physics']})
    return ModelState(model, info, chosen)
