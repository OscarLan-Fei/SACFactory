"""Recipe-level evidence and an optional beta-binomial count challenger.

Batch-balanced beta summaries are working evidence summaries, not GP forecasts.
They never replace the observed k/n on validation plots.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln
from scipy.stats import beta
from sklearn.preprocessing import StandardScaler

from .composition import transfer_feature
from .domain import ELEMENTS, recipe_key, screen


def recipe_groups(records, methods):
    groups = defaultdict(list)
    for sample in records:
        from .manual import actual_batch_recorded
        if not actual_batch_recorded(sample) or sample.get('pulse_limit_C') is None or sample.get('pulse_count') is None or any(sample['concentrations_mM'].get(e) is None for e in sample['elements']):
            continue
        if any(str(sample.get(k) or 'UNKNOWN').strip().upper() in {'UNKNOWN', 'UNASSIGNED'} or str(sample.get(k)).upper().startswith(('FILL_IN_', 'REPLACE_')) for k in ('precursor_family', 'support_batch', 'protocol_id')):
            continue
        outcome = screen(sample, methods)
        if outcome['label'] is not None and not outcome['review_conflicts']:
            groups[recipe_key(sample)].append(sample)
    return groups


def recipe_evidence(records, methods):
    """One unit of evidence per actual batch; within-batch rates remain explicit."""
    from .planning import fingerprint
    rows = []
    for key, samples in recipe_groups(records, methods).items():
        by_batch = defaultdict(list)
        for sample in samples:
            by_batch[sample['batch_id']].append(screen(sample, methods)['label'])
        rates = [float(np.mean(labels)) for labels in by_batch.values()]
        successes = sum(sum(labels) for labels in by_batch.values())
        a, b = .5 + sum(rates), .5 + len(rates) - sum(rates)
        low, high = beta.ppf([.025, .975], a, b)
        rows.append({'recipe_id': 'R_' + fingerprint(key), 'example_sample': samples[0]['sample_id'],
            'recipe': samples[0], 'n': len(samples), 'successes': int(successes), 'observed_rate': successes/len(samples),
            'independent_batches': len(rates), 'equal_batch_rate': float(np.mean(rates)), 'a': float(a), 'b': float(b),
            'working_probability': float(a/(a+b)), 'working_lower_95': float(low), 'working_upper_95': float(high),
            'between_batch_rate_variance': float(np.var(rates)) if len(rates) > 1 else 0.,
            'source_sample_ids': [s['sample_id'] for s in samples],
            'note': 'Jeffreys beta working summary with one fractional outcome per batch; not an exact binomial posterior for multi-sample batches or a calibrated GP interval.'})
    return rows


def count_table(records, methods):
    from .planning import fingerprint
    groups = defaultdict(list)
    for sample in records:
        groups[(recipe_key(sample), sample['batch_id'])].append(sample)
    return [{'recipe_id': 'R_'+fingerprint(key), 'batch_id': batch, 'n': len(samples),
             'k': int(sum(screen(s, methods)['label'] for s in samples)), 'sample': samples[0],
             'sample_ids': [s['sample_id'] for s in samples]} for (key, batch), samples in groups.items()]


def compact_features(x):
    """Small condition/count representation for a regularized count challenger."""
    x = np.atleast_2d(x)
    n = len(ELEMENTS)
    mask, logs = x[:, :n], x[:, n:2*n]
    count = mask.sum(axis=1)
    average = logs.sum(axis=1)/np.sqrt(count)
    second = (logs*logs).sum(axis=1)
    spread = np.sqrt(np.maximum(second-average*average, 0))
    process = x[:, 2*n:2*n+4]
    # Retain elemental identity with a small deterministic periodic summary.
    from .composition import descriptors
    positions = np.array([descriptors(e)[:3] for e in ELEMENTS])
    member_mean = mask @ positions / count[:, None]
    return np.column_stack([process, average, spread, average**2, process[:, -1]**2,
                            average*process[:, -1], member_mean, x[:, 2*n+4:]])


class CountReliabilityModel:
    """Penalized logistic mean with supported within-recipe/batch overdispersion."""
    def fit(self, records, methods, descriptor_profile='periodic'):
        self.descriptor_profile = descriptor_profile
        rows = count_table(records, methods)
        self.scaler = StandardScaler().fit(compact_features(np.array([transfer_feature(r['sample'], descriptor_profile, for_training=True) for r in rows])))
        raw = self.scaler.transform(compact_features(np.array([transfer_feature(r['sample'], descriptor_profile, for_training=True) for r in rows])))
        design = np.column_stack([np.ones(len(raw)), raw])
        k, trials = np.array([r['k'] for r in rows]), np.array([r['n'] for r in rows])
        repeated = [r for r in rows if r['n'] >= 2]
        enabled = len(repeated) >= 8 and len({r['batch_id'] for r in repeated}) >= 3 and len({r['recipe_id'] for r in repeated}) >= 3
        penalty = np.array([.1]+[1.]*raw.shape[1])
        start = np.zeros(design.shape[1]+int(enabled))
        if enabled:
            start[-1] = np.log(20.)

        def objective(theta):
            p = np.clip(expit(design @ theta[:design.shape[1]]), 1e-8, 1-1e-8)
            if enabled:
                precision = np.exp(theta[-1])
                likelihood = betaln(k+precision*p, trials-k+precision*(1-p)) - betaln(precision*p, precision*(1-p))
                reg = .05*(theta[-1]-np.log(20.))**2
            else:
                likelihood = k*np.log(p)+(trials-k)*np.log1p(-p)
                reg = 0.
            return -likelihood.sum() + .5*np.sum(penalty*theta[:design.shape[1]]**2) + reg

        bounds = [(-8., 8.)]*design.shape[1] + ([(np.log(2.), np.log(1000.))] if enabled else [])
        result = minimize(objective, start, method='L-BFGS-B', bounds=bounds, options={'maxiter': 150})
        theta = result.x if np.isfinite(result.fun) and result.fun <= objective(start) else start
        self.coefficients = theta[:design.shape[1]]
        self.precision = float(np.exp(theta[-1])) if enabled else None
        p = expit(design @ self.coefficients)
        effective = trials/(1+(trials-1)/(self.precision+1)) if enabled else trials
        information = (design.T * (effective*p*(1-p))) @ design + np.diag(penalty)
        self.covariance = np.linalg.inv(information)
        self.diagnostics = {'family': 'BETA_BINOMIAL' if enabled else 'BINOMIAL', 'precision': self.precision,
            'count_groups': len(rows), 'within_batch_repeated_groups': len(repeated), 'n_samples': int(trials.sum()),
            'optimizer_converged': bool(result.success), 'regularized_loss': float(objective(theta)),
            'note': 'Regularized compact logistic count challenger; no second batch random intercept. Coefficient intervals use approximate working information, not full Bayesian inference.'}
        return self

    def predict_proba(self, x):
        design = np.column_stack([np.ones(len(x)), self.scaler.transform(compact_features(x))])
        from .adaptive import marginal_probability
        variance = np.sum((design @ self.covariance)*design, axis=1)
        p = marginal_probability(design @ self.coefficients, variance)
        return np.column_stack([1-p, p])

    def probability_draws(self, x, draws=256, joint=False, seed=42, prediction_groups=None):
        design = np.column_stack([np.ones(len(x)), self.scaler.transform(compact_features(x))])
        coefficients = np.random.default_rng(seed).multivariate_normal(self.coefficients, self.covariance, size=draws)
        p = expit(coefficients @ design.T)
        if self.precision is not None:
            # Shared count-group probability for identical recipes in the same future batch.
            groups = np.zeros(len(x), dtype=int) if prediction_groups is None else np.asarray(prediction_groups)
            if len(groups) != len(x):
                raise ValueError('Prediction batch groups must align with recipe rows.')
            cache = {}
            rng = np.random.default_rng(seed+1)
            for i in range(len(x)):
                key = (tuple(np.asarray(x)[i]), str(groups[i])) if joint else (i, str(groups[i]))
                if key not in cache:
                    mean = np.clip(p[:, i], 1e-8, 1-1e-8)
                    cache[key] = rng.beta(self.precision*mean, self.precision*(1-mean))
                p[:, i] = cache[key]
        return p
