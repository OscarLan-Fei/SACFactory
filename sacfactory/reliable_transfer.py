"""Batch-held-out mixture weights for pooled, target, source and count experts.

Weight fitting is development evidence, not an independent performance result.
The ensemble remains uncalibrated even when a component has a fitted calibrator.
"""
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

from .adaptive import AdaptivePredictor, entropy
from .composition import composition_name, transfer_feature
from .domain import recipe_key, screen
from .recipe_reliability import CountReliabilityModel, recipe_evidence


VERSION = 'recipe-reliability-transfer-v2-chemistry'


def ready(records, methods):
    return len(records) >= 12 and all(len({recipe_key(s) for s in records if screen(s, methods)['label'] == label}) >= 3 for label in (0, 1))


def fit_expert(records, methods, kind, iterations, batch_effect, descriptor_profile='periodic'):
    if kind == 'count':
        return CountReliabilityModel().fit(records, methods, descriptor_profile)
    model = AdaptivePredictor(iterations, batch_effect=batch_effect, descriptor_profile=descriptor_profile)
    model.fit(np.array([transfer_feature(s, descriptor_profile, for_training=True) for s in records]),
              np.array([screen(s, methods)['label'] for s in records]), [s['batch_id'] for s in records])
    return model


def optimize_weights(probabilities, y, groups, prior):
    """Simplex stacking with equal actual-batch loss and shrinkage to pooled GP."""
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    row_weights = 1/counts[inverse]
    row_weights /= row_weights.sum()
    def loss(weights):
        p = np.clip(probabilities @ weights, 1e-8, 1-1e-8)
        return float(np.sum(row_weights*(-y*np.log(p)-(1-y)*np.log1p(-p))) + .02*np.sum((weights-prior)**2))
    result = minimize(loss, prior, method='SLSQP', bounds=[(0., 1.)]*len(prior),
                      constraints={'type': 'eq', 'fun': lambda w: w.sum()-1}, options={'maxiter': 100, 'ftol': 1e-8})
    weights = result.x if result.success and loss(result.x) <= loss(prior) else prior
    weights = np.maximum(weights, 0); weights /= weights.sum()
    return weights, {'objective_before': loss(prior), 'objective_after': loss(weights), 'optimizer_converged': bool(result.success),
        'development_rows': len(y), 'development_batches': len(set(groups)),
        'note': 'Equal-batch log-loss stacking on OOF target rows; development fit, not independent validation or causal source contribution.'}


class ReliabilityEnsemble:
    def fit(self, records, methods, context, pooled, iterations=8, batch_effect=True, descriptor_profile='periodic'):
        self.descriptor_profile = descriptor_profile
        target_name = composition_name(context)
        target = [s for s in records if composition_name(s) == target_name]
        source_names = Counter(composition_name(s) for s in records if composition_name(s) != target_name)
        specs = [('Pooled GP', 'gp', None)]
        if ready(target, methods):
            specs.append(('Target GP', 'gp', target_name))
        for name, _ in source_names.most_common(2):
            if ready([s for s in records if composition_name(s) == name], methods):
                specs.append(('Source GP: '+name, 'gp', name))
        specs.append(('Count reliability', 'count', None))
        self.experts, self.names = [pooled], [specs[0][0]]
        for name, kind, composition in specs[1:]:
            chosen = [s for s in records if composition is None or composition_name(s) == composition]
            self.experts.append(fit_expert(chosen, methods, kind, iterations, batch_effect, descriptor_profile))
            self.names.append(name)
        self.gp = pooled.gp
        self.methods = methods
        self.calibration_status = 'UNCALIBRATED_ENSEMBLE'
        self.weights = np.array([1.]+[0.]*(len(specs)-1))
        self.weight_diagnostics = {'status': 'POOLED_PRIOR_ONLY', 'folds': [], 'weights': dict(zip(self.names, self.weights.tolist()))}
        groups = np.array([s['batch_id'] for s in target])
        predictions, labels, batches, provenance = [], [], [], []
        # Both target and source experts must have valid fits on identical dev rows.
        if len(target) >= 24 and len(set(groups)) >= 3:
            for _, test_indices in GroupKFold(n_splits=min(3, len(set(groups)))).split(target, groups=groups):
                test = [target[i] for i in test_indices]
                excluded = {s['batch_id'] for s in test}
                train = [s for s in records if s['batch_id'] not in excluded]
                subsets = [[s for s in train if composition is None or composition_name(s) == composition] for _, _, composition in specs]
                if not all(ready(subset, methods) for subset in subsets):
                    continue
                experts = [fit_expert(subset, methods, kind, min(iterations, 4), batch_effect, descriptor_profile)
                           for subset, (_, kind, _) in zip(subsets, specs)]
                x = np.array([transfer_feature(s, descriptor_profile, for_training=True) for s in test])
                predictions.extend(np.column_stack([expert.predict_proba(x)[:, 1] for expert in experts]))
                labels.extend(screen(s, methods)['label'] for s in test)
                batches.extend(s['batch_id'] for s in test)
                provenance.append({'test_batch_ids': sorted(excluded),
                    'test_sample_ids': [s['sample_id'] for s in test],
                    'experts': [{'name': name, 'train_batch_ids': sorted({s['batch_id'] for s in subset}),
                        'train_sample_ids': [s['sample_id'] for s in subset]} for subset, (name, _, _) in zip(subsets, specs)]})
        if len(labels) >= 12 and len(set(labels)) == 2 and len(set(batches)) >= 3:
            self.weights, diagnostics = optimize_weights(np.asarray(predictions), np.asarray(labels), batches, self.weights)
            self.weight_diagnostics.update(diagnostics, status='OOF_TARGET_WEIGHT_FIT', folds=provenance)
        else:
            self.weight_diagnostics.update(reason='Need sufficient identical, protected two-class target development predictions. Unseen targets retain pooled transfer prior.', folds=provenance)
        self.weight_diagnostics['weights'] = dict(zip(self.names, self.weights.tolist()))
        self.reliability = recipe_evidence(records, methods)
        return self

    def calibration_mask(self, x):
        return np.zeros(len(x), dtype=bool)

    def predict_proba(self, x):
        p = sum(weight*expert.predict_proba(x)[:, 1] for weight, expert in zip(self.weights, self.experts) if weight > 1e-10)
        return np.column_stack([1-p, p])

    def probability_draws(self, x, draws=256, joint=False, seed=42, prediction_groups=None):
        # Select one expert per full draw, preserving joint rather than pointwise mixtures.
        rng = np.random.default_rng(seed)
        choices = rng.choice(len(self.experts), draws, p=self.weights)
        result = np.empty((draws, len(x)))
        for i, expert in enumerate(self.experts):
            selected = np.flatnonzero(choices == i)
            if len(selected):
                result[selected] = expert.probability_draws(x, len(selected), joint, seed+i+1, prediction_groups=prediction_groups)
        return result

    def summary(self, x, seed=42):
        p = self.predict_proba(x)[:, 1]
        draws = self.probability_draws(x, seed=seed)
        components = np.array([expert.predict_proba(x)[:, 1] for expert in self.experts])
        spread = np.sqrt(np.sum(self.weights[:, None]*(components-p)**2, axis=0))
        return {'p': p, 'low': np.quantile(draws, .05, axis=0), 'high': np.quantile(draws, .95, axis=0),
            'bald': np.maximum(entropy(draws.mean(axis=0))-entropy(draws).mean(axis=0), 0),
            'disagreement': spread, 'calibrated': np.zeros(len(x), dtype=bool)}


def fit_reliability(samples, methods, context, origin, transfer=True, settings=None):
    from .adaptive import fit_adaptive
    from .planning import fingerprint
    settings = settings or {}
    profile = settings.get('descriptor_profile', 'periodic')
    from .precursors import base_profile
    base = base_profile(profile)
    state = fit_adaptive(samples, methods, context, origin, transfer,
                         {**settings, 'skip_validation': True, 'descriptor_profile': profile})
    state.info.update(engine='reliability', model_version=VERSION, descriptor_profile=profile,
        probability_note='Provisional mixture probability; conditional approximate expert intervals. Ensemble is not calibrated or prospectively certified.')
    if profile != 'periodic':
        from .chemical_descriptors import reference
        from .precursors import state_reference, precursor_coverage
        state.info['descriptor_snapshot_sha256'] = reference()['snapshot_sha256']
        state.info['state_descriptor_snapshot_sha256'] = state_reference()['snapshot_sha256']
        state.info['target_precursor_coverage'] = precursor_coverage(context)
        state.info['chemistry_scope'] = 'Declared precursor inputs and state-matched reference descriptors; no final SAC oxidation state is inferred.'
        state.info['features'] = state.info['features'] + ['sourced_elemental_properties_and_missing_masks']
        if base in {'chemistry_ion', 'chemistry_precursor', 'chemistry_state'}:
            state.info['features'] += ['formal_precursor_oxidation_moments']
        if base in {'chemistry_precursor', 'chemistry_state'}:
            state.info['features'] += ['sourced_precursor_family_fractions', 'type_conditioned_elemental_moments']
        if base == 'chemistry_state':
            state.info['features'] += ['explicit_state_matched_ionic_radius_moments']
    if state.model is None:
        return state
    ensemble = ReliabilityEnsemble().fit(state.training, methods, context, state.model,
        int(settings.get('learning_iterations', 8)), bool(settings.get('batch_effect', False)), profile)
    state.model = ensemble
    state.info.update(mode='RECIPE_RELIABILITY_ENSEMBLE', reason='Recipe reliability, target/source OOF weight adaptation and repeat-budget planning.',
        transfer_weights=ensemble.weight_diagnostics, calibration_status=ensemble.calibration_status,
        reliability_count_model=ensemble.experts[-1].diagnostics,
        validation=None, model_comparison=[], validation_provenance=[],
        validation_note='Weight-development fit is not independent validation. Use protected reliability benchmark in Data analysis.',
        model_id=fingerprint({'samples': state.training, 'methods': methods, 'context': context, 'settings': settings, 'version': VERSION,
                              'descriptor_snapshot': state.info.get('descriptor_snapshot_sha256'),
                              'state_descriptor_snapshot': state.info.get('state_descriptor_snapshot_sha256'), 'physics': state.info.get('physics')}))
    state.info['transfer_validation_note'] = 'Target/source weights use protected development rows; strict unseen-element reliability benchmarks are available on Data analysis.'
    state.info['expansion_validation_note'] = 'Higher-stage reliability remains a transfer hypothesis until independently evaluated target outcomes exist.'
    return state
