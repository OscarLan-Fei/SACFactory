"""SACFactory adapters for the frozen, user-owned PulseICP engine.

All quantities entering this bridge are canonical metal mM and measured wt%.
No model, observation, credential or checkpoint is imported from PulseICP.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ._pulseicp.configuration import physics_settings, model_settings
from ._pulseicp.composition import Composition, component_features, decode_composition
from ._pulseicp.references import digest

ORIGINS = {'REAL': 'EXPERIMENTAL', 'DEMO': 'SIMULATED'}
FAMILIES = ['Gaussian residual', 'Conditional diffusion', 'Gradient boosting residual']
ROTATING_CV = 'Rotating batch cross-validation'
PROTOCOLS = ['Batch holdout', 'Chronological replay', 'Unseen element transfer', 'Singles to mixtures', ROTATING_CV]


def carbonization_fraction(sample, physics):
    for key in ('carbonization_loss_fraction', 'carbonization_loss'):
        if sample.get(key) is not None:
            return sample[key]
    return physics['anchor_carbonization_loss']


def configuration():
    """Return fresh snapshots; fitted models retain their own copies."""
    from ._pulseicp.screening_prior import prior_settings
    from .compute_runtime import resolve
    return {'physics': physics_settings(), 'model': {**model_settings(), **resolve(), 'thermodynamic_features': True},
            'screening_prior': prior_settings()}


def identity():
    from ._pulseicp.references import provenance
    return {'engine': 'PulseICP frozen source snapshot 2026-10-19',
            'configuration_sha256': digest(configuration()), 'references': provenance(),
            'quantity_basis': 'Metal mM; mmol = mM * solution mL / 1000; ICP W in wt%',
            'carbonization_status': 'User-reported common 6% loss; overridden by a recorded fraction',
            'kinetics_status': 'Unvalidated elicited prior; waveform assumptions are not measured sample metadata'}


def composition(sample, physics=None):
    physics = physics_settings() if physics is None else physics
    # PAN and volume must be recorded: historical unknowns are never filled here.
    if sample.get('pan_mass_g') is None or sample.get('solvent_volume_mL') is None:
        raise ValueError('Recorded PAN mass and DMF volume are required for physical ICP prediction.')
    fraction = carbonization_fraction(sample, physics)
    return Composition({e: sample['concentrations_mM'][e] for e in sample['elements']},
        sample.get('pulse_limit_C'), sample.get('pulse_count'), sample['pan_mass_g'],
        sample['solvent_volume_mL'], fraction).validate()


@lru_cache(maxsize=2048)
def _features(serialized):
    sample = json.loads(serialized)
    try:
        c = composition(sample)
        matrix = np.asarray([component_features(c, e, physics_settings(), True) for e in c.elements])
        weights = np.asarray([c.concentrations_mM[e] for e in c.elements]); weights /= weights.sum()
        return [*(weights @ matrix).tolist(), 1.]
    except (ValueError, TypeError, KeyError, OverflowError):
        # Placeholder vector plus explicit coverage; unsupported elements remain selectable.
        return [0.] * 51


def recipe_features(sample):
    keys = ('elements', 'concentrations_mM', 'pulse_limit_C', 'pulse_count', 'pan_mass_g',
            'solvent_volume_mL', 'carbonization_loss_fraction', 'carbonization_loss')
    return _features(json.dumps({k: sample.get(k) for k in keys}, sort_keys=True))


def input_frame(samples, origin, physics=None):
    """Complete member rows, with censoring and invalid specimens left in the audit."""
    from .domain import INVALID_RUN_CONDITIONS
    from .units import normalize_assay
    from .manual import actual_batch_recorded
    physics = physics_settings() if physics is None else physics
    records = []
    for sample in samples:
        if sample.get('origin') != origin:
            continue
        valid = sample.get('sample_condition') not in INVALID_RUN_CONDITIONS | {'BURNED_OUT'}
        for element in sample['elements']:
            assays = [a for a in sample.get('icp', []) if a.get('element') == element]
            assay = normalize_assay(assays[0]) if len(assays) == 1 else {}
            fraction = carbonization_fraction(sample, physics)
            records.append({'sample_id': sample['sample_id'], 'batch_id': sample.get('batch_id') if actual_batch_recorded(sample) else None,
                'origin': ORIGINS[origin], 'protocol_id': sample.get('protocol_id'),
                'element': element, 'concentration_mM': sample['concentrations_mM'].get(element),
                'temperature_C': sample.get('pulse_limit_C'), 'pulse_count': sample.get('pulse_count'),
                'pan_mass_g': sample.get('pan_mass_g'), 'volume_mL': sample.get('solvent_volume_mL'),
                'carbonization_loss_fraction': physics['anchor_carbonization_loss'] if fraction is None else fraction,
                'icp_w_wt_percent': assay.get('value'),
                'icp_flag': assay.get('flag', 'MEASURED') if valid else 'INVALID_SPECIMEN',
                'result_reported_at': sample.get('result_reported_at')})
    return pd.DataFrame(records)


def prepare(samples, origin, settings=None, physics=None):
    from ._pulseicp.data import prepare as engine_prepare
    physics = physics_settings() if physics is None else copy.deepcopy(physics)
    settings = configuration()['model'] if settings is None else copy.deepcopy(settings)
    return engine_prepare(input_frame(samples, origin, physics), physics, ORIGINS[origin],
                          settings.get('thermodynamic_features', True))


def fit(samples, origin, family=FAMILIES[0], settings=None, physics=None):
    """Fit a deployment/fold model on the supplied training specimens only."""
    from .accelerated_ensemble import ResidualEnsemble
    from .composition import process_key
    contexts = {process_key(s) for s in samples if s.get('origin') == origin}
    if len(contexts) != 1:
        raise ValueError('Fit inside one comparable precursor/support/protocol context.')
    settings = configuration()['model'] if settings is None else copy.deepcopy(settings)
    from .compute_runtime import resolve, emit
    settings.update(resolve(settings))
    emit({'stage':'Training loading model','family':family,'device':settings['device']})
    physics = physics_settings() if physics is None else copy.deepcopy(physics)
    from .compute_runtime import fit_cache
    cache=fit_cache()
    cache_key=digest({'samples':samples,'origin':origin,'family':family,'settings':settings,'physics':physics})
    if cache is not None and cache_key in cache:
        emit({'stage':'Reusing identical training-only fit','family':family})
        return cache[cache_key]
    data = prepare(samples, origin, settings, physics)
    if not data.records:
        raise ValueError('No mass-bound-compatible quantified training specimens; inspect eligibility.')
    from .token_residual import FAMILIES as TOKEN_FAMILIES,TokenResidualEnsemble
    cls=TokenResidualEnsemble if family in TOKEN_FAMILIES else ResidualEnsemble
    if family=='Gradient boosting residual':
        from .gb_residual import GBResidual
        cls=GBResidual
    if family in TOKEN_FAMILIES and settings.get('training_variant')=='validated-token-v1':
        from .tuned_token_residual import TunedTokenResidualEnsemble
        cls=TunedTokenResidualEnsemble
    model = cls(family, settings, physics).fit(data, np.arange(len(data.records)))
    model.sacfactory_process_key = next(iter(contexts))
    if cache is not None:
        # Bound live models, especially GPU allocations. No cross-request reuse.
        if len(cache)>=2:cache.pop(next(iter(cache)))
        cache[cache_key]=(model,data.audit)
    return model, data.audit


def _prediction_compositions(model,samples,origin):
    """Shared context checks for numerical predictions and decoded score draws."""
    if model.origin != ORIGINS[origin]:
        raise ValueError('Model origin differs from the selected data space.')
    if hasattr(model, 'sacfactory_process_key'):
        from .composition import process_key
        if any(digest(process_key(s)) != digest(model.sacfactory_process_key) for s in samples):
            raise ValueError('Precursor/support/protocol differs from the ICP training context.')
    compositions = [composition(s, model.physics) for s in samples]
    protocols = {s.get('protocol_id') for s in samples}
    if len(protocols) != 1 or protocols != set(model.protocols):
        raise ValueError('Predict inside the fitted model protocol; select a matching model.')
    return compositions,protocols


def predict(model, samples, origin):
    """Predict unmeasured recipe controls; observed ICP never enters this path."""
    compositions,protocols=_prediction_compositions(model,samples,origin)
    frame, _ = model.predict_compositions(compositions, origin=ORIGINS[origin], protocol=next(iter(protocols)))
    frame['sample_id'] = [samples[int(i)].get('sample_id', f'RECIPE_{int(i)+1}') for i in frame.composition_index]
    return frame


def predict_attainment(model,samples,origin,target,score_definition='capped_attainment_v1'):
    """Average capped whole-recipe utility over aligned mass-conserving draws."""
    if not np.isfinite(target) or target<=0:
        raise ValueError('A positive finite whole-recipe ICP target is required.')
    compositions,protocols=_prediction_compositions(model,samples,origin)
    frame,draws=model.predict_compositions(compositions,origin=ORIGINS[origin],protocol=next(iter(protocols)))
    draws=np.asarray(draws,dtype=float)
    if draws.ndim!=2 or len(draws)!=len(frame) or not np.isfinite(draws).all():
        raise ValueError('Aligned finite PulseICP composition draws are required.')
    output=[]
    for index,sample in enumerate(samples):
        mask=frame.composition_index.to_numpy()==index
        if int(mask.sum())!=len(compositions[index].elements):
            raise ValueError('The score needs every member of the predicted composition.')
        total=draws[mask].sum(axis=0)
        if np.any(total<0) or np.any(total>=100):
            raise ValueError('Decoded total metal fractions violate the matrix mass bound.')
        from .result_utility import loading_utility
        output.append({'sample_id':sample['sample_id'],'total_mean_wt_percent':float(total.mean()),
            'expected_attainment':float(loading_utility(total,target,score_definition).mean()),
            'loading_utility_definition':score_definition,
            'outside_training_support':bool(frame.loc[mask,'outside_training_support'].any()),
            'draws':len(total)})
    return output


def validate(samples, origin, settings, physics, output, protocol=PROTOCOLS[0], held_element=None, all_samples=None,
             family=FAMILIES[0], batches_per_fold=1, progress=None):
    """Independent train/calibration/test artifacts with global target-batch purging."""
    from .accelerated_validation import run_experiment
    from .composition import process_key
    from .compute_runtime import resolve
    settings={**settings,**resolve(settings)}
    selected = [s for s in samples if s.get('origin') == origin]
    contexts = {process_key(s) for s in selected}
    if len(contexts) != 1:
        raise ValueError('Select one comparable precursor/support/protocol context before validation.')
    if protocol == ROTATING_CV:
        from .pulseicp_cross_validation import run_cross_validation
        return run_cross_validation(input_frame(selected, origin, physics), ORIGINS[origin], physics,
            settings, output, family, batches_per_fold, list(next(iter(contexts))), progress)
    if protocol in {'Unseen element transfer', 'Singles to mixtures'}:
        target = lambda s: held_element in s['elements'] if protocol == 'Unseen element transfer' else len(s['elements']) > 1
        blocked = {s.get('batch_id') for s in (all_samples if all_samples is not None else selected)
                   if s.get('origin') == origin and target(s)}
        selected = [s for s in selected if target(s) or s.get('batch_id') not in blocked]
    result = run_experiment(input_frame(selected, origin, physics), ORIGINS[origin], physics,
        settings, output, split=protocol, held_element=held_element)
    result[3]['sacfactory_process_key'] = list(next(iter(contexts)))
    (result[0]/'run.json').write_text(json.dumps(result[3], indent=2, allow_nan=False), encoding='utf-8')
    return result


def forecast(samples, origin, root, family='gaussian', current_samples=None):
    """Use a validated snapshot if available; otherwise expose the unfitted prior."""
    from .accelerated_ensemble import ResidualEnsemble
    if family not in {'gaussian', 'diffusion'}:
        raise ValueError('Select the Gaussian or diffusion checkpoint family.')
    base = Path(root) / 'pulseicp' / origin.lower()
    pointer = base / 'latest.json'
    if pointer.exists():
        run_id = json.loads(pointer.read_text())['run_id']
        # Only an immediate run directory from this namespace may be selected.
        if Path(run_id).name != run_id or run_id in {'.', '..'}:
            raise ValueError('Invalid ICP checkpoint reference.')
        model = ResidualEnsemble.load(base / run_id / family)
        metadata = json.loads((base / run_id / 'run.json').read_text(encoding='utf-8'))
        model.sacfactory_process_key = tuple(metadata['sacfactory_process_key'])
        if current_samples is not None:
            ids = set(model.training_ids)
            current = prepare([s for s in current_samples if s['sample_id'] in ids], origin, model.settings, model.physics)
            grouped = {}
            for record in current.records:
                grouped.setdefault(record['raw']['sample_id'], []).append(record['raw'])
            raw = [r for ident in dict.fromkeys(model.training_ids) for r in grouped.get(ident, [])]
            if digest(raw) != model.training_fingerprint:
                raise ValueError('ICP training records changed or were deleted. Revalidate before using this saved checkpoint.')
        return predict(model, samples, origin)
    rows = []
    for i, sample in enumerate(samples):
        c = composition(sample)
        values = decode_composition(c, dict.fromkeys(c.elements, 0.), physics_settings())
        for e in c.elements:
            rows.append({'sample_id': sample.get('sample_id', f'RECIPE_{i+1}'), 'element': e,
                'predicted_mean_w_wt_percent': float(values[e]), 'model': 'Unfitted physical prior',
                'calibration_status': 'UNCERTAINTY_NOT_ESTIMATED',
                'model_id': digest(physics_settings())[:20]})
    return pd.DataFrame(rows)
