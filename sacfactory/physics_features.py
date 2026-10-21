"""Sourced thermal inputs, optional unfitted priors and paired ICP comparisons.

No measured peak, assay, label or synthetic latent ledger enters these features.
Elemental bulk properties are transfer covariates, not SAC binding energies.
"""
import hashlib
import json
import math
from functools import lru_cache

import numpy as np

from .domain import ELEMENTS
from .thermal_loss import elemental_loss_reference, loss_reference, loss_settings, metal_retention, positive

THERMAL_PROFILES = {'chemistry_thermal', 'chemistry_loss', 'chemistry_mixing', 'chemistry_pulseicp'}
THERMAL_FIELDS = ('melting_point_K', 'boiling_point_K', 'heat_of_formation_kJ_mol')
THERMAL_SCALES = (5000., 5000., 1000.)
MATERIAL_FIELDS = ('pan_mass_g', 'solvent_volume_mL', 'carbonization_loss')
VERSION = 'physics-assisted-features-v2-preparation-mixing'


@lru_cache(maxsize=1)
def thermal_matrix():
    """Fixed sourced values with missing masks; no invented sublimation values."""
    rows = []
    for element in ELEMENTS:
        source = elemental_loss_reference(element)
        values = [source.get(k) for k in THERMAL_FIELDS]
        rows.append([v / scale if positive(v) else 0. for v, scale in zip(values, THERMAL_SCALES)] +
                    [float(not positive(v)) for v in values])
    return np.asarray(rows)


def anchor_retention():
    """Unfitted user anchor: 1 mmol Fe gives 2.5 wt% at 1000 C / one pulse."""
    from .chemical_descriptors import reference
    weight = next(r['atomic_weight'] for r in reference()['elements'] if r['symbol'] == 'Fe')
    from ._pulseicp.configuration import physics_settings
    shared = physics_settings()
    fraction = shared['anchor_fe_w_wt_percent'] / 100.
    carrier = shared['anchor_pan_g'] * (1 - shared['anchor_carbonization_loss']) * (1 - shared['carrier_one_pulse_losses'][0])
    feed = shared['anchor_fe_mM'] * shared['anchor_volume_mL'] * weight / 1e6
    return fraction / (1 - fraction) * carrier / feed


def thermal_recipe_features(sample):
    """Molar-weighted thermal moments and cutoff-to-transition ratios."""
    positions = thermal_matrix()[[ELEMENTS.index(e) for e in sample['elements']]]
    weights = np.array([sample['concentrations_mM'][e] for e in sample['elements']], dtype=float)
    weights /= weights.sum()
    mean = weights @ positions
    sd = np.sqrt(np.maximum(weights @ (positions * positions) - mean * mean, 0.))
    temperature = float(sample['pulse_limit_C']) + 273.15
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('A finite controlled cutoff above absolute zero is required.')
    ratios = np.divide(temperature / 5000., positions[:, :2], out=np.zeros_like(positions[:, :2]), where=positions[:, :2] > 0.)
    return [*mean.tolist(), *sd.tolist(), *(weights @ ratios).tolist()]


def loss_features(sample):
    """Molar-weighted retention moments, coverage and carrier loss assumption."""
    from .physical_priors import carrier_loss
    temperature = float(sample['pulse_limit_C'])
    pulses = sample['pulse_count']
    weights = np.array([sample['concentrations_mM'][e] for e in sample['elements']], dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Positive recorded concentrations required for physical features.')
    weights /= weights.sum()
    retained, available = [], []
    for element in sample['elements']:
        row = elemental_loss_reference(element)
        known = any(positive(row.get(k)) for k in THERMAL_FIELDS)
        if known:
            r, detail = metal_retention(element, temperature, pulses, anchor_retention())
            retained.append(r)
            available.append(detail['reference_weight_coverage'])
        else:
            retained.append(0.)
            available.append(0.)
    retained = np.asarray(retained)
    # Zeros at unknown memberships are placeholders accompanied by coverage.
    mean = float(weights @ retained)
    sd = float(np.sqrt(max(weights @ (retained * retained) - mean * mean, 0.)))
    return [mean, sd, float(weights @ available), carrier_loss(temperature, pulses)]


def physics_manifest(profile):
    enabled = profile in THERMAL_PROFILES
    from .mixing import reference_identity
    from .pulseicp_bridge import identity as pulseicp_identity
    return {'version': VERSION, 'thermal_descriptors_enabled': enabled,
            'pulseicp': pulseicp_identity() if profile == 'chemistry_pulseicp' else None,
            'empirical_loss_features_enabled': profile == 'chemistry_loss',
            'mixing_descriptors_enabled': profile in {'chemistry_mixing', 'chemistry_pulseicp'},
            'mixing_reference': reference_identity() if profile in {'chemistry_mixing', 'chemistry_pulseicp'} else None,
            'thermal_reference_sha256': loss_reference()['snapshot_sha256'] if enabled else None,
            'loss_settings_sha256': loss_settings()['settings_sha256'] if profile == 'chemistry_loss' else None,
            'note': 'Frozen sourced elemental bulk properties; missing masks retained. Gas-atom formation energy is an escape-energy proxy, not a sublimation temperature or SAC binding barrier. Loss coefficients and carrier anchors are unfitted hypotheses. No assay or measured peak features; no automatic model promotion.'}


def thermal_manifest_rows(profile):
    source = loss_reference()
    rows = [{'descriptor': field, 'unit': 'K' if field.endswith('_K') else 'kJ/mol',
             'enabled': profile in THERMAL_PROFILES,
             'requirement': 'Sourced positive value; unknown remains masked',
             'source': source['documentation'] + '; ' + source['bibliography']} for field in THERMAL_FIELDS]
    rows.append({'descriptor': 'Empirical metal-retention moments and carrier loss', 'unit': 'fraction / encoding',
                 'enabled': profile == 'chemistry_loss', 'requirement': 'Controlled cutoff and integer pulses; unfitted loss coefficients',
                 'source': 'thermal_loss.py; empirical_loss_settings.json; user loss anchors'})
    return rows


def material_assumptions(physics):
    """Require explicit common-run assumptions; never substitute DEMO defaults."""
    if not isinstance(physics, dict) or physics.get('confirmed') is not True:
        raise ValueError('Confirm material assumptions before using a physical ICP prior.')
    if not str(physics.get('source', '')).strip():
        raise ValueError('Record the source and scope of the material assumptions.')
    try:
        values = {k: float(physics[k]) for k in MATERIAL_FIELDS}
    except (KeyError, ValueError, TypeError):
        raise ValueError('PAN mass, solvent volume and carbonization loss are required.') from None
    if not all(math.isfinite(v) for v in values.values()) or values['pan_mass_g'] <= 0 or values['solvent_volume_mL'] <= 0 or not 0 <= values['carbonization_loss'] < 1:
        raise ValueError('Material masses/volume must be positive; carbonization loss must be in [0, 1).')
    return values


def loading_prior(sample, element, physics):
    from .physical_priors import mass_balance
    values = material_assumptions(physics)
    # A shared product denominator requires references for every mixture member.
    for member in sample['elements']:
        row = elemental_loss_reference(member)
        if not any(positive(row.get(k)) for k in THERMAL_FIELDS):
            raise ValueError(f'No sourced loss reference for mixture member {member}.')
    # Recorded preparation takes precedence over confirmed common defaults.
    for key in ('pan_mass_g', 'solvent_volume_mL'):
        if sample.get(key) is not None:
            values[key] = sample[key]
    from .pulseicp_bridge import composition
    from ._pulseicp.composition import decode_composition
    from ._pulseicp.configuration import physics_settings
    prepared = {**sample, **values, 'carbonization_loss_fraction': sample.get('carbonization_loss_fraction', values['carbonization_loss'])}
    recipe = composition(prepared)
    return float(decode_composition(recipe, dict.fromkeys(recipe.elements, 0.), physics_settings())[element])


def prepare_loading_priors(records, element, physics, per_member=False):
    if physics is None:
        return records, {}, {'enabled': False}
    values = material_assumptions(physics)
    kept, priors, excluded = [], {}, []
    for sample, y, x in records:
        try:
            target = sample['elements'][0] if per_member else element
            prior = loading_prior(sample, target, physics)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            excluded.append({'sample_id': sample['sample_id'], 'reason': str(exc)})
            continue
        priors[sample['sample_id']] = prior
        kept.append((sample, y, x))
    canonical = json.dumps(physics, sort_keys=True, allow_nan=False).encode()
    return kept, priors, {'enabled': True, 'assumptions': {**physics, **values},
        'assumptions_sha256': hashlib.sha256(canonical).hexdigest(), 'exclusions': excluded,
        'settings_sha256': loss_settings()['settings_sha256'],
        'reference_sha256': loss_reference()['snapshot_sha256'],
        'formula': 'z = log1p(W_observed) - log1p(W_prior); W_prediction = clip(expm1(log1p(W_prior) + z_prediction), 0, 100)',
        'note': 'Prior coefficients are frozen, not fitted on held-out outcomes. All direct/residual/prior comparisons use the same eligible rows and outer folds. Residual fits and transformations use training rows only. GP intervals exclude prior uncertainty. Common material assumptions apply only to explicitly confirmed comparable experiments. DEMO agreement is not experimental validation.'}


def compare_loading_predictions(frame):
    from .loading import regression_metrics
    if frame.empty or 'direct_predicted_w_wt_percent' not in frame:
        return []
    result = []
    for name, column in [('Direct regression', 'direct_predicted_w_wt_percent'),
                         ('Unfitted physical prior', 'physical_prior_w_wt_percent'),
                         ('Physics residual', 'predicted_w_wt_percent')]:
        part = frame.copy()
        part['predicted_w_wt_percent'] = frame[column]
        if name != 'Physics residual':
            part[['lower_95_w_wt_percent', 'upper_95_w_wt_percent']] = np.nan
        result.append({'variant': name, **regression_metrics(part)})
    return result
