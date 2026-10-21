"""Unfitted physical priors and mass accounting; never measured retention values."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .chemical_descriptors import reference
from .composition import POSITIONS
from .thermal_loss import metal_retention, loss_settings, loss_reference


@lru_cache(maxsize=1)
def thermal_reference():
    raw = (Path(__file__).resolve().parents[1] / 'resources/thermal_properties.json').read_bytes()
    payload = json.loads(raw)
    payload['snapshot_sha256'] = hashlib.sha256(raw).hexdigest()
    return payload


def thermal_properties(element):
    rows = [r for r in thermal_reference()['phase_transitions'] if r['symbol'] == element and not r.get('is_transition')]
    result = {}
    for field in ('melting_point', 'boiling_point'):
        available = [r for r in rows if r.get(field) is not None]
        # Preserve ambiguity rather than average incompatible allotropes.
        unique = {(r[field], r.get('is_sublimation_point')) for r in available}
        result[field] = next(iter(unique))[0] if len(unique) == 1 else None
        result[field + '_sublimation'] = next(iter(unique))[1] if len(unique) == 1 else None
    return result


def dose_factor(element):
    return 1.8 if POSITIONS[element][1] in (1, 2) else 1.0


def mmol_to_mM(amount_mmol, volume_mL=20.0):
    if amount_mmol < 0 or volume_mL <= 0:
        raise ValueError('Nonnegative amount and positive volume are required.')
    return amount_mmol * 1000.0 / volume_mL


def carrier_loss(temperature_C, pulse_count):
    from ._pulseicp.carbon_prior import matrix_loss
    from ._pulseicp.configuration import physics_settings
    return matrix_loss(temperature_C, pulse_count, physics_settings())


def legacy_carrier_loss(temperature_C, pulse_count):
    """Archived v0.7 prior; retained for reproducibility only."""
    if pulse_count < 1 or int(pulse_count) != pulse_count:
        raise ValueError('Positive integer pulse count required.')
    # One-pulse anchors are user-reported approximate losses after carbonization.
    # The 700 C anchor and temperature-dependent plateau/tau are assumptions.
    first = float(np.interp(temperature_C, [700, 1000, 1500, 2000], [.10, .13, .15, .18]))
    settings = loss_settings()
    ratio = (float(temperature_C) + 273.15) / (settings['reference_temperature_C'] + 273.15)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('Temperature must be finite and above absolute zero.')
    increment = settings['carrier_plateau_increment_at_reference'] * ratio ** settings['carrier_increment_temperature_exponent']
    plateau = min(settings['carrier_loss_cap'], first + increment)
    tau = settings['reference_tau_pulses'] / ratio ** settings['carrier_tau_temperature_exponent']
    return first + (plateau - first) * (-math.expm1(-(pulse_count - 1) / tau))


def mass_balance(concentrations_mM, temperature_C, pulse_count, rng=None,
                 pan_mass_g=2.0, solvent_volume_mL=20.0, carbonization_loss=.06):
    """Return elemental masses and true W with one shared sample denominator.

    The remaining carrier is the empirically retained nonmetal matrix, not PAN's
    stoichiometric carbon fraction. Counterions/other residuals are not modeled.
    Retention parameters are illustrative and identifiable only with more data.
    """
    if not all(math.isfinite(float(v)) for v in (pan_mass_g,solvent_volume_mL,carbonization_loss)) or pan_mass_g <= 0 or solvent_volume_mL <= 0 or not 0 <= carbonization_loss < 1:
        raise ValueError('Invalid material balance inputs.')
    if not concentrations_mM or any(not math.isfinite(float(c)) or c <= 0 for c in concentrations_mM.values()):
        raise ValueError('Positive concentrations required.')
    atoms = {r['symbol']: r for r in reference()['elements']}
    carrier_before = pan_mass_g * (1 - carbonization_loss)
    carrier_after = carrier_before * (1 - carrier_loss(temperature_C, pulse_count))
    from ._pulseicp.configuration import physics_settings
    shared = physics_settings()
    # A fixed reference anchor must not change with the query PAN mass.
    baseline_carrier = shared['anchor_pan_g'] * (1 - shared['anchor_carbonization_loss']) * (1 - shared['carrier_one_pulse_losses'][0])
    anchor_w = shared['anchor_fe_w_wt_percent'] / 100.
    target_metal = anchor_w / (1 - anchor_w) * baseline_carrier
    anchor_feed = shared['anchor_fe_mM'] * shared['anchor_volume_mL'] * atoms['Fe']['atomic_weight'] / 1e6
    anchor_retention = target_metal / anchor_feed
    effective_dose = sum(c * solvent_volume_mL / 1000. / dose_factor(e) for e, c in concentrations_mM.items())
    entries = {}
    for element, concentration in concentrations_mM.items():
        amount = concentration * solvent_volume_mL / 1000.
        feed = amount * atoms[element]['atomic_weight'] / 1000.
        phase = thermal_properties(element)
        model_retention, loss_parameters = metal_retention(element, temperature_C, pulse_count, anchor_retention)
        noise = math.exp(float(rng.normal(0, .012))) if rng is not None else 1.0
        retention = min(1.0, max(0.0, model_retention * noise))
        retained = feed * retention
        entries[element] = {'amount_mmol': amount, 'feed_metal_g': feed,
            'retained_metal_g': retained, 'lost_metal_g': feed - retained,
            'retention_fraction': retention, 'thermal_reference': phase,
            'empirical_loss_parameters': loss_parameters}
    total_metal = sum(row['retained_metal_g'] for row in entries.values())
    recovered = carrier_after + total_metal
    for row in entries.values():
        row['true_w_wt_percent'] = 100 * row['retained_metal_g'] / recovered
    return {'loss_model_version': loss_settings()['version'],
        'loss_settings_sha256': loss_settings()['settings_sha256'],
        'loss_reference_sha256': loss_reference()['snapshot_sha256'],
        'pan_mass_g': pan_mass_g, 'solvent_volume_mL': solvent_volume_mL,
        'carbonization_loss_fraction': carbonization_loss, 'carrier_before_pulse_g': carrier_before,
        'carrier_loss_fraction': carrier_loss(temperature_C, pulse_count), 'carrier_after_pulse_g': carrier_after,
        'recovered_sample_g': recovered, 'total_retained_metal_g': total_metal,
        'effective_total_dose_mmol': effective_dose, 'elements': entries}
