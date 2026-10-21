"""Temperature-dependent empirical loss priors with explicit reference coverage."""
from functools import lru_cache
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

VERSION = 'temperature-dependent-plateau-loss-v2'
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / 'resources'


@lru_cache(maxsize=1)
def loss_reference():
    raw = (RESOURCE_ROOT / 'loss_thermochemistry.json').read_bytes()
    result = json.loads(raw)
    result['snapshot_sha256'] = hashlib.sha256(raw).hexdigest()
    return result


@lru_cache(maxsize=1)
def loss_settings():
    raw = (RESOURCE_ROOT / 'empirical_loss_settings.json').read_bytes()
    result = json.loads(raw)
    from ._pulseicp.configuration import physics_settings
    shared = physics_settings()
    # Legacy consumers read the same physical parameters as the migrated engine.
    aliases = {'reference_repeat_retention_ratio':'reference_repeat_ratio',
        'reference_tau_pulses':'metal_tau_pulses', 'tau_risk_exponent':'metal_tau_exponent',
        'carrier_plateau_increment_at_reference':'carrier_plateau_increment',
        'carrier_increment_temperature_exponent':'carrier_increment_exponent',
        'carrier_tau_temperature_exponent':'carrier_tau_exponent'}
    for key in ('reference_temperature_C', 'reference_pressure_Pa', 'thermal_weights', 'temperature_exponent', 'carrier_loss_cap'):
        result[key] = shared[key]
    result.update({key:shared[value] for key,value in aliases.items()})
    result['settings_sha256'] = hashlib.sha256(raw + json.dumps(shared, sort_keys=True, allow_nan=False).encode()).hexdigest()
    result['shared_parameter_source'] = 'sacfactory/_pulseicp/configs/physics.json'
    return result


def elemental_loss_reference(element):
    return dict(next((r for r in loss_reference()['entries'] if r['element'] == element), {'element': element}))


def positive(value):
    return value is not None and isinstance(value, Real) and math.isfinite(value) and value > 0


def thermal_risk(element, temperature_C, properties=None, settings=None):
    """Relative empirical risk; missing covariates remain absent and flagged."""
    settings = settings or loss_settings()
    temperature = float(temperature_C) + 273.15
    if not positive(temperature):
        raise ValueError('Temperature must be finite and above absolute zero.')
    if not positive(settings['temperature_exponent']) or not positive(settings['reference_temperature_C'] + 273.15):
        raise ValueError('A positive thermal exponent and valid reference temperature are required.')
    row = dict(properties) if properties is not None else elemental_loss_reference(element)
    anchor = elemental_loss_reference(settings['reference_element'])
    weights = settings['thermal_weights']
    if any(not math.isfinite(float(v)) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError('Thermal weights must be finite, nonnegative and have positive sum.')
    components = []
    for name, field in [('melting', 'melting_point_K'), ('boiling', 'boiling_point_K')]:
        if positive(row.get(field)) and positive(anchor.get(field)) and weights.get(name, 0) > 0:
            components.append({'channel': name, 'kind': field, 'weight': weights[name],
                               'value': row[field], 'reference_value': anchor[field],
                               'ratio': anchor[field] / row[field]})
    sublimation_eligible = (positive(row.get('reported_sublimation_temperature_K'))
        and positive(row.get('sublimation_pressure_Pa')) and bool(row.get('sublimation_source'))
        and math.isclose(row['sublimation_pressure_Pa'], settings['reference_pressure_Pa'], rel_tol=1e-6))
    escape_weight = weights.get('escape', 0)
    if escape_weight > 0 and sublimation_eligible:
        # Compare documented normal gas-transition temperatures at equal pressure.
        # Fe's boiling point is the reference scale, not a fabricated Fe sublimation point.
        components.append({'channel': 'escape', 'kind': 'pressure_matched_sublimation_temperature_K',
            'weight': escape_weight, 'value': row['reported_sublimation_temperature_K'],
            'reference_value': anchor['boiling_point_K'],
            'ratio': anchor['boiling_point_K'] / row['reported_sublimation_temperature_K']})
    elif escape_weight > 0 and positive(row.get('heat_of_formation_kJ_mol')) and positive(anchor.get('heat_of_formation_kJ_mol')):
        components.append({'channel': 'escape', 'kind': 'gas_atom_formation_energy_proxy_kJ_mol',
            'weight': escape_weight, 'value': row['heat_of_formation_kJ_mol'],
            'reference_value': anchor['heat_of_formation_kJ_mol'],
            'ratio': anchor['heat_of_formation_kJ_mol'] / row['heat_of_formation_kJ_mol']})
    if not components:
        raise ValueError(f'No usable sourced thermal reference for {element}.')
    available = sum(r['weight'] for r in components)
    descriptor_factor = math.exp(sum(r['weight'] * math.log(r['ratio']) for r in components) / available)
    temperature_factor = (temperature / (settings['reference_temperature_C'] + 273.15)) ** settings['temperature_exponent']
    risk = descriptor_factor * temperature_factor
    return {'thermal_risk': risk, 'descriptor_factor': descriptor_factor,
        'reference_weight_coverage': available / sum(weights.values()),
        'missing_channels': sorted(set(k for k, v in weights.items() if v > 0) - {r['channel'] for r in components}),
        'sublimation_temperature_used': sublimation_eligible and escape_weight > 0,
        'sublimation_temperature_available': positive(row.get('reported_sublimation_temperature_K')),
        'escape_proxy_used': any('energy_proxy' in r['kind'] for r in components),
        'components': components}


def metal_loss_parameters(element, temperature_C, anchor_retention, properties=None, settings=None):
    settings = settings or loss_settings()
    if not positive(anchor_retention) or anchor_retention > 1:
        raise ValueError('Reference Fe retention must be in (0, 1].')
    risk = thermal_risk(element, temperature_C, properties, settings)
    g = risk['thermal_risk']
    ratio = settings['reference_repeat_retention_ratio']
    if not 0 < ratio < 1 or not positive(settings['reference_tau_pulses']) or not math.isfinite(settings['tau_risk_exponent']) or settings['tau_risk_exponent'] < 0:
        raise ValueError('Plateau ratio and pulse time constant must be positive and bounded.')
    r_first = math.exp(math.log(anchor_retention) * g)
    r_plateau = r_first * math.exp(math.log(ratio) * g)
    tau = settings['reference_tau_pulses'] / g ** settings['tau_risk_exponent']
    return {**risk, 'loss_first_fraction': 1 - r_first,
        'loss_plateau_fraction': 1 - r_plateau, 'tau_pulses': tau,
        'model_version': VERSION, 'parameter_status': settings['parameter_status']}


def metal_retention(element, temperature_C, pulse_count, anchor_retention, properties=None, settings=None):
    if not positive(pulse_count) or int(pulse_count) != pulse_count:
        raise ValueError('Positive integer pulse count required.')
    parameters = metal_loss_parameters(element, temperature_C, anchor_retention, properties, settings)
    first, plateau, tau = [parameters[k] for k in ('loss_first_fraction', 'loss_plateau_fraction', 'tau_pulses')]
    loss = first + (plateau - first) * (-math.expm1(-(pulse_count - 1) / tau))
    return min(1., max(0., 1 - loss)), parameters
