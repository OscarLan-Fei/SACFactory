"""Independent configuration loading and reproducible hashes."""
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def physics_settings():
    settings = json.loads((ROOT/'configs/physics.json').read_text())
    validate_physics(settings)
    return settings


def model_settings():
    return json.loads((ROOT/'configs/model.json').read_text())


def validate_physics(settings):
    weights = settings['thermal_weights']
    if set(weights) != {'melting', 'boiling', 'escape'} or any(not math.isfinite(float(v)) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError('Thermal weights must be finite, nonnegative and have positive sum.')
    for key in ('temperature_exponent', 'metal_tau_pulses', 'carrier_tau_pulses',
                'reference_pressure_Pa', 'anchor_pan_g', 'anchor_volume_mL', 'anchor_fe_mM'):
        if not math.isfinite(float(settings[key])) or settings[key] <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    if not 0 < settings['reference_repeat_ratio'] < 1 or not 0 < settings['carrier_loss_cap'] < 1:
        raise ValueError('Retention ratio and carrier cap must be in (0,1).')
    if not 0 <= settings['anchor_carbonization_loss'] < 1 or not 0 < settings['anchor_fe_w_wt_percent'] < 100:
        raise ValueError('The reference material/ICP anchor is invalid.')
    if settings['reference_temperature_C'] <= -273.15:
        raise ValueError('Reference temperature must be above absolute zero.')
    return copy.deepcopy(settings)
