"""Single-element recipe and observation contracts; no implicit real-data fills."""
from dataclasses import dataclass, asdict
import math

from .references import supported_elements

MATERIAL_COLUMNS = ['pan_mass_g', 'volume_mL', 'carbonization_loss_fraction']
RECIPE_COLUMNS = ['element', 'concentration_mM', 'temperature_C', 'pulse_count'] + MATERIAL_COLUMNS
COLUMNS = ['sample_id', 'batch_id'] + RECIPE_COLUMNS + [
    'icp_w_wt_percent', 'icp_flag', 'origin', 'protocol_id', 'result_reported_at']


def numeric(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be numeric, not a boolean.')
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be explicitly recorded as a number.') from None
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite.')
    return number


@dataclass(frozen=True)
class Recipe:
    element: str = 'Fe'
    concentration_mM: float = 50.0
    temperature_C: float = 1000.0
    pulse_count: int = 1
    pan_mass_g: float = 2.0
    volume_mL: float = 20.0
    carbonization_loss_fraction: float = 0.06

    def validate(self):
        if self.element not in supported_elements():
            raise ValueError(f'{self.element}: the metal-target model needs a sourced atomic weight and at least one usable thermal channel. Missing fundamental inputs are not guessed.')
        values = {k: numeric(v, k) for k, v in asdict(self).items() if k != 'element'}
        if values['temperature_C'] <= -273.15:
            raise ValueError('Temperature must be finite and above absolute zero (-273.15 C).')
        n = values['pulse_count']
        if not n.is_integer() or n < 1:
            raise ValueError('Pulse count must be a positive integer.')
        if any(values[k] <= 0 for k in ('concentration_mM', 'pan_mass_g', 'volume_mL')):
            raise ValueError('Concentration, PAN mass and solvent volume must be positive.')
        if not 0 <= values['carbonization_loss_fraction'] < 1:
            raise ValueError('Carbonization loss must be a fraction in [0, 1).')
        return self

    @classmethod
    def from_row(cls, row):
        from .preparation import normalize_preparation
        row = normalize_preparation(row)
        missing = [k for k in RECIPE_COLUMNS if k not in row or row[k] is None or str(row[k]).strip() in {'', 'nan', '<NA>'}]
        if missing:
            raise ValueError('Missing recipe inputs: ' + ', '.join(missing))
        args = {k: numeric(row[k], k) for k in RECIPE_COLUMNS if k != 'element'}
        # Validate before converting the integer so fractional pulses never truncate.
        recipe = cls(element=str(row['element']).strip(), **args).validate()
        return cls(**{**asdict(recipe), 'pulse_count': int(recipe.pulse_count)})


def control_domain_columns(recipe):
    """Describe the original demonstration range; never use it as an input cap."""
    outside = not (800. <= recipe.temperature_C <= 1800. and 1 <= recipe.pulse_count <= 16)
    return {'outside_original_demo_window': outside,
        'control_domain_status': 'BEYOND_ORIGINAL_DEMO_WINDOW' if outside else 'ORIGINAL_DEMO_WINDOW'}
