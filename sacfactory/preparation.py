"""Declared preparation inputs and explicit mmol/mM conversion."""
import math
import re

import numpy as np

MATERIAL_FIELDS = ('pan_mass_g', 'solvent_volume_mL')
DEFAULT_PAN_G = 2.0
DEFAULT_DMF_ML = 20.0
DEFAULT_VOLTAGE_V = 32.0
DEFAULT_CURRENT_A = 80.0


def positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be a positive finite number.')
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(name + ' must be a positive finite number.') from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError(name + ' must be a positive finite number.')
    return result


def material_inputs(pan=DEFAULT_PAN_G, volume=DEFAULT_DMF_ML):
    return {'pan_mass_g': positive(pan, 'PAN mass / g'),
            'solvent_volume_mL': positive(volume, 'DMF volume / mL')}


def to_mM(value, unit='mM', volume_mL=DEFAULT_DMF_ML):
    if value is None or str(value).strip().lower() in {'', 'nan', '<na>', 'none'}:
        return None
    value = positive(value, 'Precursor amount / concentration')
    if unit not in {'mM', 'mmol'}:
        raise ValueError('Select mM or mmol for precursor input.')
    return value if unit == 'mM' else value * 1000. / positive(volume_mL, 'DMF volume / mL')


def from_mM(value, unit='mM', volume_mL=DEFAULT_DMF_ML):
    if value is None:
        return None
    concentration = positive(value, 'Precursor concentration / mM')
    if unit not in {'mM', 'mmol'}:
        raise ValueError('Select mM or mmol for precursor input.')
    return concentration if unit == 'mM' else concentration * positive(volume_mL, 'DMF volume / mL') / 1000.


def precursor_value(value, unit='mM', volume_mL=DEFAULT_DMF_ML):
    """Parse a metal-salt input, allowing an explicit unit to override the selector."""
    if value is None or str(value).strip().lower() in {'', 'nan', '<na>', 'none'}:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Precursor amount / concentration must be a positive finite number.')
    match = re.fullmatch(r'\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)\s*(mM|mmol)?\s*', str(value))
    if not match:
        raise ValueError('Enter a positive number, optionally followed by mM or mmol, such as 50 mM or 1 mmol.')
    return to_mM(float(match[1]), match[2] or unit, volume_mL)


def precursor_cell(value, elements, unit='mM', volume_mL=DEFAULT_DMF_ML):
    """Return canonical mM per selected element, or None for detail-table input."""
    text = '' if value is None else str(value).strip()
    if not text or text.upper() == 'DETAIL':
        return None
    if ':' not in text:
        concentration = precursor_value(text, unit, volume_mL)
        return {element: concentration for element in elements}
    from .batching import pairs
    supplied = pairs(text, 'Precursor input')
    if set(supplied) != set(elements):
        raise ValueError('Precursor pairs must match every selected target element.')
    return {element: precursor_value(supplied[element], unit, volume_mL) for element in elements}


def precursor_summary(values, elements):
    """Represent the displayed per-element inputs without rounding stored values."""
    if any(values.get(e) is None or str(values[e]).strip().lower() in {'', 'nan', 'none', '<na>'} for e in elements):
        return 'DETAIL'
    numbers = [positive(values[e], 'Precursor amount / concentration') for e in elements]
    if all(v == numbers[0] for v in numbers):
        return format(numbers[0], '.15g')
    return ';'.join(f'{e}:{v:.15g}' for e, v in zip(elements, numbers))


def identity(sample):
    values = tuple(sample.get(k) for k in MATERIAL_FIELDS)
    return ('DECLARED_PREPARATION', *values) if any(v is not None and v != '' for v in values) else ()


def copy_preparation(source, target):
    for key in (*MATERIAL_FIELDS, 'preparation_basis'):
        if key in source:
            target[key] = source[key]


def model_features(sample):
    values, known = [], []
    for field, reference in zip(MATERIAL_FIELDS, (DEFAULT_PAN_G, DEFAULT_DMF_ML)):
        value = sample.get(field)
        present = value is not None and value != ''
        values.append(math.log(positive(value, field) / reference) if present else 0.)
        known.append(float(present))
    burden = (sum(sample['concentrations_mM'].values()) * sample['solvent_volume_mL'] /
              (1000. * sample['pan_mass_g'])) if all(known) else None
    return [values[0], known[0], values[1], known[1], math.log1p(burden) if burden is not None else 0., float(burden is not None)]
