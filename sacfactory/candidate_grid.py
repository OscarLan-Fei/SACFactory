"""Generate executable integer cutoffs before predicting or deduplicating recipes."""
import math

import numpy as np


def temperature_step(space):
    if isinstance(space.get('temperature_step_C'), (bool, np.bool_)):
        raise ValueError('Temperature step must be a positive integer in degrees Celsius.')
    value = float(space.get('temperature_step_C', 1))
    if not math.isfinite(value) or value < 1 or not value.is_integer():
        raise ValueError('Temperature step must be a positive integer in degrees Celsius.')
    return int(value)


def on_temperature_grid(value, space):
    if value is None:
        return False
    step = temperature_step(space)
    return math.isclose(float(value) / step, round(float(value) / step), abs_tol=1e-9)


def cutoff_candidates(lower, upper, space, count=6, anchors=()):
    lower, upper = float(lower), float(upper)
    step = temperature_step(space)
    first, last = math.ceil(lower / step), math.floor(upper / step)
    if first > last:
        raise ValueError('No integer cutoff matches the selected temperature step inside these bounds. Adjust the step or bounds.')
    budget = min(last - first + 1, max(5, min(25, 2 * int(count) + 1)))
    ticks = set(np.rint(np.linspace(first, last, budget)).astype(int).tolist())
    for anchor in anchors:
        if anchor is not None and math.isfinite(float(anchor)):
            ticks.add(min(last, max(first, int(round(float(anchor) / step)))))
    return [int(tick * step) for tick in sorted(ticks)]
