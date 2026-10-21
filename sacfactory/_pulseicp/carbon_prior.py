"""Temperature and pulse-dependent remaining-matrix prior."""
import math
import numpy as np


def matrix_loss(temperature_C, pulses, settings):
    temperature = float(temperature_C)
    if not math.isfinite(temperature) or temperature <= -273.15:
        raise ValueError('Temperature must be finite and above absolute zero.')
    if isinstance(pulses, bool) or not float(pulses).is_integer() or not 1 <= pulses:
        raise ValueError('Positive integer pulses required.')
    anchors = np.asarray(settings['carrier_anchors_C'], dtype=float)
    losses = np.asarray(settings['carrier_one_pulse_losses'], dtype=float)
    if len(anchors) < 2 or len(anchors) != len(losses) or np.any(np.diff(anchors) <= 0) or not np.isfinite(losses).all():
        raise ValueError('Carrier anchors must be finite, ordered pairs.')
    first = float(np.interp(temperature, anchors, losses))
    if temperature < anchors[0]:
        first = float(losses[0] + (temperature-anchors[0]) * (losses[1]-losses[0]) / (anchors[1]-anchors[0]))
    if temperature > anchors[-1]:
        first = float(losses[-1] + (temperature-anchors[-1]) * (losses[-1]-losses[-2]) / (anchors[-1]-anchors[-2]))
    ratio = (temperature+273.15) / (settings['reference_temperature_C']+273.15)
    cap = settings['carrier_loss_cap']
    if not 0 <= first < cap < 1:
        raise ValueError('Extrapolated carrier loss is incompatible with its declared cap.')
    increment = settings['carrier_plateau_increment'] * ratio ** settings['carrier_increment_exponent']
    plateau = min(cap, first + increment)
    tau = settings['carrier_tau_pulses'] / ratio ** settings['carrier_tau_exponent']
    if tau <= 0 or increment < 0:
        raise ValueError('Carrier timescale and increment must be valid.')
    return first + (plateau-first) * (-math.expm1(-(pulses-1)/tau))


def matrix_mass(recipe, settings):
    return recipe.pan_mass_g * (1-recipe.carbonization_loss_fraction) * (1-matrix_loss(recipe.temperature_C, recipe.pulse_count, settings))
