"""Approximate ramp/hold/cool waveforms and Arrhenius-equivalent exposure."""
from functools import lru_cache
import math
import numpy as np

KB_EV_K = 8.617333262145e-5


def validate_profile(profile):
    mode = profile.get('heating_input_mode','AVERAGE_RATE')
    if mode not in {'AVERAGE_RATE','TIME_TO_REFERENCE_PEAK'}:
        raise ValueError('Choose an average rate or time-to-reference-peak heating description.')
    required = ('cooling_rate_min_C_s','cooling_rate_max_C_s','peak_hold_ms','baseline_temperature_C')
    required += ('heating_anchor_temperature_K','heating_anchor_time_s') if mode=='TIME_TO_REFERENCE_PEAK' else ('heating_rate_value',)
    for key in required:
        value = profile[key]
        if isinstance(value,bool) or not math.isfinite(float(value)):
            raise ValueError('Thermal profile values must be finite numbers.')
    if mode=='AVERAGE_RATE' and profile['heating_rate_unit'] not in {'C/s','C/min'}:
        raise ValueError('Heating rate unit must be C/s or C/min.')
    if mode=='TIME_TO_REFERENCE_PEAK':
        if profile['heating_anchor_time_s'] <= 0 or profile['heating_anchor_temperature_K'] <= profile['baseline_temperature_C']+273.15:
            raise ValueError('Heating anchor needs positive ramp time and a peak above the baseline in Kelvin.')
    elif profile['heating_rate_value'] <= 0:
        raise ValueError('Heating rate must be positive.')
    if profile['cooling_rate_min_C_s'] <= 0:
        raise ValueError('Heating and cooling rates must be positive.')
    if profile['cooling_rate_max_C_s'] < profile['cooling_rate_min_C_s']:
        raise ValueError('Cooling-rate bounds must be ordered.')
    if profile['peak_hold_ms'] < 0 or profile['baseline_temperature_C'] <= -273.15:
        raise ValueError('Hold time must be nonnegative and baseline above absolute zero.')
    return profile


def heating_rate_C_s(profile):
    validate_profile(profile)
    if profile.get('heating_input_mode')=='TIME_TO_REFERENCE_PEAK':
        return (profile['heating_anchor_temperature_K']-profile['baseline_temperature_C']-273.15)/profile['heating_anchor_time_s']
    return profile['heating_rate_value']/(60. if profile['heating_rate_unit']=='C/min' else 1.)


def phase_durations(peak_temperature_C, profile):
    validate_profile(profile)
    peak = float(peak_temperature_C)
    delta = peak-profile['baseline_temperature_C']
    if not math.isfinite(peak) or delta < 0:
        raise ValueError('Peak temperature must be at or above the profile baseline.')
    ramp = delta/heating_rate_C_s(profile)
    hold = profile['peak_hold_ms']/1000.
    cooling_min = delta/profile['cooling_rate_max_C_s']
    cooling_max = delta/profile['cooling_rate_min_C_s']
    return {'heating_seconds':ramp,'hold_seconds':hold,
        'cooling_seconds_min':cooling_min,'cooling_seconds_max':cooling_max,
        'cycle_seconds_min':ramp+hold+cooling_min,
        'cycle_seconds_max':ramp+hold+cooling_max}


@lru_cache(maxsize=1)
def _quadrature():
    nodes,weights = np.polynomial.legendre.leggauss(48)
    return (nodes+1)/2, weights/2


def equivalent_exposure(peak_temperature_C, profile, barrier_eV,
                        reference_temperature_C=1000., cooling_rates_C_s=None):
    """Return seconds equivalent to D(T_ref), without guessing D0 or distance.

    Integral = int exp[(Ea/kB)*(1/T_ref - 1/T(t))] dt. The linear-ramp
    approximation integrates heating, peak hold and cooling separately.
    """
    duration = phase_durations(peak_temperature_C,profile)
    barrier = np.atleast_1d(np.asarray(barrier_eV,dtype=float))
    if not np.isfinite(barrier).all() or np.any(barrier < 0) or reference_temperature_C <= -273.15:
        raise ValueError('Use nonnegative finite barriers and a physical reference temperature.')
    cooling = np.broadcast_to(np.asarray(cooling_rates_C_s if cooling_rates_C_s is not None else
        (profile['cooling_rate_min_C_s']+profile['cooling_rate_max_C_s'])/2,dtype=float),barrier.shape)
    if not np.isfinite(cooling).all() or np.any(cooling <= 0):
        raise ValueError('Cooling rates must be finite and positive.')
    nodes,weights = _quadrature()
    delta = peak_temperature_C-profile['baseline_temperature_C']
    temperatures_K = profile['baseline_temperature_C']+273.15+delta*nodes
    reference_K = reference_temperature_C+273.15
    log_diffusion = barrier[:,None]/KB_EV_K*(1/reference_K-1/temperatures_K[None,:])
    # Extreme waveforms outside numerical support are rejected, not clipped into
    # an apparently precise diffusion prediction.
    if np.max(log_diffusion) > 650:
        raise ValueError('Thermal migration exposure exceeds the numerical domain.')
    ramp_mean = np.exp(log_diffusion) @ weights
    peak_log = barrier/KB_EV_K*(1/reference_K-1/(peak_temperature_C+273.15))
    if np.max(peak_log) > 650:
        raise ValueError('Peak migration exposure exceeds the numerical domain.')
    return ramp_mean*(duration['heating_seconds']+delta/cooling) + duration['hold_seconds']*np.exp(peak_log)
