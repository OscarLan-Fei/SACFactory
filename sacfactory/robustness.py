"""Finite-scenario sensitivity to user-specified input tolerances; no safety claim."""
import copy
import numpy as np
from scipy.stats import qmc

from .composition import model_feature


def robustness_summary(model, recipe, space, scenarios=16):
    relative = float(space.get('concentration_tolerance_fraction', 0))
    temperature = float(space.get('cutoff_tolerance_C', 0))
    if not 0 <= relative <= .5 or not 0 <= temperature <= 200:
        raise ValueError('Concentration tolerance must be 0..0.5; cutoff tolerance 0..200 C.')
    if relative == 0 and temperature == 0:
        return None
    bounds = space['concentration_bounds_mM']
    members = recipe['elements']
    perturbations = 2*qmc.Sobol(len(members)+1, scramble=True, seed=42).random_base2(4)-1
    candidates = [recipe]
    for direction in perturbations[:scenarios]:
        proposed = copy.deepcopy(recipe)
        proposed['concentrations_mM'] = {e: recipe['concentrations_mM'][e]*(1+relative*direction[i]) for i, e in enumerate(members)}
        proposed['pulse_limit_C'] = recipe['pulse_limit_C']+temperature*direction[-1]
        if any(not bounds[e]['min_mM'] <= proposed['concentrations_mM'][e] <= bounds[e]['max_mM'] for e in members):
            continue
        if sum(proposed['concentrations_mM'].values()) > space['max_total_mM'] or not space['pulse_limit_min_C'] <= proposed['pulse_limit_C'] <= space['pulse_limit_max_C']:
            continue
        candidates.append(proposed)
    p = model.predict_proba(np.array([model_feature(r, model) for r in candidates]))[:, 1]
    return {'nominal_probability': float(p[0]), 'scenario_min_probability': float(p.min()),
        'scenario_mean_probability': float(p.mean()), 'scenario_probability_range': float(np.ptp(p)),
        'valid_scenarios': len(candidates)-1, 'requested_scenarios': len(perturbations[:scenarios]),
        'concentration_tolerance_fraction': relative, 'cutoff_tolerance_C': temperature,
        'note': 'Finite Sobol scenario sensitivity with user-specified tolerances. Out-of-bound scenarios are rejected, not clamped. Sparse retained coverage, especially at boundaries, limits interpretation; this is not a certified worst-case or safe region.'}
