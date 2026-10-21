"""Restore missing planning defaults without overwriting entered operating ranges."""
from __future__ import annotations

import copy

from .domain import base_concentration, cohort_key


def restore_operating_space(saved, context, samples, origin):
    """Merge sparse upgrade settings with namespace-specific, direct-mM defaults.

    REAL fallback ranges are fixed references and require review when a saved
    configuration lacks ranges. DEMO ranges illustrate the workflow only.
    Explicit invalid ranges remain invalid so validation can request correction.
    """
    bounds = {}
    for element in context['elements']:
        reference = context['concentrations_mM'].get(element) or base_concentration(element)
        observed=[s['concentrations_mM'].get(element) for s in samples if origin=='DEMO' and s.get('origin')==origin
            and cohort_key(s)==cohort_key(context) and s['concentrations_mM'].get(element) is not None]
        bounds[element] = {
            'min_mM': reference if origin == 'REAL' else min(observed or [reference*.3]),
            'max_mM': reference if origin == 'REAL' else max(observed or [reference*3]),
        }
    defaults = {
        'concentration_bounds_mM': bounds,
        'pulses': [1] if origin == 'REAL' else [1, 2, 3, 4, 6, 8],
        'max_total_mM': max(2.75, sum(b['max_mM'] for b in bounds.values())),
        'verified': True,
        'baseline_repeat': True,
        'range_source': 'Existing reference concentrations only; one pulse' if origin == 'REAL' else 'Simulated range; demonstration only',
    }
    cutoff = context.get('pulse_limit_C')
    if cutoff is not None:
        observed = [s['pulse_limit_C'] for s in samples
                    if origin == 'DEMO' and s.get('origin') == origin
                    and cohort_key(s) == cohort_key(context) and s.get('pulse_limit_C') is not None]
        defaults.update(pulse_limit_min_C=min(observed or [cutoff]),
                        pulse_limit_max_C=max(observed or [cutoff]))
    if saved is None:
        return defaults
    saved = copy.deepcopy(saved)
    has_legacy = 'alpha_min' in saved or 'alpha_max' in saved
    if has_legacy and 'concentration_bounds_mM' not in saved:
        defaults.pop('concentration_bounds_mM')
    if 'pulse_limit_min_C' in saved or 'pulse_limit_max_C' in saved:
        defaults.pop('pulse_limit_min_C', None)
        defaults.pop('pulse_limit_max_C', None)
    if origin == 'REAL' and 'concentration_bounds_mM' not in saved and not has_legacy:
        defaults['verified'] = False
        saved['verified'] = False
    return {**defaults, **saved}


def restore_planning_space(saved, context, samples, methods, origin, campaign_policy=None):
    """Apply runtime campaign constraints to the same restored operating space."""
    space = restore_operating_space(saved, context, samples, origin)
    space.pop('_single_element_campaign', None)
    if (campaign_policy or {}).get('enabled'):
        from .element_campaign import campaign_progress
        from .domain import TARGETS
        covered = int(campaign_progress(samples, methods, origin).candidate_covered.sum())
        space['_single_element_campaign'] = {'enabled': True, 'covered': covered,
            'total': len(TARGETS), 'ready': covered == len(TARGETS)}
        space['element_transfer'] = True
    return space
