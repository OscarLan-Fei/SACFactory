"""Names for the user-declared shared PAN/DMF pulse-heating workflow."""
import copy

from .precursors import documented

PROCESS_DEFAULTS = {'precursor_family': 'METAL_SALTS_V1',
                    'support_batch': 'PAN_DERIVED_CARBON',
                    'protocol_id': 'PAN_DMF_PULSE_V1'}
PROCESS_BASIS = 'User-declared common metal-salt/PAN/DMF pulse-heating workflow; shared labels do not establish identical runs.'


def shared_process_values(context):
    return {key: context[key] if documented(context.get(key)) else value for key, value in PROCESS_DEFAULTS.items()}


def link_shared_process(sample):
    result = copy.deepcopy(sample)
    values = shared_process_values(sample)
    if any(values[key] != sample.get(key) for key in values):
        result.update(values, process_metadata_basis=PROCESS_BASIS)
    return result


def fixed_search_space(space, elements):
    """Explain a collapsed allowed range without inventing new device limits."""
    from .planning import concentration_bounds
    bounds = concentration_bounds(space, elements)
    fixed = all(b['min_mM'] == b['max_mM'] for b in bounds.values())
    pulses = sorted(set(space.get('pulses', [])))
    temperatures_fixed = space.get('pulse_limit_min_C') == space.get('pulse_limit_max_C')
    if fixed and len(pulses) == 1 and temperatures_fixed:
        return ('Only one distinct recipe is allowed: all precursor concentrations, pulse count and cutoff temperature '
                'are fixed. Increase at least one allowed range and save it to request different recipes. '
                'Also check the total-concentration cap and open Pending reservations. Existing recipes are deduplicated; '
                'repeats are labeled separately. Fitting gates do not set the requested batch size.')
    return None
