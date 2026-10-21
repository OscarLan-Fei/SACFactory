"""Allocate bounded next-batch repeat slots and an adaptive future repeat budget.

One instance of each recipe is proposed per preview. Additional instances are
reconsidered after returned results, preserving pending recipe deduplication.
"""
import copy
import numpy as np

from .domain import recipe_key
from .recipe_reliability import recipe_evidence


def future_repeat_budget(a, b, batches, half_width=.15, cap=20, minimum_batches=3):
    if not .05 <= half_width <= .45 or not 1 <= cap <= 100:
        raise ValueError('Repeat half-width must be 0.05..0.45 and cap 1..100.')
    total = a+b
    variance = a*b/(total*total*(total+1))
    for m in range(cap+1):
        expected_variance = variance*total/(total+m)
        scale = 1.96*np.sqrt(expected_variance)
        if scale <= half_width and batches+m >= minimum_batches:
            return m, float(scale), False
    return cap, float(1.96*np.sqrt(variance*total/(total+cap))), True


def allocate_repetitions(state, context, bounds, pulses, tmin, tmax, max_total, reserved, count, space):
    fraction = float(space.get('repeat_budget_fraction', .5))
    if not 0 <= fraction <= 1:
        raise ValueError('Repeat budget fraction must be between 0 and 1.')
    limit = min(count, int(np.floor(count*fraction)))
    goal, cap = float(space.get('repeat_goal_half_width', .15)), int(space.get('repeat_cap', 20))
    candidates = []
    for evidence in recipe_evidence(state.training, state.model.methods):
        recipe = evidence['recipe']
        from .candidate_grid import on_temperature_grid
        if not on_temperature_grid(recipe.get('pulse_limit_C'), space):
            continue
        if set(recipe['elements']) != set(context['elements']) or recipe_key(recipe) in reserved:
            continue
        if recipe['pulse_count'] not in pulses or not tmin <= recipe['pulse_limit_C'] <= tmax or sum(recipe['concentrations_mM'].values()) > max_total:
            continue
        if not all(bounds[e]['min_mM'] <= recipe['concentrations_mM'][e] <= bounds[e]['max_mM'] for e in context['elements']):
            continue
        m, precision, capped = future_repeat_budget(evidence['a'], evidence['b'], evidence['independent_batches'], goal, cap)
        if not m:
            continue
        total = evidence['a']+evidence['b']
        variance = evidence['a']*evidence['b']/(total*total*(total+1))
        value = variance/(total+1)*(1+2*evidence['between_batch_rate_variance'])
        rationale = {k: copy.deepcopy(v) for k, v in evidence.items() if k != 'recipe'}
        rationale.update(planned_additional_repeats=m, further_repeats_after_this=m-1, repeat_goal_half_width=goal,
            expected_interval_scale=precision, repeat_cap_reached=capped, precision_value=float(value),
            schedule='One instance per recipe per preview; use a new independent batch; reconsider after results.',
            stopping_note='Adaptive learning budget using working beta expected variance; not guaranteed CI coverage or fixed validation allocation.')
        candidate = {k: copy.deepcopy(recipe[k]) for k in ('origin', 'elements', 'concentrations_mM', 'pulse_count', 'pulse_limit_C',
            'precursor_family', 'support_batch', 'protocol_id')}
        candidate['temperature_C'] = None
        if 'precursors' in recipe:
            candidate['precursors'] = copy.deepcopy(recipe['precursors'])
        candidates.append((candidate, rationale))
    candidates.sort(key=lambda item: (-item[1]['precision_value'], item[1]['recipe_id']))
    return candidates[:limit]
