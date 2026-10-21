"""Adaptive expected-success plus joint Bernoulli information acquisition."""
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from .adaptive import entropy
from .composition import bridge_recipe, model_feature, input_extrapolation, composition_name, process_key
from .domain import RULE_VERSION, common_alpha, recipe_key


def critical_repeat(state, context, bounds, pulses, tmin, tmax, max_total, reserved, seed=42, space=None):
    """One existing target recipe for a new batch, ranked by uncertainty/instability."""
    groups = {}
    space = space or {}
    for sample in state.training:
        from .candidate_grid import on_temperature_grid
        if not on_temperature_grid(sample.get('pulse_limit_C'), space):
            continue
        if set(sample['elements']) != set(context['elements']):
            continue
        key = recipe_key(sample)
        if key in reserved or sample['pulse_count'] not in pulses or not tmin <= sample['pulse_limit_C'] <= tmax:
            continue
        if sum(sample['concentrations_mM'].values()) > max_total:
            continue
        if not all(bounds[e]['min_mM'] <= sample['concentrations_mM'][e] <= bounds[e]['max_mM'] for e in context['elements']):
            continue
        groups.setdefault(key, []).append(sample)
    if not groups:
        return None, None
    from .domain import screen
    recipes = [items[0] for items in groups.values()]
    summary = state.model.summary(np.array([model_feature(s, state.model) for s in recipes]), seed)
    diagnostics = []
    for i, items in enumerate(groups.values()):
        batch_rates = {}
        for sample in items:
            batch_rates.setdefault(sample['batch_id'], []).append(screen(sample, state.model.methods)['label'])
        rates = [np.mean(values) for values in batch_rates.values()]
        n = len(items)
        p = summary['p'][i]
        width = summary['high'][i] - summary['low'][i]
        instability = float(np.var(rates)) if len(rates) > 1 else 0.
        priority = width + 4*p*(1-p)/np.sqrt(n) + 2*instability
        diagnostics.append({'repeat_priority': float(priority), 'posterior_interval_width': float(width),
            'previous_samples': n, 'previous_independent_batches': len(rates), 'observed_batch_rate_variance': instability,
            'source_sample_ids': [s['sample_id'] for s in items],
            'rule': 'Conditional interval width + outcome ambiguity / sqrt(n) + twice observed batch-rate variance; heuristic.'})
    index = int(np.argmax([d['repeat_priority'] for d in diagnostics]))
    selected = recipes[index]
    candidate = {k: selected[k] for k in ('origin', 'elements', 'concentrations_mM', 'pulse_count', 'pulse_limit_C',
                                         'precursor_family', 'support_batch', 'protocol_id')}
    candidate['temperature_C'] = None
    from .preparation import copy_preparation
    copy_preparation(selected, candidate)
    if 'precursors' in selected:
        candidate['precursors'] = selected['precursors']
    return candidate, diagnostics[index]


def joint_information(probabilities, selected):
    """Exact label-state entropy under a finite coherent posterior sample mixture."""
    draws = probabilities[:, selected]
    conditional = np.ones((len(draws), 1))
    for p in draws.T:
        conditional = np.concatenate([conditional * (1 - p[:, None]), conditional * p[:, None]], axis=1)
    mass = conditional.mean(axis=0)
    joint_entropy = -np.sum(mass * np.log2(np.maximum(mass, 1e-300)))
    return max(0., float(joint_entropy - entropy(draws).sum(axis=1).mean()))


def greedy_batch(probabilities, expected_pass, count, information_weight):
    """Cache conditional label states so batch redundancy enters each greedy step."""
    probabilities = np.asarray(probabilities)
    count = min(count, probabilities.shape[1])
    conditional = np.ones((len(probabilities), 1))
    selected, trace = [], []
    conditional_entropy = 0.
    information = 0.
    for _ in range(count):
        remaining = np.array([i for i in range(probabilities.shape[1]) if i not in selected])
        mass = conditional.mean(axis=0)
        yes = conditional.T @ probabilities[:, remaining] / len(probabilities)
        no = np.maximum(mass[:, None] - yes, 0)
        h = -np.sum(yes * np.log2(np.maximum(yes, 1e-300)) + no * np.log2(np.maximum(no, 1e-300)), axis=0)
        new_conditional_entropy = conditional_entropy + entropy(probabilities[:, remaining]).mean(axis=0)
        joint = np.maximum(h - new_conditional_entropy, 0)
        gains = np.maximum(joint - information, 0)
        utility = expected_pass[remaining] + information_weight * gains
        slot = int(np.argmax(utility))
        index = int(remaining[slot])
        selected.append(index)
        trace.append({'joint_information_bits': float(joint[slot]), 'incremental_information_bits': float(gains[slot]),
                      'expected_pass': float(expected_pass[index]), 'utility_increment': float(utility[slot])})
        p = probabilities[:, index, None]
        conditional = np.concatenate([conditional * (1 - p), conditional * p], axis=1)
        conditional_entropy = float(new_conditional_entropy[slot])
        information = float(joint[slot])
    return selected, trace


def dynamic_policy(state, samples, methods, space, summaries, reservations):
    policy = space.get('acquisition_policy', 'balanced')
    base = {'success': .25, 'balanced': 1., 'exploration': 3.}.get(policy)
    if base is None:
        raise ValueError('Select success, balanced or exploration acquisition policy.')
    novel = (float(state.info['unseen_target']) + len(state.info['unseen_elements']) / max(1, len(state.info['target_elements']))) / 2
    uncertainty = float(np.mean(summaries['bald']))
    recent_brier, n_forecasts = None, 0
    # Use only linked, resolved, provenance-eligible prior forecasts from this composition.
    if methods is not None:
        from .analysis_extended import prediction_records
        scoped = [s for s in samples if s.get('origin') == state.info['origin'] and process_key(s) == tuple(state.info['cohort'][1:])]
        scoped_reservations = [r for r in reservations if r.get('origin') == state.info['origin']]
        frame = prediction_records(scoped, methods, scoped_reservations)
        if not frame.empty:
            frame = frame.loc[frame['eligible'] & (frame['composition'] == composition_name({'elements': state.info['target_elements']}))]
            frame = frame.sort_values('result_reported_at').tail(30)
            if len(frame) >= 5 and frame['batch_id'].nunique() >= 2:
                recent_brier = float(np.mean((frame['label'].astype(float) - frame['p_screen_pass']) ** 2))
                n_forecasts = len(frame)
    weight = min(6., base * (1 + 2 * novel) * (1 + uncertainty) * (1 + (recent_brier or 0)))
    return {'name': policy, 'information_weight': float(weight), 'novelty_score': novel,
            'mean_bald_bits': uncertainty, 'recent_forecast_brier': recent_brier, 'recent_forecast_n': n_forecasts,
            'rule_version': 'dynamic-acquisition-v1', 'note': 'Declared adaptive utility rule, not an optimized or experimentally proven policy.'}


def recommend_adaptive(state, samples, context, space, origin, count, reservations):
    from .element_campaign import enforce_campaign
    enforce_campaign(context, space)
    from .planning import concentration_design, validate_space
    _, _, pulses, max_total = validate_space(space, origin)
    count = max(1, min(12, int(count)))
    if context.get('pulse_limit_C') is None:
        raise ValueError('The adaptive engine requires a controlled cutoff temperature.')
    if space.get('pulse_limit_min_C') is None or space.get('pulse_limit_max_C') is None:
        raise ValueError('Register controlled cutoff-temperature bounds.')
    designs, reference, bounds = concentration_design({**space, 'candidate_design': 'sobol'}, context)
    bridge, bridge_sources = bridge_recipe(state, context, bounds)
    designs.append(bridge['concentrations_mM'])
    designs = list({tuple(c[e] for e in context['elements']): c for c in designs}.values())
    tmin, tmax = space['pulse_limit_min_C'], space['pulse_limit_max_C']
    from .candidate_grid import cutoff_candidates, on_temperature_grid
    temperatures = cutoff_candidates(tmin, tmax, space, count, [context['pulse_limit_C'], bridge['pulse_limit_C']])
    known = {recipe_key(s) for s in samples if s.get('origin') == origin}
    reserved = {recipe_key(r['recipe']) for r in reservations if r.get('state') == 'OPEN' and r.get('origin') == origin}
    common = {k: context[k] for k in ('elements', 'precursor_family', 'support_batch', 'protocol_id')}
    from .preparation import copy_preparation
    copy_preparation(context, common)
    if 'precursors' in context:
        common['precursors'] = context['precursors']
    common.update(origin=origin, temperature_C=None)
    pool = [{**common, 'concentrations_mM': c, 'pulse_count': pulse, 'pulse_limit_C': temperature}
            for c in designs for pulse in pulses for temperature in temperatures
            if sum(c.values()) <= max_total + 1e-9]
    pool = [r for r in pool if recipe_key(r) not in known | reserved]
    candidate_budget = int(space.get('candidate_budget', 2048))
    if not 256 <= candidate_budget <= 4096:
        raise ValueError('Candidate search budget must be between 256 and 4096.')
    full_candidate_count = len(pool)
    if len(pool) > candidate_budget:
        indices = np.sort(np.random.default_rng(int(space.get('acquisition_seed', 42))).choice(len(pool), candidate_budget, replace=False))
        pool = [pool[i] for i in indices]
    baseline = {**common, 'concentrations_mM': reference, 'pulse_count': 1, 'pulse_limit_C': context['pulse_limit_C']}
    repeat = (space.get('baseline_repeat', True) and on_temperature_grid(context['pulse_limit_C'], space) and 1 in pulses and tmin <= context['pulse_limit_C'] <= tmax
              and sum(reference.values()) <= max_total and recipe_key(baseline) in known and recipe_key(baseline) not in reserved
              and all(bounds[e]['min_mM'] <= reference[e] <= bounds[e]['max_mM'] for e in context['elements']))
    seed = int(space.get('acquisition_seed', 42))
    repeat_rationale = None
    repeat_strategy = space.get('repeat_strategy', 'reference')
    if repeat_strategy not in {'reference', 'critical', 'allocation', 'none'}:
        raise ValueError('Select reference, critical, allocation or none repeat strategy.')
    allocated = []
    if repeat_strategy == 'allocation':
        if state.info.get('engine') != 'reliability':
            raise ValueError('Repeat-budget allocation requires the recipe reliability engine.')
        from .repetition_allocation import allocate_repetitions
        allocated = allocate_repetitions(state, context, bounds, pulses, tmin, tmax, max_total, reserved, count, space) if space.get('baseline_repeat', True) else []
        repeat = False
    if repeat_strategy == 'none' or not space.get('baseline_repeat', True):
        repeat = False
    elif repeat_strategy == 'critical':
        selected_repeat, repeat_rationale = critical_repeat(state, context, bounds, pulses, tmin, tmax, max_total, reserved, seed, space)
        repeat = selected_repeat is not None
        if repeat:
            baseline = selected_repeat
    chosen, details = [], []
    policy = {'name': space.get('acquisition_policy', 'balanced'), 'information_weight': None}
    reserved_repeat_slots = int(repeat)+len(allocated)
    if pool and count > reserved_repeat_slots:
        x = np.array([model_feature(r, state.model) for r in pool])
        pieces = [state.model.summary(x[start:start + 512], seed) for start in range(0, len(x), 512)]
        summary = {k: np.concatenate([part[k] for part in pieces]) for k in pieces[0]}
        shortlist = np.unique(np.concatenate([np.argsort(-summary['p'], kind='stable')[:48], np.argsort(-summary['bald'], kind='stable')[:48],
                                               np.argsort(-summary['disagreement'], kind='stable')[:24]]))
        probabilities = state.model.probability_draws(x[shortlist], draws=128, joint=True, seed=seed)
        policy = dynamic_policy(state, samples, getattr(state.model, 'methods', None), space, summary, reservations)
        utility_probability = summary['p'][shortlist].copy()
        robustness_weight = float(space.get('robustness_weight', 0))
        if not 0 <= robustness_weight <= 1:
            raise ValueError('Robustness utility weight must be between 0 and 1.')
        if robustness_weight:
            from .robustness import robustness_summary
            for position, index in enumerate(shortlist):
                check = robustness_summary(state.model, pool[index], {**space, 'concentration_bounds_mM': bounds})
                if check is None or check['valid_scenarios'] < 4:
                    raise ValueError('Robust selection needs user-specified tolerances and at least four valid scenarios per shortlisted recipe. Use interior candidates or disable robust utility until coverage is adequate.')
                utility_probability[position] = (1-robustness_weight)*utility_probability[position] + robustness_weight*check['scenario_min_probability']
        policy['robustness_weight'] = robustness_weight
        picked, trace = greedy_batch(probabilities, utility_probability, count - reserved_repeat_slots, policy['information_weight'])
        for i, acquisition in zip(picked, trace):
            index = int(shortlist[i])
            acquisition.update(expected_pass=float(summary['p'][index]), success_utility_probability=float(utility_probability[i]))
            chosen.append(pool[index])
            details.append({**acquisition, 'p_screen_pass': float(summary['p'][index]), 'p_screen_low': float(summary['low'][index]),
                            'p_screen_high': float(summary['high'][index]), 'bald_bits': float(summary['bald'][index]),
                            'model_disagreement': float(summary['disagreement'][index]),
                            'calibration_state': state.model.calibration_status if summary['calibrated'][index] else 'UNCALIBRATED_TARGET'})
    if repeat and len(chosen) < count:
        result = state.model.summary(np.array([model_feature(baseline, state.model)]), seed)
        chosen.append(baseline)
        details.append({'p_screen_pass': float(result['p'][0]), 'p_screen_low': float(result['low'][0]), 'p_screen_high': float(result['high'][0]),
                        'bald_bits': float(result['bald'][0]), 'calibration_state': state.model.calibration_status if result['calibrated'][0] else 'UNCALIBRATED_TARGET', 'is_repeat': True,
                        'repeat_rationale': repeat_rationale, 'repeat_strategy': repeat_strategy})
    for candidate, rationale in allocated:
        result = state.model.summary(np.array([model_feature(candidate, state.model)]), seed)
        chosen.append(candidate)
        details.append({'p_screen_pass': float(result['p'][0]), 'p_screen_low': float(result['low'][0]), 'p_screen_high': float(result['high'][0]),
            'bald_bits': float(result['bald'][0]), 'calibration_state': state.model.calibration_status,
            'is_repeat': True, 'repeat_strategy': 'allocation', 'repeat_rationale': rationale})
    created = datetime.now(timezone.utc).isoformat()
    batch = 'ADAPT-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:4]
    capture = {'rule_version': RULE_VERSION, 'training_batch_ids': sorted({s['batch_id'] for s in state.training}),
               'training_records': [{'sample_id': s['sample_id'], 'record_uid': s.get('record_uid')} for s in state.training]}
    clean_space = {k: v for k, v in space.items() if k != '_methods'}
    policy.update(full_candidate_count=full_candidate_count, evaluated_candidate_count=len(pool), candidate_budget=candidate_budget,
                  posterior_draws=128, shortlist_budget=120)
    rows = []
    for i, (candidate, detail) in enumerate(zip(chosen, details)):
        affinity = state.model.gp.kernel(np.array([model_feature(candidate, state.model)]), np.array([model_feature(s, state.model, for_training=True) for s in state.training]))[0]
        sources = np.argsort(-affinity, kind='stable')[:3]
        normalized_affinity = affinity / np.sqrt(state.model.gp.kernel.diag(np.array([model_feature(candidate, state.model)]))[0]
                            * state.model.gp.kernel.diag(np.array([model_feature(s, state.model, for_training=True) for s in state.training])))
        extrapolation = input_extrapolation(state.training, candidate)
        purpose = ('Precision-budget repeat (new batch_id required)' if repeat_strategy == 'allocation' else
                   'Critical-condition repeat (new batch_id required)' if repeat_strategy == 'critical' else
                   'Independent baseline repeat (new batch_id required)') if detail.get('is_repeat') else 'Joint information and success acquisition'
        rows.append({'recommendation_id': f'{batch}-{i+1:02d}', 'origin': origin, 'state': 'OPEN', 'recipe': candidate,
                     'alpha': common_alpha(candidate), 'purpose': purpose, 'extrapolation': bool(extrapolation or state.info['unseen_target']),
                     'input_extrapolation': extrapolation, 'chemistry_extrapolation': state.info['unseen_target'],
                     'model_id': state.info['model_id'], 'model_info': state.info, 'space': clean_space, 'created': created,
                     'policy': 'ADAPTIVE_JOINT_INFORMATION', 'acquisition': policy, 'evaluation_capture': capture,
                     'prediction_mode': 'TRANSFER_HYPOTHESIS' if state.info['unseen_target'] else 'TARGET_AND_TRANSFER',
                     'target_labeled_n': state.info['target_labeled_n'], 'target_element_count': len(context['elements']),
                     'highest_source_stage': state.info['highest_source_stage'], 'expansion_step': len(context['elements']) > state.info['highest_source_stage'],
                     'unobserved_pair_count': state.info['unobserved_pair_count'], 'unseen_elements': state.info['unseen_elements'],
                     'bridge_sources': bridge_sources, 'source_sample_ids': [state.training[j]['sample_id'] for j in sources],
                     'source_elements': sorted({e for j in sources for e in state.training[j]['elements']}),
                     'source_compositions': [composition_name(state.training[j]) for j in sources],
                     'support_level': 'TRANSFER_UNVALIDATED' if state.info['unseen_target'] else 'TARGET_OBSERVED',
                     'source_affinity': float(normalized_affinity[sources[0]]), 'interval_note': '90% conditional model posterior interval; not guaranteed or atomic-dispersion confidence', **detail})
        if state.info.get('engine') == 'reliability':
            rows[-1]['policy'] = 'RELIABILITY_INFORMATION_AND_REPEAT_BUDGET'
            if not detail.get('is_repeat'):
                rows[-1]['calibration_state'] = state.model.calibration_status
        from .robustness import robustness_summary
        sensitivity = robustness_summary(state.model, candidate, {**space, 'concentration_bounds_mM': bounds})
        if sensitivity is not None:
            rows[-1]['robustness'] = sensitivity
    engine_name = 'Recipe reliability ensemble' if state.info.get('engine') == 'reliability' else 'Adaptive hierarchical GP'
    note = f'{engine_name}: {len(pool)} eligible candidates; joint posterior sampling and greedy batch information selection. Dynamic policy: {policy["name"]}. Model/transfer reliability requires prospective experiments.'
    if len(rows) < count:
        note += f' Only {len(rows)} proposals fit the registered range and duplicate exclusions.'
    return rows, note
