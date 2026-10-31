from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from scipy.spatial.distance import cdist

from .domain import cohort_key, common_alpha, recipe, recipe_key, screen, RULE_VERSION, finite_number, base_concentration


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=list).encode()).hexdigest()[:16]


def feature(s):
    """Actual element concentrations, pulse count and optional controlled cutoff."""
    n = len(s["elements"])
    values = [math.log(finite_number(s["concentrations_mM"].get(e), f"{e} concentration", True) / .05) / math.sqrt(n) for e in s["elements"]]
    values.append((finite_number(s.get("pulse_count"), "Pulse count", True) - 1) / 2.0)
    if s.get("pulse_limit_C") is not None:
        values.append((float(s["pulse_limit_C"]) - 1000.) / 200.)
    return values


def independent_groups(samples):
    """Connect rows sharing either synthesis batch or exact recipe for strict CV."""
    parent = list(range(len(samples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen_batch, seen_recipe = {}, {}
    for i, s in enumerate(samples):
        for key, seen in ((s["batch_id"], seen_batch), (recipe_key(s), seen_recipe)):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    return np.array([find(i) for i in range(len(samples))])


def classifier(dimensions=2, concentrations=1):
    # Fixed kernel first baseline; no hyperparameter tuning on held-out folds.
    return GaussianProcessClassifier(ConstantKernel(1.0, constant_value_bounds="fixed") * RBF([0.65] * concentrations + [1.0] * (dimensions - concentrations), length_scale_bounds="fixed"), optimizer=None, random_state=42, max_iter_predict=100)


@dataclass
class ModelState:
    model: object
    info: dict
    training: list


def fit_model(samples, methods, context, origin="REAL", transfer=False, engine='baseline', settings=None):
    if engine == 'reliability':
        from .reliable_transfer import fit_reliability
        return fit_reliability(samples, methods, context, origin, transfer, settings)
    if engine == 'adaptive':
        from .adaptive import fit_adaptive
        return fit_adaptive(samples, methods, context, origin, transfer, settings)
    if engine != 'baseline':
        raise ValueError('Select baseline or adaptive prediction engine.')
    if transfer:
        from .composition import fit_transfer_model
        return fit_transfer_model(samples, methods, context, origin)
    chosen, exclusions = [], Counter()
    for s in samples:
        if s.get("pulse_count") is None or (s.get("pulse_limit_C") is None and s.get("temperature_C") is None) or any(s["concentrations_mM"].get(e) is None for e in s["elements"]):
            exclusions["Incomplete synthesis conditions"] += 1
            continue
        if s.get("origin") != origin:
            exclusions["Different data namespace"] += 1
            continue
        if cohort_key(s) != cohort_key(context):
            exclusions["Different composition, precursor, support or protocol"] += 1
            continue
        if any(s[k].strip().upper() in {"UNKNOWN", "UNASSIGNED", ""} or s[k].strip().upper().startswith(("FILL_IN_", "REPLACE_")) for k in ("precursor_family", "support_batch", "protocol_id", "batch_id")):
            exclusions["Unresolved process metadata"] += 1
            continue
        if screen(s, methods)["label"] is None:
            exclusions["Incomplete ICP/XRD screening"] += 1
            continue
        if screen(s, methods)["review_conflicts"]:
            exclusions["Conflicting reported and numeric judgments"] += 1
            continue
        chosen.append(s)
    labels = [screen(s, methods)["label"] for s in chosen]
    batches = len(set(s["batch_id"] for s in chosen))
    recipes = len(set(recipe_key(s) for s in chosen))
    count = Counter(labels)
    info = {"mode": "INITIAL_DESIGN", "origin": origin, "n_labeled": len(chosen), "n_batches": batches, "n_recipes": recipes,
            "n_pass": count[1], "n_fail": count[0], "exclusions": dict(exclusions), "validation": None,
            "cohort": list(cohort_key(context)), "model_version": "gpc-fixed-kernel-v3-element-concentrations", "probability_note": "Research baseline probabilities; not prospectively calibrated, for experiment prioritization only."}
    info["features"] = [f"log_concentration_mM_{e}" for e in context["elements"]] + ["pulse_count"] + (["controlled_pulse_limit_C"] if context.get("pulse_limit_C") is not None else [])
    info["concentration_transform"] = "Log of entered mM divided by a fixed 0.05 mM numerical reference, scaled by sqrt(element count). No shared concentration ratio is required."
    info["stem_role"] = "Optional confirmation only; not a training label or predictor"
    info["screening_rule_version"] = RULE_VERSION
    info["reported_label_count"] = sum(screen(s, methods)["icp_basis"] == "USER_REPORTED" or screen(s, methods)["xrd_basis"] == "USER_REPORTED" for s in chosen)
    # Engineering gates, not statistical evidence of sufficient power.
    class_recipes = {label: len({recipe_key(s) for s, y in zip(chosen, labels) if y == label}) for label in (0, 1)}
    min_batches = 8 if origin == "REAL" else 3
    info["minimum_batches"] = min_batches
    if len(chosen) < 12 or batches < min_batches or min(class_recipes.values()) < 3:
        info["reason"] = f"This cohort needs at least 12 valid records, {min_batches} synthesis batches, and 3 distinct recipes per outcome class. These engineering gates do not ensure statistical sufficiency. DEMO uses a lower batch gate to illustrate the workflow."
        return ModelState(None, info, chosen)
    x, y = np.array([feature(s) for s in chosen]), np.array(labels)
    groups = independent_groups(chosen)
    group_count = len(set(groups))
    if group_count >= 3:
        p_cv = np.full(len(y), np.nan)
        for train, test in GroupKFold(n_splits=min(3, group_count)).split(x, y, groups):
            if len(set(y[train])) != 2:
                continue
            m = classifier(x.shape[1], len(context["elements"])).fit(x[train], y[train])
            p_cv[test] = m.predict_proba(x[test])[:, 1]
        valid = np.isfinite(p_cv)
        if valid.all() and len(set(y[valid])) == 2:
            info["validation"] = {"type": "Cross-validation grouped by connected synthesis batches and repeated recipes", "n": int(valid.sum()), "n_groups": group_count,
                                  "balanced_accuracy": float(balanced_accuracy_score(y, p_cv >= .5)), "brier": float(brier_score_loss(y, p_cv)),
                                  "false_pass_count": int(np.sum((p_cv >= .5) & (y == 0)))}
        else:
            info["validation_note"] = "Some training folds contain only one class; full validation metrics are unavailable."
    else:
        info["validation_note"] = "Fewer than 3 independent connected groups; collect independent validation before reporting grouped metrics."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = classifier(x.shape[1], len(context["elements"])).fit(x, y)
    info["mode"] = "GPC_ACTIVE_LEARNING"
    info["reason"] = "Joint ICP/XRD screening classification from " + ", ".join(info["features"]) + ", for the current composition and process cohort."
    info["model_id"] = fingerprint({"samples": chosen, "labels": labels, "methods": methods, "version": info["model_version"], "screening_rule_version": RULE_VERSION})
    return ModelState(model, info, chosen)


def validate_space(space, origin):
    if "concentration_bounds_mM" in space:
        bounds = space["concentration_bounds_mM"]
        if not isinstance(bounds, dict) or not bounds:
            raise ValueError("Enter a concentration range for each selected element, in mM.")
        for e, b in bounds.items():
            if not isinstance(b, dict):
                raise ValueError(f"{e}: enter a lower and upper concentration in mM.")
            lo = finite_number(b.get("min_mM"), f"{e} concentration lower bound", True)
            hi = finite_number(b.get("max_mM"), f"{e} concentration upper bound", True)
            if hi < lo: raise ValueError(f"{e}: lower concentration exceeds upper concentration.")
        low, high = min(float(b["min_mM"]) for b in bounds.values()), max(float(b["max_mM"]) for b in bounds.values())
    else:
        low, high = legacy_concentration_range(space)
    try:
        pulses = list(space['pulses'])
    except (KeyError, TypeError):
        raise ValueError('Enter available pulse counts in recommendation settings.') from None
    if not all(math.isfinite(v) for v in (low, high)) or low <= 0 or high < low:
        raise ValueError("Concentration bounds must be finite and positive, with lower bound no greater than upper bound.")
    if not pulses or any(int(n) != n or n < 1 for n in pulses):
        raise ValueError("Candidate pulse counts must be positive integers.")
    if origin == "REAL" and not space.get("verified"):
        raise ValueError("Register the device-validated operating range in recommendation settings first.")
    max_total = float(space.get("max_total_mM", 1000))
    if not math.isfinite(max_total) or max_total <= 0:
        raise ValueError("The maximum total concentration must be finite and positive.")
    if space.get("pulse_limit_min_C") is not None or space.get("pulse_limit_max_C") is not None:
        try:
            tmin, tmax = float(space["pulse_limit_min_C"]), float(space["pulse_limit_max_C"])
        except (ValueError, TypeError, KeyError):
            raise ValueError("Provide both controlled cutoff-temperature bounds.") from None
        if not all(math.isfinite(t) for t in (tmin, tmax)) or tmin <= 0 or tmax < tmin:
            raise ValueError("Cutoff-temperature bounds must be finite, positive and ordered.")
    return low, high, sorted(set(int(n) for n in pulses)), max_total


def legacy_concentration_range(space):
    """Validate compatibility settings; missing direct ranges need user review."""
    if 'alpha_min' not in space or 'alpha_max' not in space:
        raise ValueError('Enter a lower and upper concentration for each selected element in mM on Next batch.')
    low = finite_number(space['alpha_min'], 'Legacy concentration lower bound', True)
    high = finite_number(space['alpha_max'], 'Legacy concentration upper bound', True)
    if high < low:
        raise ValueError('Legacy concentration lower bound exceeds the upper bound.')
    return low, high


def concentration_bounds(space, elements):
    """Read direct mM ranges; convert old stored multiplier ranges only for compatibility."""
    if "concentration_bounds_mM" in space:
        bounds = space['concentration_bounds_mM']
        if not isinstance(bounds, dict) or set(bounds) != set(elements):
            raise ValueError("Concentration ranges must match the selected element set exactly.")
        if any(not isinstance(b, dict) for b in bounds.values()):
            raise ValueError('Enter a lower and upper concentration in mM for each element.')
        return {e: {k: finite_number(space["concentration_bounds_mM"][e].get(k), f"{e} {k}", True) for k in ("min_mM", "max_mM")} for e in elements}
    low, high = legacy_concentration_range(space)
    return {e: {'min_mM': base_concentration(e) * low, 'max_mM': base_concentration(e) * high} for e in elements}


def concentration_design(space, context):
    """Bounded deterministic candidates in actual mM, including registered references."""
    elements = context["elements"]
    bounds = concentration_bounds(space, elements)
    reference = {e: context["concentrations_mM"].get(e) or base_concentration(e) for e in elements}
    if "concentration_bounds_mM" not in space:
        values = np.geomspace(space["alpha_min"], space["alpha_max"], 21)
        designs = [recipe(elements, v) for v in values]
    else:
        lo = np.array([bounds[e]["min_mM"] for e in elements]); hi = np.array([bounds[e]["max_mM"] for e in elements])
        varying = np.flatnonzero(hi > lo)
        if len(varying) <= 1:
            fractions = np.linspace(0, 1, 21)[:, None] if len(varying) else np.zeros((1, 1))
        else:
            # Stratified finite design avoids an exponential Cartesian grid.
            if space.get('candidate_design') == 'sobol':
                from scipy.stats import qmc
                fractions = qmc.Sobol(d=len(varying), scramble=True, seed=42).random_base2(m=8)
            else:
                rng = np.random.default_rng(42)
                fractions = np.column_stack([(rng.permutation(128) + rng.random(128)) / 128 for _ in varying])
            fractions = np.vstack([fractions, np.zeros(len(varying)), np.ones(len(varying))])
        designs = []
        for row in fractions:
            concentrations = lo.copy()
            for j, index in enumerate(varying): concentrations[index] = np.clip(np.exp(np.log(lo[index]) + row[j] * np.log(hi[index] / lo[index])), lo[index], hi[index])
            designs.append(dict(zip(elements, map(float, concentrations))))
    if all(bounds[e]["min_mM"] <= reference[e] <= bounds[e]["max_mM"] for e in elements): designs.append(reference)
    unique = {tuple(round(c[e], 12) for e in elements): c for c in designs}
    return list(unique.values()), reference, bounds


def recommend(state, samples, context, space, origin, count=6, reservations=(),sac_gen=None):
    from .element_campaign import enforce_campaign
    enforce_campaign(context, space)
    if sac_gen is None and state.info.get('engine') in {'adaptive', 'reliability'} and state.model is not None:
        from .adaptive_acquisition import recommend_adaptive
        return recommend_adaptive(state, samples, context, space, origin, count, reservations)
    if context.get("pulse_limit_C") is None and context.get("temperature_C") is None:
        raise ValueError("Saved partial records are allowed. Configure a temperature and operating range on Next batch before proposing recipes.")
    low, high, pulses, max_total = validate_space(space, origin)
    count = max(1, min(12, int(count)))
    designs, reference, bounds = concentration_design(space, context)
    transferring = state.info.get("feature_kind") == "COMPOSITION_TRANSFER"
    if transferring:
        from .composition import transfer_feature, source_evidence, descriptors, bridge_recipe, input_extrapolation
        encode = transfer_feature
        bridge, bridge_sources = bridge_recipe(state, {**context, "concentrations_mM": reference, "pulse_count": context.get("pulse_count") or 1}, bounds)
        designs.append(bridge["concentrations_mM"])
        for source in state.training:
            projected = {}
            for element in context["elements"]:
                symbol = element if element in source["elements"] else min(source["elements"], key=lambda e: np.sum((np.array(descriptors(e)) - descriptors(element)) ** 2))
                value = source["concentrations_mM"][symbol]
                projected[element] = min(bounds[element]["max_mM"], max(bounds[element]["min_mM"], value))
            designs.append(projected)
        designs = list({tuple(c[e] for e in context["elements"]): c for c in designs}.values())
    else:
        encode = feature
    candidates = []
    existing = {recipe_key(s) for s in samples if s["origin"] == origin}
    reserved = {recipe_key(r["recipe"]) for r in reservations if r.get("state") == "OPEN" and r.get("origin") == origin}
    synthesis_context = {k: context[k] for k in ("elements", "temperature_C", "precursor_family", "support_batch", "protocol_id")}
    from .preparation import copy_preparation
    copy_preparation(context, synthesis_context)
    if 'precursors' in context:
        synthesis_context['precursors'] = context['precursors']
    synthesis_context["origin"] = origin
    temperatures = [None]
    if context.get("pulse_limit_C") is not None:
        if space.get("pulse_limit_min_C") is None or space.get("pulse_limit_max_C") is None:
            raise ValueError("Register validated cutoff-temperature bounds for the controlled-temperature model.")
        tmin, tmax = float(space["pulse_limit_min_C"]), float(space["pulse_limit_max_C"])
        from .candidate_grid import cutoff_candidates
        anchors = [context['pulse_limit_C']]
        if transferring:
            anchors += [s['pulse_limit_C'] for s in state.training if tmin <= s['pulse_limit_C'] <= tmax] + [bridge['pulse_limit_C']]
        temperatures = cutoff_candidates(tmin, tmax, space, count, anchors)
        synthesis_context["temperature_C"] = None  # Actual temperature is measured after synthesis.
    for concentrations in designs:
        for pulse in pulses:
            for temperature in temperatures:
                r = {**synthesis_context, "concentrations_mM": concentrations, "pulse_count": pulse, "pulse_limit_C": temperature}
                if sum(r["concentrations_mM"].values()) <= max_total + 1e-9 and recipe_key(r) not in existing | reserved:
                    candidates.append(r)
    selected = []
    if candidates:
        x = np.array([encode(r) for r in candidates])
        # Chunk prediction to bound kernel memory for many-element designs.
        if sac_gen is not None:
            sac_forecasts=sac_gen.predict(candidates)
            p=np.asarray([v['p_xrd_pass'] if v['p_xrd_pass'] is not None else np.nan for v in sac_forecasts])
        else:
            p = np.concatenate([state.model.predict_proba(x[start:start + 512])[:, 1] for start in range(0, len(x), 512)]) if state.model is not None else None
        references = np.array([encode(s) for s in samples if s["origin"] == origin and cohort_key(s) == cohort_key(context) and s.get("pulse_count") is not None and all(s["concentrations_mM"].get(e) is not None for e in s["elements"])])
        bridge_anchor = np.array(encode(bridge)) if transferring and state.model is None and bridge_sources else None
        for slot in range(min(count, len(candidates))):
            remaining = [i for i in range(len(candidates)) if i not in selected]
            anchors = [x[i] for i in selected] + list(references)
            if anchors:
                distance = cdist(x[remaining], np.array(anchors), metric="sqeuclidean").min(axis=1)
            else:
                # Seed closest to the established baseline, then cover space.
                anchor = bridge_anchor if bridge_anchor is not None else np.array(encode({**context, "concentrations_mM": reference, "pulse_count": 1}))
                distance = -np.sum((x[remaining] - anchor) ** 2, axis=1) if slot == 0 else np.ones(len(remaining))
            if p is None:
                score = distance
            else:
                diversity = np.maximum(distance, 0) / (1 + np.maximum(distance, 0))
                pp = np.clip(p[remaining], 1e-9, 1 - 1e-9)
                entropy = -(pp * np.log(pp) + (1 - pp) * np.log(1 - pp)) / math.log(2)
                score = (.8 * entropy + .2 * diversity) if slot % 5 < 3 else (.8 * pp + .2 * diversity)
                if sac_gen is not None:
                    objective=space.get('sac_gen_strategy','Balanced boundary learning')
                    if objective=='High-loading XRD failure mapping':
                        loading=np.asarray([sac_forecasts[i]['predicted_total_wt_percent'] or 0. for i in remaining])
                        loading=loading/max(float(loading.max()),1e-9)
                        score=.65*loading*(1-pp)+.2*entropy+.15*diversity
                    elif objective=='XRD pass region validation':score=.8*pp+.2*diversity
                score=np.where(np.isfinite(score),score,diversity)
            selected.append(remaining[int(np.argmax(score))])
    else:
        p = None
    rows = []
    batch = datetime.now(timezone.utc).strftime("PLAN-%Y%m%d-%H%M%S-") + uuid4().hex[:4]
    for slot, idx in enumerate(selected):
        r = candidates[idx]
        predicted = float(p[idx]) if p is not None and np.isfinite(p[idx]) else None
        # Outside observed min/max is marked extrapolation even within device bounds.
        extrapolated = False
        if state.training:
            if transferring:
                extrapolated = input_extrapolation(state.training, r)
            else:
                xt = np.array([encode(s) for s in state.training])
                observed = np.array(encode(r))
                extrapolated = bool(np.any(observed < xt.min(axis=0)) or np.any(observed > xt.max(axis=0)))
        purpose = "Initial space coverage" if p is None else ("Boundary exploration" if slot % 5 < 3 else "Predicted feasible-region validation")
        if transferring and state.training:
            purpose = "Source-guided bridge experiment" if p is None and slot == 0 else "Source-guided space coverage" if p is None else "Transfer boundary exploration" if slot % 5 < 3 else "Transfer hypothesis validation"
        rows.append({"recommendation_id": f"{batch}-{slot + 1:02d}", "origin": origin, "state": "OPEN", "recipe": r,
                     "alpha": common_alpha(r), "p_screen_pass": predicted, "purpose": purpose,
                     "extrapolation": extrapolated, "model_id": state.info.get("model_id", "NO_MODEL"),
                     "model_info": state.info, "space": space, "created": datetime.now(timezone.utc).isoformat()})
        if sac_gen is not None:rows[-1]['sac_gen_prediction']=sac_forecasts[idx]
        if sac_gen is not None:rows[-1]['purpose']=space.get('sac_gen_strategy','Balanced boundary learning')
        if transferring:
            rows[-1].update(**source_evidence(state, r), target_labeled_n=state.info["target_labeled_n"],
                            chemistry_extrapolation=state.info["unseen_target"], input_extrapolation=extrapolated,
                            target_element_count=len(context["elements"]), highest_source_stage=state.info["highest_source_stage"],
                            expansion_step=len(context["elements"]) > state.info["highest_source_stage"],
                            unobserved_pair_count=state.info["unobserved_pair_count"], unseen_elements=state.info["unseen_elements"], bridge_sources=bridge_sources,
                            extrapolation=extrapolated or state.info["unseen_target"],
                            prediction_mode="TRANSFER_HYPOTHESIS" if state.info["unseen_target"] else "TARGET_AND_TRANSFER",
                            predictive_entropy=None if predicted is None else float(-(max(predicted, 1e-12) * math.log2(max(predicted, 1e-12)) + max(1 - predicted, 1e-12) * math.log2(max(1 - predicted, 1e-12)))))
    # Explicit baseline repeat, at most one, never disguised as a novel recipe.
    baseline = {**synthesis_context, "concentrations_mM": reference, "pulse_count": 1, "pulse_limit_C": context.get("pulse_limit_C")}
    if len(rows) < count and space.get("baseline_repeat", True) and baseline.get("pulse_limit_C") in temperatures and all(bounds[e]["min_mM"] <= reference[e] <= bounds[e]["max_mM"] for e in context["elements"]) and 1 in pulses and sum(baseline["concentrations_mM"].values()) <= max_total and recipe_key(baseline) in existing and recipe_key(baseline) not in reserved:
        rows.append({"recommendation_id": f"{batch}-R", "origin": origin, "state": "OPEN", "recipe": baseline, "alpha": common_alpha(baseline),
                     "p_screen_pass": None, "purpose": "Independent baseline repeat (new batch_id required)", "extrapolation": False,
                     "model_id": state.info.get("model_id", "NO_MODEL"), "model_info": state.info, "space": space,
                     "created": datetime.now(timezone.utc).isoformat()})
    note = "These are proposed experiments; predicted values are not measurements."
    if len(rows) < count:
        note += f" The current range supports only {len(rows)} proposals. Add a validated range or close completed reservations."
    if state.model is None:
        note += " Insufficient data: initial experimental design, with no invented pass probabilities."
    else:
        note += " Classification active learning combines uncertainty exploration, feasibility validation and batch diversity. The objective is screening success."
    if transferring:
        note += " Composition transfer uses the same precursor/support/protocol, individual and total concentrations, member sharing and unmeasured pair/composition residuals. Single-element success is not a mixture label. Source affinity and predictive entropy are not calibrated confidence. Validate bridge experiments on the target element."
    capture = {"rule_version": RULE_VERSION,
               "training_batch_ids": sorted({s["batch_id"] for s in state.training}),
               "training_records": [{"sample_id": s["sample_id"], "record_uid": s.get("record_uid")} for s in state.training]}
    for row in rows:
        row["evaluation_capture"] = capture
        row["policy"] = "CLASSIFICATION_ACTIVE_LEARNING" if state.model is not None else "SOURCE_GUIDED_INITIAL_DESIGN"
        if sac_gen is not None:
            predicted=row.get('sac_gen_prediction') or sac_gen.predict([row['recipe']])[0]
            row.update(sac_gen_prediction=predicted,p_xrd_pass=predicted['p_xrd_pass'],p_xrd_fail=predicted['p_xrd_fail'],
                xrd_pass_threshold=predicted.get('xrd_pass_threshold',.5),predicted_xrd_state=predicted.get('predicted_xrd_state'),
                predicted_total_wt_percent=predicted['predicted_total_wt_percent'],p_screen_pass=None,
                model_id=sac_gen.model_id,model_info={'engine':'sac_gen','name':sac_gen.variant_name,'loading_family':sac_gen.family},
                policy='SAC_GEN_XRD_ACTIVE_LEARNING',prediction_target='XRD_PASS',
                evaluation_capture={'rule_version':RULE_VERSION,'training_batch_ids':sorted({s['batch_id'] for s in sac_gen.training_samples}),
                    'training_records':[{'sample_id':s['sample_id'],'record_uid':s.get('record_uid')} for s in sac_gen.training_samples]})
    if sac_gen is not None:
        note=sac_gen.variant_name+' predicts continuous ICP loading and trained XRD/STEM structural-assessment probability. Acquisition alternates XRD boundary exploration, predicted pass regions and recipe diversity. High-loading XRD FAIL experiments remain valid learning observations. No ICP pass probability or material-quality score is used. Probabilities require empirical calibration; unsupported conditions remain flagged.'
    return rows, note


def recommendation_table(rows):
    from .mixing import forecast_metadata
    return [{"recommendation_id": r["recommendation_id"], "target_elements": ";".join(r["recipe"]["elements"]),
             **forecast_metadata(r['recipe'], r.get('model_info'), r.get('p_screen_pass')),
             "concentration_mM": "; ".join(f"{e}: {v:.6g}" for e, v in r["recipe"]["concentrations_mM"].items()), "pulse_count": r["recipe"]["pulse_count"], "temperature_C": r["recipe"]["temperature_C"],
             "pulse_limit_C": r["recipe"].get("pulse_limit_C"),
             "total_concentration_mM": sum(r["recipe"]["concentrations_mM"].values()),
             "concentrations_mM_json": json.dumps(r["recipe"]["concentrations_mM"], ensure_ascii=False),
             'precursor_compounds': '; '.join(f"{e}: {r['recipe'].get('precursors', {}).get(e, {}).get('compound') or 'UNKNOWN'}" for e in r['recipe']['elements']),
             'precursors_json': json.dumps(r['recipe'].get('precursors', {}), ensure_ascii=False),
             "precursor_family": r["recipe"]["precursor_family"], "support_batch": r["recipe"]["support_batch"], "protocol_id": r["recipe"]["protocol_id"],
             "p_screen_pass": r["p_screen_pass"], "purpose": r["purpose"], "extrapolation": r["extrapolation"],
             'p_xrd_pass':r.get('p_xrd_pass'),'p_xrd_fail':r.get('p_xrd_fail'),
             'xrd_pass_threshold':r.get('xrd_pass_threshold'),'predicted_xrd_state':r.get('predicted_xrd_state'),
             'predicted_total_wt_percent':r.get('predicted_total_wt_percent'),
             'p_screen_low': r.get('p_screen_low'), 'p_screen_high': r.get('p_screen_high'),
             'bald_bits': r.get('bald_bits'), 'model_disagreement': r.get('model_disagreement'),
             'calibration_state': r.get('calibration_state'), 'joint_information_bits': r.get('joint_information_bits'),
             'information_weight': (r.get('acquisition') or {}).get('information_weight'),
             'is_repeat': r.get('is_repeat', False), 'repeat_strategy': r.get('repeat_strategy'),
             'planned_additional_repeats': (r.get('repeat_rationale') or {}).get('planned_additional_repeats'),
             'further_repeats_after_this': (r.get('repeat_rationale') or {}).get('further_repeats_after_this'),
             "model_id": r["model_id"], "origin": r["origin"],
             "prediction_mode": r.get("prediction_mode", "TARGET_ONLY"), "target_labeled_n": r.get("target_labeled_n"),
             "source_elements": ";".join(r.get("source_elements", [])), "source_sample_ids": ";".join(r.get("source_sample_ids", [])),
             "source_affinity": r.get("source_affinity"), "predictive_entropy": r.get("predictive_entropy"),
             "chemistry_extrapolation": r.get("chemistry_extrapolation", False),
             "source_compositions": " | ".join(r.get("source_compositions", [])), "support_level": r.get("support_level", "TARGET_ONLY"),
             "target_element_count": len(r["recipe"]["elements"]), "highest_source_stage": r.get("highest_source_stage"),
             "expansion_step": r.get("expansion_step", False), "unobserved_pair_count": r.get("unobserved_pair_count"),
             "unseen_elements": ";".join(r.get("unseen_elements", []))} for r in rows]
