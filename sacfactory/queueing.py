"""Link manually recorded or imported results to reserved and pending proposals."""
import copy

from .domain import parse_measurement, recipe_key, screen, RULE_VERSION


def controlled_recipe_key(sample):
    """Compare entered synthesis settings, not an unmeasured peak observation."""
    from .precursors import chemistry_identity
    key = (tuple(sample["elements"]), tuple(sorted((e, None if c is None else round(float(c), 9)) for e, c in sample["concentrations_mM"].items())), sample["pulse_count"], sample.get("pulse_limit_C"),
            sample["precursor_family"], sample["support_batch"], sample["protocol_id"])
    chemistry = chemistry_identity(sample)
    key = key + (chemistry,) if chemistry else key
    from .preparation import identity
    materials = identity(sample)
    return key + (materials,) if materials else key


def prepare_reports(samples, store, methods):
    revised, updates = [], []
    for supplied in samples:
        sample = copy.deepcopy(supplied)
        old = store.get("sample", sample["sample_id"]) or {}
        for key in ("recommendation_id", "proposal", "manual_prediction", "manual_prediction_history", "planned_temperature_C", "reported_recipe_deviation", "result_reported_at"):
            if key in old and (key not in sample or sample[key] in ("", None)):
                sample[key] = copy.deepcopy(old[key])
        ident = sample.get("recommendation_id") or sample["sample_id"]
        reserved = store.get("recommendation", ident)
        if reserved and not sample.get("recommendation_id") and reserved.get("deletion_note"):
            reserved = None
        if reserved and reserved["state"] != "CANCELLED":
            mapped = reserved.get("pending_sample_id")
            if mapped and mapped != sample["sample_id"]:
                raise ValueError("A proposal is already linked to a different pending sample.")
            from .manual import actual_batch_recorded
            actual_batch = actual_batch_recorded(sample)
            has_results = any(a.get("flag") in {"ND", "BLQ"} or parse_measurement(a.get("value", ""))["kind"] != "missing" for a in sample.get("icp", []) if a.get("flag") != "MISSING")
            has_results = has_results or sample.get("sample_condition") == "BURNED_OUT"
            has_results = has_results or sample.get("icp_review", {}).get("state") in {"PASS", "FAIL"}
            has_results = has_results or (sample.get("xrd", {}).get("mode") == "MANUAL" and sample["xrd"].get("state") in {"PASS", "FAIL"})
            result = screen(sample, methods)
            compare = controlled_recipe_key if mapped else recipe_key
            audited_correction = old.get("reported_recipe_deviation") and controlled_recipe_key(old) == controlled_recipe_key(sample)
            if (mapped or (actual_batch and has_results)) and compare(reserved["recipe"]) != compare(sample) and not audited_correction:
                raise ValueError(f"Reported recipe {sample['sample_id']} differs from its reserved proposal. Use a new sample ID for an altered recipe.")
            completed = actual_batch and has_results and (not mapped or (result["label"] is not None and not result["review_conflicts"]))
            if completed or mapped:
                updated = copy.deepcopy(reserved)
                updated.update(state="COMPLETED" if completed else "OPEN", reported_sample_id=sample["sample_id"],
                               screening_status=result["status"], screening_rule_version=RULE_VERSION)
                if updated != reserved:
                    updates.append(updated)
        from .manual import actual_batch_recorded
        from .storage import now
        result = screen(sample, methods)
        if actual_batch_recorded(sample) and result['label'] is not None and not result['review_conflicts']:
            sample['result_reported_at'] = old.get('result_reported_at') or now()
        revised.append(sample)
    return revised, updates
