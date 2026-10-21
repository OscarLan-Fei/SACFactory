"""Explicit human revisions and atomic result feedback for recorded samples."""
import copy
import json

from .domain import parse_measurement, screen, validate_sample, RULE_VERSION
from .planning import fingerprint
from .queueing import controlled_recipe_key
from .storage import now
from .precursors import to_columns as precursor_to_columns, from_row as precursor_from_row, COLUMNS as PRECURSOR_COLUMNS


def has_observations(sample):
    return ((sample.get("sample_condition") or "NOT_RECORDED") != "NOT_RECORDED"
        or sample.get('temperature_C') is not None and sample.get('temperature_C') != ''
        or sample.get('electrical_input_kind') != 'SETTING' and any(sample.get(k) is not None and sample.get(k) != '' for k in ('voltage_V', 'current_A'))
        or sample.get("icp_review", {}).get("state") in {"PASS", "FAIL"}
        or sample.get("xrd", {}).get("state") in {"PASS", "FAIL"}
        or sample.get("stem", {}).get("state", "NOT_TESTED") != "NOT_TESTED"
        or any(a.get("flag") in {"ND", "BLQ"} or parse_measurement(a.get("value", ""))["kind"] != "missing" for a in sample.get("icp", [])))


def actual_batch_recorded(sample):
    batch = sample.get("batch_id", "UNKNOWN").strip().upper()
    return bool(batch and batch not in {"UNKNOWN", "UNASSIGNED"} and not batch.startswith(("REPLACE_", "FILL_IN_")))


def pending_samples(samples, recommendations, methods):
    linked = {r.get("pending_sample_id") or r.get("reported_sample_id"): r for r in recommendations if not (r.get("state") == "CANCELLED" and r.get("deletion_note"))}
    return [s for s in samples if (linked.get(s["sample_id"], {}).get("state") == "OPEN"
        or not linked.get(s["sample_id"]) and screen(s, methods)["label"] is None)]


def element_rows(sample, elements=None):
    """Editable decimal-percent content, separate from precursor concentration."""
    assays = {a["element"]: a for a in sample.get("icp", [])}
    return [{"element": e, "precursor_mM": sample["concentrations_mM"].get(e),
        "value": str(assays.get(e, {}).get("value", "")), "unit": "wt%",
        "flag": assays.get(e, {}).get("flag", ""), "method_id": assays.get(e, {}).get("method_id", ""),
        "source": assays.get(e, {}).get("source", ""), **precursor_to_columns(sample, e)} for e in (elements if elements is not None else sample["elements"])]


def apply_element_rows(sample, rows, preserve_assays=None):
    updated = copy.deepcopy(sample)
    concentrations, assays = {}, []
    for row in rows:
        e = row["element"]
        if e in concentrations:
            raise ValueError("Duplicate element in the manual element table.")
        concentrations[e] = row.get("precursor_mM")
        raw = row.get("value")
        text = "" if raw is None else str(raw).strip()
        flag = str(row.get("flag") or "").strip()
        if text.upper() in {"ND", "N.D."}:
            flag = "ND"
        elif text.upper() in {"BLQ", "<LOQ", "BLOQ"}:
            flag = "BLQ"
        elif text and flag in {"", "MISSING"}:
            flag = "MEASURED"
        elif not text and flag not in {"ND", "BLQ"}:
            flag = "MISSING"
        assays.append({"element": e, "value": text, "unit": "wt%", "flag": flag,
                       "method_id": str(row.get("method_id") or ""), "source": str(row.get("source") or "")})
    if set(concentrations) != set(updated["elements"]):
        raise ValueError("The element table must match the selected target elements.")
    extras = preserve_assays if preserve_assays is not None else [a for a in sample.get("icp", []) if a["element"] not in sample["elements"]]
    assays.extend(copy.deepcopy(a) for a in extras if a["element"] not in concentrations)
    updated.update(concentrations_mM=concentrations, icp=assays)
    if any(any(k in row for k in PRECURSOR_COLUMNS) for row in rows):
        updated['precursors'] = {row['element']: precursor_from_row(row) for row in rows}
    return updated


def save_manual_revisions(store, supplied, methods, mode, expected):
    """Update existing identities and linked tasks together, with optimistic locking."""
    if mode not in {"records", "pending", "feedback"}:
        raise ValueError("Unknown manual editing mode.")
    ids = [s.get("sample_id") for s in supplied]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Select existing, unique Sample IDs.")
    result = {"samples": 0, "completed_tasks": 0, "open_tasks": 0, "statuses": {}}
    with store.connect() as c:
        c.execute("BEGIN IMMEDIATE")
        writes = []
        for supplied_sample in supplied:
            ident = supplied_sample["sample_id"]
            encoded = c.execute("SELECT payload FROM items WHERE kind='sample' AND id=?", (ident,)).fetchone()
            if not encoded:
                raise ValueError(f"Sample {ident} no longer exists. Refresh the editor.")
            old = json.loads(encoded[0])
            if expected.get(ident) != fingerprint(old):
                raise ValueError(f"Sample {ident} changed while editing. Reload its latest data.")
            if supplied_sample.get("origin") != store.origin:
                raise ValueError("REAL and DEMO records cannot be mixed.")
            sample = copy.deepcopy(old)
            for key in ("process_metadata_basis", "electrical_input_kind", "pan_mass_g", "solvent_volume_mL", "preparation_basis", "concentration_input_unit", "batch_id", "elements", "concentrations_mM", "pulse_count", "pulse_limit_C", "temperature_C", "voltage_V", "current_A", "precursor_family", "precursors", "support_batch", "protocol_id", "notes", "sample_condition", "icp", "icp_review", "xrd", "stem"):
                if key in supplied_sample:
                    sample[key] = copy.deepcopy(supplied_sample[key])
            sample = validate_sample(sample)
            recipe_changed = controlled_recipe_key(sample) != controlled_recipe_key(old)
            if mode == "pending":
                if has_observations(sample) != has_observations(old) or any(sample.get(k, "NOT_RECORDED") != old.get(k, "NOT_RECORDED") for k in ("sample_condition", "icp", "icp_review", "xrd", "stem", "temperature_C", "voltage_V", "current_A")):
                    raise ValueError("Enter observations under Result feedback; Pending edits update planned conditions only.")
                if recipe_changed and has_observations(old):
                    raise ValueError("This sample has already been measured. Correct its conditions under Experiment records or Result feedback.")
            link = old.get("recommendation_id") or ident
            queue_row = c.execute("SELECT payload FROM items WHERE kind='recommendation' AND id=?", (link,)).fetchone()
            queue = json.loads(queue_row[0]) if queue_row else None
            if queue and not old.get("recommendation_id") and queue.get("deletion_note"):
                queue = None
            if queue and queue.get("pending_sample_id") not in {None, ident}:
                raise ValueError("This task is linked to another sample.")
            if mode == "pending" and queue and queue["state"] != "OPEN":
                raise ValueError("Only open tasks can be edited as Pending experiments.")
            if mode == "pending" and not queue and screen(old, methods)["label"] is not None:
                raise ValueError("This record is resolved. Edit it under Experiment records.")
            if recipe_changed and sample.get("proposal"):
                sample["proposal"]["prediction_stale"] = True
            if recipe_changed and sample.get('manual_prediction'):
                sample['manual_prediction']['prediction_stale'] = True
            if queue and queue["state"] != "CANCELLED":
                amended = copy.deepcopy(queue)
                if recipe_changed:
                    amended.setdefault("original_proposal", copy.deepcopy(queue))
                    amended.update(prediction_stale=True, p_screen_pass=None,p_xrd_pass=None,p_xrd_fail=None, predictive_entropy=None)
                    if amended.get('manual_prediction'):
                        amended['manual_prediction']['prediction_stale'] = True
                if mode == "pending" and recipe_changed:
                    amended.setdefault("original_proposal", copy.deepcopy(queue))
                    recipe = copy.deepcopy(queue["recipe"])
                    for key in ("pan_mass_g", "solvent_volume_mL", "elements", "concentrations_mM", "pulse_count", "pulse_limit_C", "precursor_family", "support_batch", "protocol_id"):
                        recipe[key] = copy.deepcopy(sample.get(key))
                    if 'precursors' in sample:
                        recipe['precursors'] = copy.deepcopy(sample['precursors'])
                    else:
                        recipe.pop('precursors', None)
                    amended.update(recipe=recipe, prediction_stale=True, p_screen_pass=None,p_xrd_pass=None,p_xrd_fail=None, predictive_entropy=None,
                                   purpose="Manually edited pending recipe; original prediction is historical.")
                    amended.setdefault("plan_amendments", []).append({"created": now(), "before_recipe": queue["recipe"], "after_recipe": recipe})
                elif controlled_recipe_key(sample) != controlled_recipe_key(queue["recipe"]):
                    sample["reported_recipe_deviation"] = {"basis": "Explicit manual correction of actual conditions", "planned_recipe": copy.deepcopy(queue["recipe"]),
                        "actual_conditions": {k: copy.deepcopy(sample.get(k)) for k in ("pan_mass_g", "solvent_volume_mL", "elements", "concentrations_mM", "pulse_count", "pulse_limit_C", "precursor_family", "precursors", "support_batch", "protocol_id")}}
                    amended["reported_recipe_deviation"] = copy.deepcopy(sample["reported_recipe_deviation"])
                    amended["prediction_stale"] = True
                else:
                    sample.pop("reported_recipe_deviation", None)
                    amended.pop("reported_recipe_deviation", None)
                screening = screen(sample, methods)
                completed = actual_batch_recorded(sample) and screening["label"] is not None and not screening["review_conflicts"]
                amended.update(state="COMPLETED" if completed else "OPEN", reported_sample_id=ident,
                    screening_status=screening["status"], screening_rule_version=RULE_VERSION)
                if amended != queue:
                    writes.append(("recommendation", link, queue_row[0], amended))
                result["completed_tasks" if completed else "open_tasks"] += 1
            result["statuses"][ident] = screen(sample, methods)["status"]
            if sample != old:
                resolved = screen(sample, methods)
                if actual_batch_recorded(sample) and resolved["label"] is not None and not resolved["review_conflicts"]:
                    sample["result_reported_at"] = old.get("result_reported_at") or now()
                writes.append(("sample", ident, encoded[0], sample))
                result["samples"] += 1
        for kind, ident, before, obj in writes:
            encoded = json.dumps(obj, ensure_ascii=False, allow_nan=False)
            c.execute("UPDATE items SET payload=? WHERE kind=? AND id=?", (encoded, kind, ident))
            c.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,?,?,?,?,?)",
                      (now(), "manual_" + mode, kind, ident, before, encoded))
    return result
