from __future__ import annotations

import math
import re
import hashlib
import json
from collections import Counter

TARGETS = "Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Sr Y Zr Nb Mo Ru Rh Pd Ag In Sn Sb Ba Hf Ta W Ir Pt Au Pb Bi La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
ELEMENTS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
# Keep the established target ordering stable for historical recipe identities.
ELEMENT_ORDER = TARGETS + [e for e in ELEMENTS if e not in TARGETS]
HIGH_BASE = {"K", "Mg", "Ca", "Sr", "Ba"}
RULE_VERSION = "screen-v1.4-material-outcomes"
STATUS_NAMES = {"PASS": "Candidate screening passed", "ICP_FAIL": "ICP criterion failed", "XRD_FAIL": "XRD criterion failed", "BOTH_FAIL": "Both criteria failed", "BURNED_OUT": "Complete sample burnout", "PENDING": "Evidence pending", "RETEST": "Retest required"}
SAMPLE_CONDITIONS = {"NOT_RECORDED": "Not recorded", "RECOVERED": "Sample recovered", "PARTIAL_LOSS": "Partial material loss", "BURNED_OUT": "Completely burned out", "ACCIDENTAL_LOSS": "Lost during handling", "EQUIPMENT_ABORT": "Run aborted / device problem", "INSUFFICIENT_SAMPLE": "Insufficient sample for testing"}
INVALID_RUN_CONDITIONS = {"ACCIDENTAL_LOSS", "EQUIPMENT_ABORT", "INSUFFICIENT_SAMPLE"}


def base_concentration(element):
    return 0.1 if element in HIGH_BASE else 0.05


def parse_elements(text):
    values = re.split(r"[,;，；\s|]+", str(text).strip())
    values = [x for x in values if x]
    if not values or len(values) != len(set(values)):
        raise ValueError("Target elements must be nonempty and unique.")
    unknown = set(values) - set(ELEMENTS)
    if unknown:
        raise ValueError(f"Unknown element symbols: {', '.join(sorted(unknown))}.")
    return sorted(values, key=ELEMENT_ORDER.index)


def recipe(elements, alpha=1.0):
    return {e: base_concentration(e) * float(alpha) for e in elements}


def finite_number(value, name, positive=False):
    try:
        x = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be numeric.") from None
    if not math.isfinite(x) or (positive and x <= 0):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number.")
    return x


def validate_sample(s):
    s["sample_condition"] = s.get("sample_condition") or "NOT_RECORDED"
    if s["sample_condition"] not in SAMPLE_CONDITIONS:
        raise ValueError("Select a supported sample condition; do not infer burnout from ICP/XRD failure.")
    if s.get("origin") not in {"REAL", "DEMO"}:
        raise ValueError("Origin must be REAL or DEMO.")
    if not str(s.get("sample_id", "")).strip():
        raise ValueError("sample_id cannot be empty.")
    for key in ("batch_id", "precursor_family", "support_batch", "protocol_id"):
        s[key] = str(s.get(key) or "UNKNOWN").strip() or "UNKNOWN"
    elements = parse_elements(" ".join(s.get("elements", [])))
    s["elements"] = elements
    if 'precursors' in s:
        from .precursors import normalize_precursors
        s['precursors'] = normalize_precursors(s['precursors'], elements)
    conc = s.get("concentrations_mM", {})
    if set(conc) != set(elements):
        raise ValueError("Precursor concentrations must match the target element set exactly.")
    s["concentrations_mM"] = {e: None if conc[e] is None or str(conc[e]).strip() == "" else finite_number(conc[e], f"{e} concentration", True) for e in elements}
    from .preparation import MATERIAL_FIELDS, positive
    for key in MATERIAL_FIELDS:
        if key in s:
            s[key] = None if s[key] is None or str(s[key]).strip() == '' else positive(s[key], key)
    if s.get('solvent_volume_mL') is not None:
        s['precursor_amounts_mmol'] = {e: None if c is None else c * s['solvent_volume_mL'] / 1000.
                                     for e, c in s['concentrations_mM'].items()}
    else:
        s.pop('precursor_amounts_mmol', None)
    raw_pulse = s.get("pulse_count")
    pulse = None if raw_pulse is None or str(raw_pulse).strip() == "" else finite_number(raw_pulse, "Pulse count", True)
    if pulse is not None and pulse != int(pulse):
        raise ValueError("Pulse count must be a positive integer.")
    s["pulse_count"] = None if pulse is None else int(pulse)
    for key, name in (("temperature_C", "Measured peak temperature"), ("pulse_limit_C", "Controlled pulse cutoff temperature"), ("voltage_V", "Voltage"), ("current_A", "Current")):
        raw = s.get(key)
        s[key] = None if raw is None or str(raw).strip() == "" else finite_number(raw, name, True)
    review = s.setdefault("icp_review", {"state": "AUTO", "source": "", "reviewer": ""})
    if review.get("state", "AUTO") not in {"AUTO", "PASS", "FAIL", "PENDING"}:
        raise ValueError("ICP judgment must be AUTO, PASS, FAIL or PENDING.")
    stem = s.setdefault("stem", {"state": "NOT_TESTED", "source": "", "reviewer": "", "notes": ""})
    if stem.get("state", "NOT_TESTED") not in {"NOT_TESTED", "CONFIRMED", "NOT_CONFIRMED", "INCONCLUSIVE"}:
        raise ValueError("STEM state must be NOT_TESTED, CONFIRMED, NOT_CONFIRMED or INCONCLUSIVE.")
    xrd = s.get("xrd", {})
    if xrd.get("state", "PENDING") not in {"PASS", "FAIL", "PENDING"}:
        raise ValueError("XRD state must be PASS, FAIL or PENDING.")
    if xrd.get("mode", "AUTO") not in {"AUTO", "MANUAL"}:
        raise ValueError("XRD judgment mode must be AUTO or MANUAL.")
    from .units import normalize_assay
    icp = s["icp"] = [normalize_assay(a) for a in s.get("icp", [])]
    symbols = [r.get("element") for r in icp]
    if len(symbols) != len(set(symbols)):
        raise ValueError("Use one screening result per element per sample. Preserve raw replicate readings and describe the aggregation method.")
    for r in icp:
        if r.get("unit") != "wt%":
            raise ValueError("Standard input accepts W / wt% only. Convert solution concentrations using sample mass, digestion volume and dilution factor first.")
        if r.get("flag", "MEASURED") not in {"MEASURED", "ND", "MISSING", "BLQ"}:
            raise ValueError("ICP flag must be MEASURED, ND, MISSING or BLQ.")
        parse_measurement(r.get("value", ""))
    return s


def parse_measurement(value):
    """Keep censoring intervals; never impute ND as zero."""
    text = str(value).strip()
    if text.upper() in {"", "ND", "N.D.", "NA", "N/A", "NAN", "NOT_PROVIDED"}:
        return {"kind": "missing", "raw": text}
    if text.upper() in {"BLQ", "<LOQ", "BLOQ"}:
        return {"kind": "below_loq", "raw": text}
    if text.startswith(("<", "≤")):
        bound = finite_number(text[1:], "Censoring upper bound", True)
        return {"kind": "censored", "upper": bound, "operator": text[0], "raw": text}
    number = finite_number(text, "ICP value")
    if number < 0:
        raise ValueError("Review negative ICP values before import.")
    return {"kind": "measured", "value": number, "raw": text}


def common_alpha(s):
    if any(s["concentrations_mM"].get(e) is None for e in s["elements"]):
        return None
    ratios = [s["concentrations_mM"][e] / base_concentration(e) for e in s["elements"]]
    if max(ratios) - min(ratios) > 1e-7 * max(1.0, max(ratios)):
        return None
    return sum(ratios) / len(ratios)


def cohort_key(s):
    basis = ("CONTROLLED_CUTOFF",) if s.get("pulse_limit_C") is not None else ("LEGACY_FIXED_TEMPERATURE", round(float(s["temperature_C"]), 3)) if s.get("temperature_C") is not None else ("TEMPERATURE_UNKNOWN",)
    return (tuple(s["elements"]), s["precursor_family"], s["support_batch"], s["protocol_id"], basis)


def recipe_key(s):
    from .precursors import chemistry_identity
    key = (cohort_key(s), tuple((e, None if s["concentrations_mM"].get(e) is None else round(float(s["concentrations_mM"][e]), 9)) for e in s["elements"]), s.get("pulse_count"), s.get("pulse_limit_C"))
    chemistry = chemistry_identity(s)
    key = key + (chemistry,) if chemistry else key
    from .preparation import identity
    materials = identity(s)
    return key + (materials,) if materials else key


def stem_confirmation(s):
    """Optional structural evidence, independent of the ICP/XRD training label."""
    evidence = s.get("stem", {})
    state = evidence.get("state", "NOT_TESTED")
    if state != "NOT_TESTED" and not all(evidence.get(k) for k in ("source", "reviewer")):
        return "INCOMPLETE_EVIDENCE"
    return state


def screen(s, methods):
    """Three-valued evidence logic. Missing evidence cannot become a PASS."""
    results, margins = [], []
    for e in s["elements"]:
        rows = [r for r in s.get("icp", []) if r["element"] == e]
        row = rows[0] if rows else None
        state, reason, margin = "PENDING", "Missing ICP result", None
        if row:
            method = methods.get(row.get("method_id"), {})
            rule = method.get("rule", "LOQ")
            limit = method.get("limits", {}).get(e, {}).get(rule.lower())
            confirmed = method.get("confirmed") and method.get("origin") == s["origin"] and method.get("source") and limit is not None and float(limit) > 0
            val = parse_measurement(row.get("value", ""))
            flag = row.get("flag", "MEASURED")
            reason = "Method limits or criterion source are unconfirmed"
            if confirmed:
                if flag == "MISSING":
                    reason = "Missing result"
                elif flag == "ND" or ((flag == "BLQ" or val["kind"] == "below_loq") and rule == "LOQ"):
                    reported = "BLQ" if val["kind"] == "below_loq" else flag
                    state, reason = "FAIL", f"Reported flag {reported} fails the {rule} criterion"
                elif val["kind"] == "missing" or val["kind"] == "below_loq":
                    reason = "Missing result" if flag != "BLQ" and val["kind"] != "below_loq" else "BLQ alone does not establish detection above LOD"
                elif flag == "BLQ" and rule == "LOD" and val["kind"] != "measured":
                    reason = "BLQ alone does not establish detection above LOD; provide a value or explicit detection evidence"
                elif val["kind"] == "censored":
                    if val["upper"] < float(limit) or (val["upper"] == float(limit) and val["operator"] == "<"):
                        state, reason = "FAIL", f"Censoring upper bound is below {rule}"
                    else:
                        reason = f"Censoring interval overlaps {rule}; unresolved"
                elif val["kind"] == "measured":
                    margin = val["value"] / float(limit)
                    margins.append((e, margin))
                    state = "PASS" if margin >= 1 else "FAIL"
                    reason = f"{rule} margin {margin:.3g}"
        results.append({"element": e, "state": state, "reason": reason, "margin": margin})
    states = [r["state"] for r in results]
    icp_state = "FAIL" if "FAIL" in states else ("PENDING" if "PENDING" in states else "PASS")
    numeric_icp = icp_state
    review = s.get("icp_review", {})
    reported_icp = review.get("state", "AUTO")
    icp_basis = "NUMERIC_CRITERIA" if reported_icp == "AUTO" else "USER_REPORTED"
    conflicts = []
    if reported_icp != "AUTO":
        icp_state = reported_icp
        if reported_icp in {"PASS", "FAIL"} and numeric_icp in {"PASS", "FAIL"} and reported_icp != numeric_icp:
            conflicts.append("Reported ICP judgment conflicts with the numeric criterion result")
    xrd = s.get("xrd", {})
    xrd_state = xrd.get("state", "PENDING")
    xrd_basis = "USER_REPORTED" if xrd.get("mode") == "MANUAL" else "REVIEWED_CRITERIA"
    if xrd_basis != "USER_REPORTED" and xrd_state != "PENDING" and not all(xrd.get(k) for k in ("source", "criteria", "reviewer")):
        xrd_state = "PENDING"
    if icp_state == "FAIL" and xrd_state == "FAIL":
        status = "BOTH_FAIL"
    elif icp_state == "FAIL":
        status = "ICP_FAIL"
    elif xrd_state == "FAIL":
        status = "XRD_FAIL"
    elif icp_state == "PASS" and xrd_state == "PASS":
        status = "PASS"
    else:
        status = "PENDING"
    condition = s.get("sample_condition") or "NOT_RECORDED"
    if condition == "BURNED_OUT":
        status = "BURNED_OUT"
    elif condition in INVALID_RUN_CONDITIONS:
        status = "RETEST"
    weakest = min(margins, key=lambda item: item[1]) if margins else None
    return {"status": status, "sample_condition": condition, "icp": icp_state, "xrd": xrd_state, "stem": stem_confirmation(s), "elements": results,
            "icp_basis": icp_basis, "xrd_basis": xrd_basis, "numeric_icp": numeric_icp, "review_conflicts": conflicts,
            "weakest": weakest, "rule_version": RULE_VERSION,
            "method_versions": {k: v.get("version", "") for k, v in methods.items() if k in {r.get("method_id") for r in s.get("icp", [])}},
            "method_fingerprints": {k: hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16] for k, v in methods.items() if k in {r.get("method_id") for r in s.get("icp", [])}},
            "label": 1 if status == "PASS" else (0 if status in {"ICP_FAIL", "XRD_FAIL", "BOTH_FAIL", "BURNED_OUT"} else None)}


def summary(samples, methods):
    counts = Counter(screen(s, methods)["status"] for s in samples)
    return {"total": len(samples), "counts": dict(counts), "labeled": sum(counts[k] for k in ("PASS", "ICP_FAIL", "XRD_FAIL", "BOTH_FAIL", "BURNED_OUT"))}
