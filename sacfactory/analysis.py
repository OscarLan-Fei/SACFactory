"""Descriptive mathematics on recorded conditions and existing screening labels.

No assay interpretation, spectrum processing, imputation or model fitting occurs here.
"""
from math import isfinite, sqrt

import pandas as pd

from .domain import cohort_key, recipe_key


PARAMETERS = {
    "pulse_count": "Pulse count",
    "pulse_limit_C": "Controlled cutoff / C",
    "temperature_C": "Measured peak / C",
    "voltage_V": "Voltage / V",
    "current_A": "Current / A",
}
FAILURES = {"ICP_FAIL", "XRD_FAIL", "BOTH_FAIL", "BURNED_OUT"}


def numeric(value):
    """Keep missing or nonfinite values missing; preserve legitimate zeros."""
    try:
        number = float(value)
        return number if isfinite(number) else None
    except (TypeError, ValueError):
        return None


def experiment_frame(samples, outcomes):
    """Consume existing labels keyed by sample ID, without reading raw assays."""
    rows = []
    for sample in samples:
        result = outcomes.get(sample["sample_id"], {})
        status = result.get("status", "PENDING")
        conflict = bool(result.get("review_conflicts"))
        label = None if conflict else 1 if status == "PASS" else 0 if status in FAILURES else None
        concentrations = sample.get("concentrations_mM", {})
        complete = all(numeric(concentrations.get(e)) is not None and numeric(concentrations[e]) > 0 for e in sample["elements"])
        complete = complete and numeric(sample.get("pulse_count")) is not None and numeric(sample.get("pulse_count")) > 0
        complete = complete and (numeric(sample.get("pulse_limit_C")) is not None or numeric(sample.get("temperature_C")) is not None)
        row = {
            "sample_id": sample["sample_id"], "batch_id": sample.get("batch_id") or "UNASSIGNED",
            "sample_condition": sample.get("sample_condition", "NOT_RECORDED"), "status": status, "label": label, "conflict": conflict,
            "stem": sample.get("stem", {}).get("state", "NOT_TESTED"),
            "element_count": len(sample["elements"]), "complete_conditions": bool(complete),
            "cohort": repr(cohort_key(sample)), "recipe": repr(recipe_key(sample)) if complete else None,
            **{key: numeric(sample.get(key)) for key in PARAMETERS},
            'pan_mass_g': numeric(sample.get('pan_mass_g')),
            'solvent_volume_mL': numeric(sample.get('solvent_volume_mL')),
            **{f"{element}_mM": numeric(concentrations.get(element)) for element in sample["elements"]},
            **{f"{element}_mmol": numeric(concentrations.get(element)) * numeric(sample.get('solvent_volume_mL')) / 1000.
               if numeric(concentrations.get(element)) is not None and numeric(sample.get('solvent_volume_mL')) is not None else None
               for element in sample['elements']},
        }
        rows.append(row)
    columns = ["sample_id", "batch_id", "sample_condition", "status", "label", "conflict", "stem", "element_count", "complete_conditions", "cohort", "recipe", *PARAMETERS]
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    for column in ["label", *PARAMETERS, *[c for c in frame if c.endswith("_mM")]]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.copy()


def wilson_interval(passed, labeled, z=1.959963984540054):
    """95% Wilson score interval for a descriptive sample-level proportion."""
    if labeled == 0:
        return None, None
    p = passed / labeled
    divisor = 1 + z * z / labeled
    center = (p + z * z / (2 * labeled)) / divisor
    half = z * sqrt(p * (1 - p) / labeled + z * z / (4 * labeled * labeled)) / divisor
    return max(0., center - half), min(1., center + half)


def metrics(frame):
    labeled = frame["label"].dropna()
    passed = int((labeled == 1).sum())
    return {
        "samples": len(frame), "labeled": len(labeled), "passed": passed,
        "failed": int((labeled == 0).sum()), "pending": int((frame["status"] == "PENDING").sum()),
        "conflicts": int(frame["conflict"].sum()),
        "pass_rate": passed / len(labeled) if len(labeled) else None,
        "complete_conditions": int(frame["complete_conditions"].sum()),
        "distinct_recipes": int(frame["recipe"].nunique()),
        "batches": int(frame.loc[frame["batch_id"] != "UNASSIGNED", "batch_id"].nunique()),
    }


def grouped_rates(frame, column):
    """Report denominators explicitly; unresolved and conflicting rows are excluded."""
    rows = []
    for value, group in frame.groupby(column, sort=True, observed=True):
        info = metrics(group)
        low, high = wilson_interval(info["passed"], info["labeled"])
        rows.append({column: value, **{key: info[key] for key in ("samples", "labeled", "passed", "failed", "pending", "conflicts", "pass_rate")}, "lower_95": low, "upper_95": high})
    return pd.DataFrame(rows, columns=[column, "samples", "labeled", "passed", "failed", "pending", "conflicts", "pass_rate", "lower_95", "upper_95"])


def parameter_rates(frame, parameter, bins=5):
    """Exact values for small designs; equal-frequency bins for continuous coverage."""
    observed = frame.loc[frame[parameter].notna()].copy()
    if observed.empty:
        return grouped_rates(observed.assign(parameter_range=pd.Series(dtype=str)), "parameter_range")
    if observed[parameter].nunique() <= bins:
        def value_text(value):
            return str(int(value)) if float(value).is_integer() else str(float(value))
        categories = [value_text(value) for value in sorted(observed[parameter].unique())]
        observed["parameter_range"] = pd.Categorical(observed[parameter].map(value_text), categories=categories, ordered=True)
    else:
        ranges = pd.qcut(observed[parameter], bins, duplicates="drop")
        observed["parameter_range"] = ranges.cat.rename_categories(str)
    return grouped_rates(observed, "parameter_range")


def correlations(frame, columns, minimum_pairs=3):
    """Pairwise Spearman association, with paired n and no significance claims."""
    rows = []
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            pair = frame[[left, right]].dropna()
            n = len(pair)
            rho = None
            if n >= minimum_pairs and all(pair[c].nunique() > 1 for c in (left, right)):
                rho = float(pair[left].rank().corr(pair[right].rank()))
            rows.append({"parameter_1": left, "parameter_2": right, "paired_n": n, "spearman_rho": rho})
    return pd.DataFrame(rows, columns=["parameter_1", "parameter_2", "paired_n", "spearman_rho"])
