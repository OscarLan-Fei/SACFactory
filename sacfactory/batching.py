"""Compact batch tables and element selection for the manual synthesis loop."""
from __future__ import annotations

from .units import normalize_assay
from .domain import ELEMENTS, base_concentration, finite_number, parse_elements
from .importers import SAMPLE_COLUMNS, ICP_COLUMNS
from .precursors import COLUMNS as PRECURSOR_COLUMNS, to_columns as precursor_to_columns

BATCH_COLUMNS = ["sample_id", "concentration_mM", "pulse", "voltage_V", "current_A", "temperature_C", "pulse_limit_C", "icp_result", "xrd_result", "stem_result", "target_elements", "sample_condition", "pan_mass_g", "solvent_volume_mL"]
SETTINGS_COLUMNS = ["batch_id", "origin", "target_elements", "precursor_family", "support_batch", "protocol_id", "method_id", "xrd_source", "xrd_criteria", "xrd_reviewer", "stem_source", "stem_reviewer", "pan_mass_g", "solvent_volume_mL", "precursor_input_unit"]

# Group positions are one-based. Ln and An link the detached series visually.
PERIODS = [
    {1: "H", 18: "He"},
    dict(zip([1, 2, 13, 14, 15, 16, 17, 18], "Li Be B C N O F Ne".split())),
    dict(zip([1, 2, 13, 14, 15, 16, 17, 18], "Na Mg Al Si P S Cl Ar".split())),
    dict(enumerate("K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split(), 1)),
    dict(enumerate("Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe".split(), 1)),
    dict(enumerate("Cs Ba Ln Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split(), 1)),
    dict(enumerate("Fr Ra An Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split(), 1)),
    dict(enumerate("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split(), 3)),
    dict(enumerate("Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split(), 3)),
]


def clean(value):
    return "" if value is None or str(value).strip().lower() == "nan" else str(value).strip()


def pairs(text, name):
    result = {}
    for token in clean(text).split(";"):
        if not token.strip(): continue
        parts = token.split(":")
        if len(parts) != 2 or parts[0].strip() in result:
            raise ValueError(f"{name}: use unique Element:value pairs separated by semicolons.")
        result[parts[0].strip()] = parts[1].strip()
    return result


def batch_rows_to_standard(rows, details, settings, source="Interactive batch table"):
    """Normalize compact human input without inventing measurements or limits."""
    default_elements = parse_elements(settings["target_elements"])
    detail_map = {}
    for item in details:
        row = normalize_assay({k: clean(v) for k, v in item.items()})
        if not row.get("sample_id") and not row.get("element"): continue
        key = (row.get("sample_id"), row.get("element"))
        if key in detail_map or key[1] not in ELEMENTS:
            raise ValueError("Duplicate or unsupported element in the ICP detail table.")
        detail_map[key] = row
    sample_rows, icp_rows = [], []
    ids = set()
    for item in rows:
        row = {k: clean(v) for k, v in item.items()}
        if not row.get("sample_id"):
            if any(row.values()): raise ValueError("Every populated batch row needs a Sample ID.")
            continue
        ident = row["sample_id"]
        elements = parse_elements(row.get("target_elements") or ";".join(default_elements))
        if ident in ids: raise ValueError("Duplicate Sample ID in the batch table.")
        ids.add(ident)
        concentration = row.get("concentration_mM", "")
        if concentration.upper() in {"DETAIL", ""}:
            conc = {e: detail_map.get((ident, e), {}).get("precursor_mM", "") for e in elements}
        elif ":" in concentration:
            conc = pairs(concentration, "Concentration")
            if set(conc) != set(elements): raise ValueError("Concentration pairs must match every selected target element.")
        else:
            conc = {e: concentration for e in elements}
        from .preparation import material_inputs, precursor_value
        material = material_inputs(row.get('pan_mass_g') or settings.get('pan_mass_g') or 2.,
                                   row.get('solvent_volume_mL') or settings.get('solvent_volume_mL') or 20.)
        unit = settings.get('precursor_input_unit') or 'mM'
        amount_input = unit == 'mmol' or any(str(v).strip().endswith('mmol') for v in conc.values())
        conc = {e: precursor_value(v, unit, material['solvent_volume_mL']) for e, v in conc.items()}
        judgment = row.get("icp_result", "").upper()
        icp_state = judgment if judgment in {"AUTO", "PASS", "FAIL", "PENDING"} else "AUTO"
        inline = {} if judgment in {"AUTO", "PASS", "FAIL", "PENDING"} else pairs(row.get("icp_result", ""), "ICP result")
        if set(inline) - set(elements): raise ValueError("Inline ICP results must refer to selected target elements.")
        out = {k: "" for k in SAMPLE_COLUMNS}
        out.update({k: clean(settings.get(k)) for k in ("batch_id", "origin", "precursor_family", "support_batch", "protocol_id", "xrd_source", "xrd_criteria", "xrd_reviewer", "stem_source", "stem_reviewer")})
        out.update(sample_id=ident, target_elements=";".join(elements), alpha="", pulse_count=row.get("pulse", ""), sample_condition=row.get("sample_condition") or "NOT_RECORDED", record_uid=row.get("record_uid", ""),
                   icp_state=icp_state, icp_source=source if icp_state != "AUTO" else "", xrd_mode="MANUAL",
                   temperature_C=row.get("temperature_C", ""), pulse_limit_C=row.get("pulse_limit_C", ""), voltage_V=row.get("voltage_V", ""), current_A=row.get("current_A", ""),
                   xrd_state=row.get("xrd_result") or "PENDING", stem_state=row.get("stem_result") or "NOT_TESTED", notes="Reported through compact batch table")
        out['electrical_input_kind'] = 'SETTING'
        if any(row.get(k) or settings.get(k) for k in ('pan_mass_g', 'solvent_volume_mL')) or amount_input:
            out.update(material, preparation_basis='User-declared preparation; protocol defaults if unchanged', concentration_input_unit=unit)
        if out["xrd_state"] not in {"PASS", "FAIL", "PENDING"}: raise ValueError("XRD result must be PASS, FAIL or PENDING.")
        if not out["xrd_source"]: out["xrd_source"] = source
        if not out["stem_source"] and out["stem_state"] != "NOT_TESTED": out["stem_source"] = source
        sample_rows.append(out)
        for e in elements + [e for sid, e in detail_map if sid == ident and e not in elements]:
            detail = detail_map.get((ident, e), {})
            detail_value = detail.get("value", "")
            detail_flag = detail.get("flag", "")
            if e in inline and (detail_value or detail_flag in {"ND", "BLQ"}):
                raise ValueError("Report each ICP value either inline or in the detail table, not both.")
            value = inline.get(e, detail_value)
            inferred_flag = "ND" if value.upper() in {"ND", "N.D."} else "BLQ" if value.upper() in {"BLQ", "<LOQ", "BLOQ"} else "MEASURED" if value else "MISSING"
            flag = inferred_flag if e in inline else (detail_flag or inferred_flag)
            if value and flag == "MISSING": raise ValueError("An entered ICP value cannot retain the MISSING flag; choose MEASURED, ND or BLQ.")
            icp_rows.append({"sample_id": ident, "element": e, "value": value, "unit": detail.get("unit") or "wt%", "flag": flag,
                             "method_id": detail.get("method_id") or clean(settings.get("method_id")), "source": detail.get("source") or source, "precursor_mM": str(conc[e]) if conc.get(e) is not None else "", **{k: detail.get(k, '') for k in PRECURSOR_COLUMNS}})
    if {sid for sid, e in detail_map} - ids: raise ValueError("ICP detail contains Sample IDs absent from the batch table.")
    if not sample_rows: raise ValueError("Enter at least one sample.")
    return sample_rows, icp_rows


def blank_batch(sample_ids, elements):
    rows = [{**{k: "" for k in BATCH_COLUMNS}, "sample_id": sid, "sample_condition": "NOT_RECORDED", "concentration_mM": "DETAIL", "pulse": 1, "icp_result": "AUTO", "xrd_result": "PENDING", "stem_result": "NOT_TESTED", "target_elements": ";".join(elements)} for sid in sample_ids]
    details = [{"sample_id": sid, "element": e, "precursor_mM": base_concentration(e), "value": "", "unit": "wt%", "flag": "", "method_id": "", "source": "", **{k: None if k == 'precursor_oxidation_state' else '' for k in PRECURSOR_COLUMNS}} for sid in sample_ids for e in elements]
    return rows, details


def compact_reporting_zip(recommendations):
    """Fresh editable condition/detail/settings CSVs for each proposed batch."""
    import io
    import zipfile
    from .importers import csv_bytes
    if not recommendations: raise ValueError("No proposed experiments.")
    batch, details = [], []
    for proposal in recommendations:
        recipe = proposal["recipe"]
        rows, assays = blank_batch([proposal["recommendation_id"]], recipe["elements"])
        rows[0].update(pulse=recipe["pulse_count"], pulse_limit_C=recipe.get("pulse_limit_C") or "")
        rows[0].update({k: recipe[k] for k in ('pan_mass_g', 'solvent_volume_mL') if k in recipe})
        for a in assays:
            a["precursor_mM"] = recipe["concentrations_mM"][a["element"]]
            a.update(precursor_to_columns(recipe, a['element']))
        batch.extend(rows); details.extend(assays)
    first = recommendations[0]
    settings = {k: "" for k in SETTINGS_COLUMNS}
    settings.update(batch_id="REPLACE_WITH_SYNTHESIS_BATCH_ID", origin=first["origin"], target_elements=";".join(first["recipe"]["elements"]),
                    **{k: first["recipe"][k] for k in ("precursor_family", "support_batch", "protocol_id")})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in (("batch.csv", csv_bytes(batch, BATCH_COLUMNS)), ("icp.csv", csv_bytes(details, ICP_COLUMNS)), ("settings.csv", csv_bytes([settings], SETTINGS_COLUMNS))):
            archive.writestr(name, data)
    return output.getvalue()
