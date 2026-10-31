"""English batch-entry UI with a native periodic-table selector."""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from .batching import BATCH_COLUMNS, SETTINGS_COLUMNS, blank_batch, batch_rows_to_standard
from .domain import parse_elements, SAMPLE_CONDITIONS
from .importers import csv_bytes, SAMPLE_COLUMNS, ICP_COLUMNS
from .ingestion import preview_import, commit_import
from .planning import fingerprint
from .ui_chemistry import chemistry_columns
from .precursors import to_columns as precursor_to_columns
from .ui_elements import periodic_selector


def render_batch_panel(store, context, methods, root):
    origin = store.origin
    from .ui_preparation import preparation_inputs
    from .preparation import from_mM, to_mM, precursor_cell, precursor_summary, DEFAULT_VOLTAGE_V, DEFAULT_CURRENT_A
    from .batching import clean
    unit, materials = preparation_inputs(f'entry_preparation_{origin}')
    cache_key = f'entry_cells_{origin}'
    cached_input = st.session_state.get(cache_key, {})
    if cached_input.get('material_defaults') and cached_input['material_defaults'] != materials:
        st.session_state[f'entry_epoch_{origin}'] = st.session_state.get(f'entry_epoch_{origin}', 0) + 1
    prior_unit = cached_input.get('input_unit', 'mM')
    if prior_unit != unit:
        row_volumes = {r['sample_id']: float(clean(r.get('solvent_volume_mL')) or materials['solvent_volume_mL'])
                       for r in cached_input.get('batch', [])}
        for detail in cached_input.get('detail', []):
            volume = row_volumes.get(detail['sample_id'], materials['solvent_volume_mL'])
            canonical = to_mM(detail.get('precursor_mM'), prior_unit, volume)
            detail['precursor_mM'] = from_mM(canonical, unit, volume)
        for row in cached_input.get('batch', []):
            text = str(row.get('concentration_mM') or '')
            if text and text.upper() != 'DETAIL':
                volume = row_volumes[row['sample_id']]
                converted = precursor_cell(text, parse_elements(row['target_elements']), prior_unit, volume)
                row['concentration_mM'] = precursor_summary({e: from_mM(v, unit, volume) for e, v in converted.items()}, list(converted))
        st.session_state[cache_key] = cached_input
        st.session_state[f'entry_epoch_{origin}'] = st.session_state.get(f'entry_epoch_{origin}', 0) + 1
    n = st.number_input("Samples in this reporting batch", min_value=1, max_value=100, value=6, key=f"batch_n_{origin}")
    id_mode = st.radio("Batch Sample ID numbering", ["Automatic", "Manual"], horizontal=True, key=f"entry_id_mode_{origin}")
    automatic = id_mode == "Automatic"
    next_ids = store.next_sample_ids(n)
    if automatic:
        st.caption(f"This batch: {next_ids[0]} to {next_ids[-1]}. Numbers advance after saving and continue across batches. Batch ID is stored separately.")
    else:
        st.caption("Edit Sample ID in the conditions table. IDs must be unique; SAC numeric IDs advance the automatic sequence after saving.")
    def load_sample_elements():
        scope = st.session_state[f"entry_scope_{origin}"]
        cached = st.session_state.get(f"entry_cells_{origin}", {}).get("batch", [])
        if scope and scope - 1 < len(cached):
            st.session_state[f"batch_elements_{origin}"] = parse_elements(cached[scope - 1].get("target_elements") or ";".join(context["elements"]))
    scope = st.selectbox("Periodic-table selection applies to", range(n + 1), format_func=lambda i: "All samples in this batch" if i == 0 else f"Sample row {i}",
                         key=f"entry_scope_{origin}", on_change=load_sample_elements)
    elements = periodic_selector(f"batch_elements_{origin}", context["elements"])
    if not elements:
        st.info("Select at least one target element to create the batch table.")
        return
    batch_key = f'entry_batch_id_{origin}'
    last_auto_key = f'entry_last_auto_batch_{origin}'
    next_batch = store.next_batch_id()
    if (st.session_state.pop(f'entry_batch_reset_{origin}', False) or not clean(st.session_state.get(batch_key))
            or st.session_state.get(batch_key) == st.session_state.get(last_auto_key) and st.session_state.get(batch_key) != next_batch):
        st.session_state[batch_key] = next_batch
        st.session_state[last_auto_key] = next_batch
    batch_id = clean(st.text_input("Synthesis batch ID", key=batch_key,
        help='Each saved entry group receives a new Batch ID. Edit it if these samples belong to an existing synthesis batch. Clearing the field restores the next automatic ID.')) or next_batch
    if st.session_state.get(f'entry_saved_batch_{origin}'):
        st.success(f"Saved {st.session_state[f'entry_saved_batch_{origin}']}. Next automatic Batch ID: {next_batch}.")
    st.caption('Each save creates a batch. Sample and batch IDs are assigned automatically.')
    from .experiment_defaults import shared_process_values
    process_defaults = shared_process_values(context)
    with st.expander("Shared process and evidence metadata", expanded=False):
        precursor = st.text_input("Batch precursor system", process_defaults["precursor_family"], key=f"entry_precursor_{origin}")
        support = st.text_input("Batch support system", process_defaults["support_batch"], key=f"entry_support_{origin}")
        protocol = st.text_input("Batch protocol", process_defaults["protocol_id"], key=f"entry_protocol_{origin}")
        xrd_source = st.text_input("Batch XRD/STEM source record", key=f"entry_xrd_source_{origin}")
        xrd_criteria = st.text_input("Batch XRD/STEM comparison criteria", key=f"entry_xrd_criteria_{origin}")
        reviewer = st.text_input("XRD/STEM reviewer", key=f"entry_reviewer_{origin}")
        stem_source = st.text_input("Optional STEM source record", key=f"entry_stem_source_{origin}")
        stem_reviewer = st.text_input("Optional STEM reviewer", key=f"entry_stem_reviewer_{origin}")
    settings = {"origin": origin, "batch_id": batch_id, "target_elements": ";".join(elements), "method_id": "",
                "precursor_family": precursor, "support_batch": support, "protocol_id": protocol,
                "xrd_source": xrd_source, "xrd_criteria": xrd_criteria, "xrd_reviewer": reviewer,
                "stem_source": stem_source, "stem_reviewer": stem_reviewer}
    settings.update(materials, precursor_input_unit=unit)
    st.caption("Save and continue with partial results. Choose ICP/XRD/STEM PASS or FAIL directly; files, reviewers and measured values are optional. This table records your judgments and does not analyze assays or spectra.")
    j1, j2, j3 = st.columns(3)
    all_icp = j1.selectbox("ICP judgment for all rows", ["PENDING", "PASS", "FAIL"], key=f"entry_all_icp_{origin}")
    all_xrd = j2.selectbox("XRD/STEM judgment for all rows", ["PENDING", "PASS", "FAIL"], key=f"entry_all_xrd_{origin}")
    if j3.button("Apply judgments to all rows", key=f"entry_apply_judgments_{origin}"):
        stored = st.session_state.get(f"entry_cells_{origin}", {})
        for row in stored.get("batch", []): row.update(icp_result=all_icp, xrd_result=all_xrd)
        st.session_state[f"entry_cells_{origin}"] = stored
        st.session_state[f"entry_epoch_{origin}"] = st.session_state.get(f"entry_epoch_{origin}", 0) + 1
        st.rerun()
    rows, details = blank_batch(next_ids, elements)
    cache = st.session_state.get(f"entry_cells_{origin}", {})
    for i, row in enumerate(rows):
        if i < len(cache.get("batch", [])): row.update(cache["batch"][i])
        if i >= len(cache.get('batch', [])) or not cache.get('electrical_defaults_applied'):
            for field, value in [('voltage_V', DEFAULT_VOLTAGE_V), ('current_A', DEFAULT_CURRENT_A)]:
                if not clean(row.get(field)):
                    row[field] = value
        if row.get("icp_result") == "AUTO": row["icp_result"] = "PENDING"
        if automatic: row["sample_id"] = next_ids[i]
        if scope == 0 or scope == i + 1: row["target_elements"] = ";".join(elements)
        for field, value in materials.items():
            if row.get(field) is None or row.get(field) == '' or row.get(field) == cache.get('material_defaults', {}).get(field):
                row[field] = value
    prior_details = {(r["sample_id"], r["element"]): r for r in cache.get("detail", [])}
    sig = fingerprint({"origin": origin, "compositions": [r["target_elements"] for r in rows], "n": n, "batch_id": batch_id, "id_mode": id_mode, "ids": next_ids if automatic else [], "epoch": st.session_state.get(f"entry_epoch_{origin}", 0)})
    st.write("**Sample conditions and results**")
    st.caption('Record complete material loss as BURNED_OUT; two failed tests alone do not establish burnout.')
    st.caption('Enter a value with mmol or mM; mixed recipes accept per-element entries such as Fe:1 mmol; Co:0.5 mmol.')
    batch = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", key=f"batch_grid_{sig}",
        disabled=["target_elements"] + (["sample_id"] if automatic else []),
        column_order=["sample_id", "sample_condition", "target_elements", "concentration_mM", "pan_mass_g", "solvent_volume_mL", "pulse", "pulse_limit_C", "icp_result", "xrd_result", "temperature_C", "voltage_V", "current_A"],
        column_config={'concentration_mM': st.column_config.TextColumn('Precursor salt / ' + unit, help='Explicit mM or mmol units are also accepted in each cell.'),
                       'pan_mass_g': st.column_config.NumberColumn('PAN / g', min_value=.000001),
                       'solvent_volume_mL': st.column_config.NumberColumn('DMF / mL', min_value=.000001),
                       "sample_condition": st.column_config.SelectboxColumn("Sample condition", options=list(SAMPLE_CONDITIONS)), "icp_result": st.column_config.SelectboxColumn(options=["PENDING", "PASS", "FAIL"]),
                       "xrd_result": st.column_config.SelectboxColumn("XRD/STEM assessment", options=["PENDING", "PASS", "FAIL"]),
                       "stem_result": None})
    if automatic:
        # Enforce automatic IDs independently of client-side editor restrictions.
        batch["sample_id"] = next_ids
    synchronization_errors = []
    top_changed = False
    for i, row in enumerate(batch.to_dict('records')):
        previous = cache.get('batch', [])[i] if i < len(cache.get('batch', [])) else {}
        if clean(row.get('concentration_mM')) == clean(previous.get('concentration_mM')):
            continue
        try:
            volume = float(clean(row.get('solvent_volume_mL')) or materials['solvent_volume_mL'])
            canonical = precursor_cell(row.get('concentration_mM'), parse_elements(row['target_elements']), unit, volume)
            if canonical is not None:
                for element, value in canonical.items():
                    cell = prior_details.setdefault((row['sample_id'], element), {'sample_id': row['sample_id'], 'element': element})
                    cell['precursor_mM'] = from_mM(value, unit, volume)
                top_changed = True
        except ValueError as exc:
            synchronization_errors.append(row['sample_id'] + ': ' + str(exc))
    for message in synchronization_errors:
        st.error(message)
    st.write("**Element concentrations and measured ICP values**")
    st.caption('Enter precursor quantities and measured ICP W (wt%). Unmeasured values may remain blank.')
    details = []
    for i, row in enumerate(batch.to_dict("records")):
        _, initial = blank_batch([row["sample_id"]], parse_elements(row["target_elements"]))
        if unit == 'mmol':
            for item in initial:
                item['precursor_mM'] = from_mM(item['precursor_mM'], unit, row.get('solvent_volume_mL') or materials['solvent_volume_mL'])
        from .ui_materials import planning_precursors
        chemistry_context = {**context, 'precursors': planning_precursors(
            store, context, [r['element'] for r in initial])}
        old_id = cache.get("batch", [])[i]["sample_id"] if i < len(cache.get("batch", [])) else row["sample_id"]
        details.extend({**r, **precursor_to_columns(chemistry_context, r['element']), **prior_details.get((r["sample_id"], r["element"]), prior_details.get((old_id, r["element"]), {})), "sample_id": row["sample_id"]} for r in initial)
    st.caption('Editable chloride defaults are provided. Confirm the actual reagent before using state-specific descriptors.')
    opposite_unit = 'mmol' if unit == 'mM' else 'mM'
    volumes = {r['sample_id']: float(clean(r.get('solvent_volume_mL')) or materials['solvent_volume_mL']) for r in batch.to_dict('records')}
    for item in details:
        try:
            value = to_mM(item.get('precursor_mM'), unit, volumes[item['sample_id']])
            item['converted_precursor'] = from_mM(value, opposite_unit, volumes[item['sample_id']])
        except ValueError as exc:
            item['converted_precursor'] = None
            st.error(item['sample_id'] + ': ' + str(exc))
    detail_sig = fingerprint({"sig": sig, "top_values": batch['concentration_mM'].tolist(), 'volumes': volumes})
    detail = st.data_editor(pd.DataFrame(details), hide_index=True, width="stretch", key=f"element_grid_{detail_sig}",
                           disabled=["sample_id", "element", "unit", "converted_precursor"], column_config={"converted_precursor": st.column_config.NumberColumn('Equivalent / ' + opposite_unit, format='%.8g'), "value": st.column_config.TextColumn("W / wt%", help="ICP mass percent: 0.25 means 0.25 wt%. ND, BLQ and <0.01 are supported."), "precursor_mM": st.column_config.NumberColumn("Precursor salt / " + unit, min_value=1e-12), "flag": st.column_config.SelectboxColumn(options=["", "MEASURED", "ND", "BLQ", "MISSING"]), **chemistry_columns()})
    synchronized = False
    if not synchronization_errors:
        for index, row in batch.iterrows():
            elements_for_row = parse_elements(row['target_elements'])
            amounts = {r['element']: r.get('precursor_mM') for r in detail.to_dict('records') if r['sample_id'] == row['sample_id']}
            try:
                summary = precursor_summary(amounts, elements_for_row)
            except ValueError as exc:
                st.error(row['sample_id'] + ': ' + str(exc))
                continue
            if clean(row['concentration_mM']) != summary:
                batch.loc[index, 'concentration_mM'] = summary
                synchronized = True
        # A changed element value refreshes its read-only converted counterpart.
        synchronized = synchronized or any(clean(a.get('precursor_mM')) != clean(b.get('precursor_mM')) for a, b in zip(details, detail.to_dict('records')))
    detail = detail.drop(columns=['converted_precursor'], errors='ignore')
    prior_details.update({(r["sample_id"], r["element"]): r for r in detail.to_dict("records")})
    st.session_state[f"entry_cells_{origin}"] = {"batch": batch.to_dict("records"), "detail": list(prior_details.values()), 'input_unit': unit, 'material_defaults': materials, 'electrical_defaults_applied': True}
    revise = st.checkbox("Revise existing batch sample IDs with history", disabled=automatic, key=f"entry_revise_{origin}") and not automatic
    entry_sig = fingerprint({"batch": batch.to_dict("records"), "detail": detail.to_dict("records"), "settings": settings, "revise": revise})
    if st.button("Preview filled batch", type="primary", key=f"entry_preview_{origin}"):
        try:
            ss, ii = batch_rows_to_standard(batch.to_dict("records"), detail.to_dict("records"), settings)
            files = [("interactive_samples.csv", csv_bytes(ss, SAMPLE_COLUMNS)), ("interactive_icp.csv", csv_bytes(ii, ICP_COLUMNS))]
            checked = preview_import(files, store, revise)
            batch_by_id={s['sample_id']:s['batch_id'] for s in checked.samples}
            if any(value != batch_id for value in batch_by_id.values()):
                raise ValueError('The batch preview did not retain its synthesis Batch ID. Preview again before saving.')
            st.session_state[f"entry_preview_data_{origin}"] = {"files": files, "sig": entry_sig, "revise": revise,
                "rows": [{**row,'batch_id':batch_by_id[row['sample_id']]} for row in checked.rows]}
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    saved = st.session_state.get(f"entry_preview_data_{origin}")
    if saved and saved["sig"] == entry_sig:
        st.dataframe(pd.DataFrame(saved["rows"]), hide_index=True, width="stretch")
        if st.button("Save filled batch", key=f"entry_save_{origin}"):
            try:
                result = commit_import(preview_import(saved["files"], store, saved["revise"]), store)
                st.session_state[f"last_import_{origin}"] = result
                st.session_state[f'entry_saved_batch_{origin}'] = batch_id
                st.session_state.pop(f"entry_cells_{origin}", None)
                st.session_state.pop(f"entry_preview_data_{origin}", None)
                st.session_state[f'entry_batch_reset_{origin}'] = True
                st.session_state[f"entry_epoch_{origin}"] = st.session_state.get(f"entry_epoch_{origin}", 0) + 1
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    if st.button("Start an empty entry table", key=f"entry_reset_{origin}"):
        st.session_state.pop(f"entry_cells_{origin}", None)
        st.session_state.pop(f"entry_preview_data_{origin}", None)
        st.session_state.pop(f"batch_grid_{sig}", None)
        st.session_state.pop(f"element_grid_{sig}", None)
        st.session_state[f'entry_batch_reset_{origin}'] = True
        st.session_state[f"entry_epoch_{origin}"] = st.session_state.get(f"entry_epoch_{origin}", 0) + 1
        st.rerun()
    a, b, c = st.columns(3)
    a.download_button("Export filled batch CSV", csv_bytes(batch.to_dict("records"), BATCH_COLUMNS), f"{origin.lower()}_filled_batch.csv", key=f"entry_export_batch_{origin}")
    b.download_button("Export element detail CSV", csv_bytes(detail.to_dict("records"), ICP_COLUMNS), f"{origin.lower()}_element_detail.csv", key=f"entry_export_detail_{origin}")
    c.download_button("Export batch settings CSV", csv_bytes([settings], SETTINGS_COLUMNS), f"{origin.lower()}_batch_settings.csv", key=f"entry_export_settings_{origin}")
    path = root / "examples/compact_batch_template.xlsx"
    if path.exists():
        st.download_button("Download compact Excel template", path.read_bytes(), "SACFactory_batch_template.xlsx", key=f"entry_template_{origin}")
    if synchronized or top_changed:
        st.session_state[f'entry_epoch_{origin}'] = st.session_state.get(f'entry_epoch_{origin}', 0) + 1
        st.rerun()
