"""Active sample records, deletion and evidence revisions."""
import copy
import json

import pandas as pd
import streamlit as st

from .dialogue import diagnose
from .domain import RULE_VERSION, STATUS_NAMES, screen, validate_sample, SAMPLE_CONDITIONS
from .queueing import prepare_reports
from .importers import sample_tables, csv_bytes, SAMPLE_COLUMNS, ICP_COLUMNS
from .ui_manual import render_sample_editor, notice


def invalidate_previews(origin):
    for key in (f"preview_{origin}", f"plan_note_{origin}", f"entry_preview_data_{origin}"):
        st.session_state.pop(key, None)


def render_records(store, samples, methods, concentration_text):
    origin = store.origin
    notice("records", origin)
    selection = st.radio("Record filter", ["All samples", "Pending samples", "Resolved samples"], horizontal=True, key=f"record_filter_{origin}")
    visible = [s for s in samples if selection == "All samples" or (screen(s, methods)["label"] is None) == (selection == "Pending samples")]
    rows = []
    for s in visible:
        result = screen(s, methods)
        rows.append({"Sample ID": s["sample_id"], "Sample condition": SAMPLE_CONDITIONS.get(s.get("sample_condition") or "NOT_RECORDED"), "Batch": s["batch_id"], "Element count": len(s["elements"]),
            "Concentration / mM": concentration_text(s["concentrations_mM"]), "Pulse count": s["pulse_count"],
            "Measured peak / °C": s.get("temperature_C"), "Cutoff / °C": s.get("pulse_limit_C"),
            "Voltage / V": s.get("voltage_V"), "Current / A": s.get("current_A"),
            "Screening status": STATUS_NAMES[result["status"]], "ICP": result["icp"], "XRD": result["xrd"], "STEM": result["stem"]})
    if rows:
        with st.expander('Screening summary'):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        from .ui_record_tables import render_record_tables
        render_record_tables(store, visible, methods)
    else:
        st.info("No samples in this view. Add a batch using Fill batch table, or select another record filter.")
    with st.expander("Delete samples"):
        st.caption('Delete selected records or compact automatic sample and batch numbering.')
        realign = st.checkbox("Realign SAC IDs after deletion", value=True, key=f"realign_after_delete_{origin}")
        ids = st.multiselect("Samples to delete", [s["sample_id"] for s in samples], key=f"delete_ids_{origin}")
        if st.button("Delete selected samples", disabled=not ids, key=f"delete_samples_{origin}"):
            try:
                store.delete_samples(ids, realign=realign)
                invalidate_previews(origin)
                st.session_state[f"record_action_{origin}"] = f"Deleted {len(ids)} samples. " + (f"SAC IDs realigned; next ID: {store.next_sample_ids()[0]}." if realign else "Remaining IDs were preserved.") + f" Next automatic Batch ID: {store.next_batch_id()}."
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Realign existing SAC IDs", disabled=not samples, key=f"realign_ids_{origin}"):
            mapping = store.realign_sample_ids()
            invalidate_previews(origin)
            st.session_state[f"record_action_{origin}"] = f"Realigned {sum(old != new for old, new in mapping.items())} SAC IDs. Next ID: {store.next_sample_ids()[0]}. Re-export files before reporting by upload."
            st.rerun()
        if origin == "REAL":
            st.divider()
            st.caption("Clear all REAL data permanently removes samples, methods, recommendations, chat and settings without backup. Automatic numbering restarts at SAC_0001. DEMO is unchanged.")
            if st.button("Clear all REAL data", key="clear_real_data"):
                store.clear_namespace()
                st.session_state.pop(f"record_action_{origin}", None)
                st.rerun()
    if st.session_state.get(f"record_action_{origin}"):
        st.success(st.session_state[f"record_action_{origin}"])
    if visible:
        selected_id = st.selectbox("Inspect sample", [s["sample_id"] for s in visible])
        sample = next(s for s in visible if s["sample_id"] == selected_id)
        st.markdown(diagnose(sample, methods))
        if sample.get("proposal"):
            with st.expander("Original proposal and transfer provenance"):
                st.caption("Predictions are preserved as planning metadata, not measurements. An actual synthesis batch ID and resolved outcomes complete the linked task.")
        assays = pd.DataFrame(sample.get("icp", []))
        if not assays.empty:
            st.dataframe(assays[["element", "value", "unit", "flag", "method_id", "source"]], hide_index=True, width="stretch", column_config={"value": st.column_config.TextColumn("W / wt%")})
        with st.expander("Edit experiment data", expanded=True):
            render_sample_editor(store, sample, methods)
        with st.expander("Update ICP / XRD judgments / optional STEM (revision history retained)"):
            with st.form("xrd_stem_update"):
                actual_batch = st.text_input("Actual synthesis batch ID", value=sample["batch_id"], help="Replace UNKNOWN after the actual experiment. A proposal ID is not an independent synthesis batch.")
                observed_temperature = st.text_input("Recorded measured peak / C", value="" if sample.get("temperature_C") is None else str(sample["temperature_C"]))
                observed_voltage = st.text_input("Recorded voltage / V", value="" if sample.get("voltage_V") is None else str(sample["voltage_V"]))
                observed_current = st.text_input("Recorded current / A", value="" if sample.get("current_A") is None else str(sample["current_A"]))
                icp_states = ["PENDING", "PASS", "FAIL"]
                current_icp = sample.get("icp_review", {}).get("state", "PENDING")
                if current_icp not in icp_states:
                    icp_states.insert(0, "Keep existing outcome")
                    current_icp = "Keep existing outcome"
                icp_state = st.selectbox("ICP judgment", icp_states, index=icp_states.index(current_icp))
                st.caption("PASS/FAIL judgments can be saved immediately. Files, values and reviewer details are optional. Record your judgment; assays and spectra are not analyzed here.")
                xrd_state = st.selectbox("XRD relevant-phase criterion", ["PENDING", "PASS", "FAIL"], index=["PENDING", "PASS", "FAIL"].index(sample["xrd"]["state"]))
                xfile = st.file_uploader("Optional: archive raw XRD file", key=f"xrd_raw_{origin}_{sample['sample_id']}")
                source = st.text_input("XRD source / file record", value=sample["xrd"].get("source", ""))
                criteria = st.text_input("Comparison criteria (support background / relevant phases / method)", value=sample["xrd"].get("criteria", ""))
                reviewer = st.text_input("Reviewer", value=sample["xrd"].get("reviewer", ""))
                stem = sample.get("stem", {})
                stem_states = ["NOT_TESTED", "CONFIRMED", "NOT_CONFIRMED", "INCONCLUSIVE"]
                stem_state = st.selectbox("Optional STEM confirmation", stem_states, index=stem_states.index(stem.get("state", "NOT_TESTED")))
                stem_source = st.text_input("STEM source", value=stem.get("source", ""))
                stem_reviewer = st.text_input("STEM reviewer", value=stem.get("reviewer", ""))
                stem_notes = st.text_input("STEM interpretation notes", value=stem.get("notes", ""))
                if st.form_submit_button("Save evidence revision"):
                    updated = copy.deepcopy(sample)
                    if xfile:
                        meta = store.source(xfile.name, xfile.getvalue())
                        updated["xrd_raw"] = meta
                        source = meta["file"]
                    if icp_state != "Keep existing outcome":
                        updated["icp_review"] = {"state": icp_state, "source": "Direct judgment through sample editor", "reviewer": reviewer}
                    updated["xrd"] = {"state": xrd_state, "mode": "MANUAL", "source": source, "criteria": criteria, "reviewer": reviewer}
                    updated["stem"] = {"state": stem_state, "source": stem_source, "reviewer": stem_reviewer, "notes": stem_notes}
                    updated.update(batch_id=actual_batch, temperature_C=observed_temperature, voltage_V=observed_voltage, current_A=observed_current)
                    try:
                        revised, queue_updates = prepare_reports([validate_sample(updated)], store, methods)
                        store.import_bundle(revised, [], overwrite=True, queue_updates=queue_updates)
                        invalidate_previews(origin)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
    ss, ii = sample_tables(samples)
    cols = st.columns(3)
    cols[0].download_button("Export samples CSV", csv_bytes(ss, SAMPLE_COLUMNS), "samples.csv", mime="text/csv")
    cols[1].download_button("Export ICP long table", csv_bytes(ii, ICP_COLUMNS), "icp.csv", mime="text/csv")
    payload = {**store.export(), "rule_version": RULE_VERSION, "current_labels": {s["sample_id"]: screen(s, methods) for s in samples}}
    cols[2].download_button("Export data and audit JSON", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), f"{origin.lower()}_audit_export.json", mime="application/json")
