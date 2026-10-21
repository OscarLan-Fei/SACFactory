"""English interfaces for full record editing, plans and manual batch feedback."""
import copy

import pandas as pd
import streamlit as st

from .domain import ELEMENT_ORDER, parse_elements, screen, STATUS_NAMES, SAMPLE_CONDITIONS
from .manual import element_rows, apply_element_rows, save_manual_revisions, pending_samples, has_observations
from .planning import fingerprint
from .ui_chemistry import render_precursor_editor, chemistry_columns
from .precursors import normalize_precursors, confirm_catalog_precursors
from .ui_elements import element_multiselect


def blank(value):
    return "" if value is None else str(value)


def clean_cells(rows):
    """Normalize nullable string/numeric editor cells before scalar validation."""
    return [{k: None if pd.isna(v) else v for k, v in r.items()} for r in rows]


def saved(store, result, scope):
    for key in (f"preview_{store.origin}", f"plan_note_{store.origin}", f"entry_preview_data_{store.origin}"):
        st.session_state.pop(key, None)
    st.session_state[f"manual_notice_{scope}_{store.origin}"] = (
        f"Saved {result['samples']} sample revisions. Linked tasks: {result['completed_tasks']} completed, {result['open_tasks']} open. "
        "Next batch will use the updated eligible observations.")
    st.rerun()


def notice(scope, origin):
    if st.session_state.get(f"manual_notice_{scope}_{origin}"):
        st.success(st.session_state[f"manual_notice_{scope}_{origin}"])


def render_sample_editor(store, sample, methods, mode="records"):
    """Edit one identity; source/proposal history stays outside editable fields."""
    pending = mode == "pending"
    origin, ident = store.origin, sample["sample_id"]
    prefix = f"manual_{mode}_{origin}_{ident}_{fingerprint(sample)}"
    st.caption(f"Sample ID: {ident}. This identity is fixed; changes create an audited revision of the same record. Blank optional fields remain unrecorded.")
    elements = element_multiselect("Planned target elements" if pending else "Record target elements", ELEMENT_ORDER,
                             default=sample["elements"], key=prefix + "_elements")
    if not elements:
        st.info("Select at least one target element.")
        return
    elements = parse_elements(";".join(elements))
    from .ui_preparation import preparation_inputs
    from .preparation import from_mM, to_mM
    unit, materials = preparation_inputs(prefix + '_preparation', sample)
    from .reagent_links import link_precursors
    library = (store.get('meta', 'reagent_library') or {}).get('precursors', {})
    linked = link_precursors({**sample, 'elements': elements}, library)
    with st.form(prefix + "_form"):
        batch = st.text_input("Pending batch ID (optional)" if pending else "Record actual batch ID", blank(sample.get("batch_id")), key=prefix + "_batch")
        pulse = st.text_input("Planned pulse count" if pending else "Record pulse count", blank(sample.get("pulse_count")), key=prefix + "_pulse")
        cutoff = st.text_input("Planned cutoff / C" if pending else "Record controlled cutoff / C", blank(sample.get("pulse_limit_C")), key=prefix + "_cutoff")
        if pending:
            table = [{"element": e, "precursor_mM": sample["concentrations_mM"].get(e)} for e in elements]
        else:
            table = element_rows(linked, elements)
        for row in table:
            row['precursor_mM'] = from_mM(row.get('precursor_mM'), unit, materials['solvent_volume_mL'])
        columns = {"precursor_mM": st.column_config.NumberColumn("Precursor / " + unit, min_value=.000001, format="%.6f"),
                   "value": st.column_config.TextColumn("W / wt%", help="Enter 0.25 for 0.25 wt%. ND, BLQ and <0.005 are supported."),
                   "flag": st.column_config.SelectboxColumn("ICP flag", options=["", "MEASURED", "ND", "BLQ", "MISSING"])}
        columns.update(chemistry_columns())
        st.write('**Planned precursor salt / ' + unit + '**' if pending else "**Element precursor inputs and ICP W / wt%**")
        if not pending:
            st.caption("ICP content is mass percent. Flags are inferred for newly entered values; ND and BLQ remain explicit. Method IDs and source files are optional for direct PASS/FAIL.")
        edited = st.data_editor(pd.DataFrame(table), disabled=["element"] + ([] if pending else ["unit"]),
            hide_index=True, width="stretch", column_config=columns, key=prefix + "_table_" + fingerprint([elements, unit, materials['solvent_volume_mL']]))
        if pending:
            from .ui_materials import planning_precursors
            pending_chemistry = render_precursor_editor(elements,
                {**planning_precursors(store, {}, elements),
                 **{e: v for e, v in sample.get('precursors', {}).items() if e in elements}},
                prefix + '_chemistry')
        precursor = st.text_input("Planned precursor system" if pending else "Record precursor system", sample["precursor_family"], key=prefix + "_precursor")
        support = st.text_input("Planned support system" if pending else "Record support system", sample["support_batch"], key=prefix + "_support")
        protocol = st.text_input("Planned protocol" if pending else "Record protocol", sample["protocol_id"], key=prefix + "_protocol")
        notes = st.text_area("Pending notes" if pending else "Record notes", sample.get("notes", ""), key=prefix + "_notes")
        confirm_catalog = st.checkbox('Confirm the listed catalogue precursor identities were actually used', value=False,
            help='Optional. Confirm only after checking each exact reagent/hydrate; otherwise catalogue assumptions stay masked during training.', key=prefix + '_confirm_catalog') if not pending else False
        if not pending:
            condition = st.selectbox("Record sample condition", list(SAMPLE_CONDITIONS), index=list(SAMPLE_CONDITIONS).index(sample.get("sample_condition") or "NOT_RECORDED"), format_func=SAMPLE_CONDITIONS.get, key=prefix + "_condition")
            peak = st.text_input("Record measured peak / C", blank(sample.get("temperature_C")), key=prefix + "_peak")
            voltage = st.text_input("Record voltage / V", blank(sample.get("voltage_V")), key=prefix + "_voltage")
            current = st.text_input("Record current / A", blank(sample.get("current_A")), key=prefix + "_current")
            icp_options = ["PENDING", "PASS", "FAIL"]
            existing = sample.get("icp_review", {}).get("state", "PENDING")
            if existing == "AUTO":
                icp_options.insert(0, "Keep numeric criterion")
                existing = "Keep numeric criterion"
            icp = st.selectbox("Record ICP judgment", icp_options, index=icp_options.index(existing), key=prefix + "_icp")
            xrd = st.selectbox("Record XRD judgment", ["PENDING", "PASS", "FAIL"], index=["PENDING", "PASS", "FAIL"].index(sample["xrd"].get("state", "PENDING")), key=prefix + "_xrd")
            stem_options = ["NOT_TESTED", "CONFIRMED", "NOT_CONFIRMED", "INCONCLUSIVE"]
            stem = st.selectbox("Record optional STEM", stem_options, index=stem_options.index(sample.get("stem", {}).get("state", "NOT_TESTED")), key=prefix + "_stem")
            stem_source = st.text_input("Record STEM source (optional)", sample.get("stem", {}).get("source", ""), key=prefix + "_stem_source")
            stem_reviewer = st.text_input("Record STEM reviewer (optional)", sample.get("stem", {}).get("reviewer", ""), key=prefix + "_stem_reviewer")
            st.caption("Direct ICP/XRD judgments save without measurements or files. Corrections to a linked recipe retain its original proposal and explicitly record any deviation.")
        submitted = st.form_submit_button("Save pending changes" if pending else "Save experiment changes", type="primary")
    if submitted:
        try:
            updated = copy.deepcopy(sample)
            updated.update(elements=elements, batch_id=batch, pulse_count=pulse, pulse_limit_C=cutoff,
                precursor_family=precursor, support_batch=support, protocol_id=protocol, notes=notes)
            updated.update(materials, preparation_basis='User-declared preparation; protocol defaults if unchanged', concentration_input_unit=unit)
            rows = clean_cells(edited.to_dict("records"))
            for row in rows:
                row['precursor_mM'] = to_mM(row.get('precursor_mM'), unit, materials['solvent_volume_mL'])
            if pending:
                updated["concentrations_mM"] = {r["element"]: r["precursor_mM"] for r in rows}
                updated['precursors'] = normalize_precursors(pending_chemistry, elements)
            else:
                updated = apply_element_rows(updated, rows, preserve_assays=[a for a in sample.get("icp", []) if a["element"] not in sample["elements"]])
                if confirm_catalog:
                    updated = confirm_catalog_precursors(updated)
                updated.update(sample_condition=condition, temperature_C=peak, voltage_V=voltage, current_A=current)
                if icp != "Keep numeric criterion":
                    updated["icp_review"] = {**updated.get("icp_review", {}), "state": icp, "source": "Manual experiment record edit"}
                updated["xrd"] = {**updated["xrd"], "state": xrd, "mode": "MANUAL"}
                updated["stem"] = {**updated.get("stem", {}), "state": stem, "source": stem_source, "reviewer": stem_reviewer}
            result = save_manual_revisions(store, [updated], methods, mode, {ident: fingerprint(sample)})
            saved(store, result, mode)
        except (ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))


def render_pending(store, samples, methods):
    notice("pending", store.origin)
    pending = pending_samples(samples, store.all("recommendation"), methods)
    st.subheader("Pending experiments")
    st.caption("Edit planned targets, precursor concentrations, pulses and cutoff before synthesis. A changed recipe invalidates its old estimate and keeps the original proposal in history. Enter measured results under Result feedback.")
    if not pending:
        st.info("No pending samples. Register samples or use Next batch > Add batch to pending.")
        return
    st.dataframe(pd.DataFrame([{"Sample ID": s["sample_id"], "Batch": s["batch_id"], "Targets": ";".join(s["elements"]),
        "Pulses": s["pulse_count"], "Cutoff / C": s.get("pulse_limit_C"),
        "Current predicted pass probability": current_pending_probability(s),
        "Screening": STATUS_NAMES[screen(s, methods)["status"]]} for s in pending]), hide_index=True, width="stretch")
    editable = [s for s in pending if not has_observations(s) and not s.get('result_reported_at')]
    if editable:
        st.write('**Edit pending parameters in tables**')
        signature = fingerprint(editable)
        with st.form(f'pending_parameters_{store.origin}_{signature}'):
            conditions = st.data_editor(pd.DataFrame([{'sample_id': s['sample_id'], 'pulse_count': s['pulse_count'],
                'pulse_limit_C': s.get('pulse_limit_C'), 'notes': s.get('notes', '')} for s in editable]),
                disabled=['sample_id'], hide_index=True, width='stretch',
                column_config={'pulse_count': st.column_config.NumberColumn('Pulses', min_value=1, step=1),
                               'pulse_limit_C': st.column_config.NumberColumn('Controlled cutoff / C', min_value=0.000001)},
                key=f'pending_conditions_{store.origin}_{signature}')
            concentrations = st.data_editor(pd.DataFrame([{'sample_id': s['sample_id'], 'element': e,
                'concentration_mM': s['concentrations_mM'].get(e)} for s in editable for e in s['elements']]),
                disabled=['sample_id', 'element'], hide_index=True, width='stretch',
                column_config={'concentration_mM': st.column_config.NumberColumn('Precursor / mM', min_value=.000001, format='%.6f')},
                key=f'pending_concentrations_{store.origin}_{signature}')
            submitted = st.form_submit_button('Save pending table changes', type='primary')
        if submitted:
            try:
                originals = {s['sample_id']: s for s in editable}
                details = clean_cells(concentrations.to_dict('records'))
                updated = []
                for row in clean_cells(conditions.to_dict('records')):
                    sample = copy.deepcopy(originals[row['sample_id']])
                    sample.update({k: row[k] for k in ('pulse_count', 'pulse_limit_C', 'notes')})
                    sample['concentrations_mM'] = {r['element']: r['concentration_mM'] for r in details if r['sample_id'] == sample['sample_id']}
                    updated.append(sample)
                result = save_manual_revisions(store, updated, methods, 'pending', {s['sample_id']: fingerprint(s) for s in editable})
                saved(store, result, 'pending')
            except (ValueError, TypeError, KeyError) as exc:
                st.error(str(exc))
    eligible_ids=[s['sample_id'] for s in pending if not has_observations(s) and not s.get('result_reported_at')]
    key=f'pending_reforecast_samples_{store.origin}'
    if key in st.session_state:
        st.session_state[key]=[i for i in st.session_state[key] if i in eligible_ids]
    if st.button('Select all unmeasured pending samples',key=key+'_all'):
        st.session_state[key]=eligible_ids
    forecast_ids=st.multiselect('Pending samples to reforecast',eligible_ids,key=key)
    if st.button('Predict selected pending parameters',disabled=not forecast_ids,key=f'pending_predict_batch_{store.origin}'):
        from .pending_prediction import refresh_pending_prediction
        by_id={s['sample_id']:s for s in pending};report=[]
        progress=st.progress(0.)
        try:
            for index,sample_id in enumerate(forecast_ids,1):
                try:
                    refresh_pending_prediction(store,sample_id,fingerprint(by_id[sample_id]))
                    report.append({'sample_id':sample_id,'status':'Forecast metadata saved'})
                except (ValueError,TypeError,KeyError) as exc:
                    report.append({'sample_id':sample_id,'status':'Unavailable','reason':str(exc)})
                progress.progress(index/len(forecast_ids),text=f'Predicted sample {index} / {len(forecast_ids)}')
        finally:
            progress.empty()
        st.session_state[f'pending_batch_forecast_report_{store.origin}']=report
        st.rerun()
    if st.session_state.get(f'pending_batch_forecast_report_{store.origin}'):
        st.dataframe(pd.DataFrame(st.session_state[f'pending_batch_forecast_report_{store.origin}']),hide_index=True,width='stretch')
    ident = st.selectbox("Pending sample to edit", [s["sample_id"] for s in pending], key=f"manual_pending_choice_{store.origin}")
    sample = next(s for s in pending if s['sample_id'] == ident)
    probability = current_pending_probability(sample)
    if probability is not None:
        st.metric('Current pending predicted screening probability', f'{probability:.1%}')
    from .mixing import forecast_metadata
    current_forecast = sample.get('manual_prediction') or sample.get('proposal') or {}
    if current_forecast.get('sac_gen_prediction'):
        from .queueing import controlled_recipe_key
        stale=current_forecast.get('prediction_stale') or (current_forecast.get('recipe') and
            controlled_recipe_key(current_forecast['recipe'])!=controlled_recipe_key(sample))
        if stale:st.info('The saved SAC-Gen forecast belongs to earlier controls. Save and predict again for current values.')
        else:
            st.write('**SAC-Gen forecast for this recipe**')
            st.dataframe(pd.DataFrame([{'metric': label, 'value': str(current_forecast['sac_gen_prediction'].get(key, 'Unavailable'))}
                for key,label in [('p_xrd_pass','XRD PASS probability'),('p_xrd_fail','XRD FAIL probability'),
                    ('predicted_total_wt_percent','Predicted total ICP / wt%'),('outside_training_support','Outside training support')]]),
                hide_index=True,width='stretch')
    st.write('**Mixing descriptors for the saved pending recipe**')
    st.dataframe(pd.DataFrame([forecast_metadata(sample, current_forecast.get('model_info'), probability)]),
                 hide_index=True, width='stretch')
    prediction = sample.get('manual_prediction')
    if prediction:
        st.caption('Human-adjusted recipe forecast' + (' (stale; save changes and predict again).' if prediction.get('prediction_stale') else '. Saved before result entry; separate from the original recommendation.'))
    if st.button('Predict saved pending parameters', disabled=has_observations(sample) or bool(sample.get('result_reported_at')),
                 key=f'pending_predict_{store.origin}_{ident}'):
        try:
            from .pending_prediction import refresh_pending_prediction
            with st.spinner('Refitting from eligible saved observations and predicting the saved recipe...'):
                refresh_pending_prediction(store, ident, fingerprint(sample))
            st.session_state[f'manual_notice_pending_{store.origin}'] = 'Saved a new model forecast for the manually adjusted recipe. Original proposal history is retained.'
            st.rerun()
        except (ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
    st.caption('Save table or individual edits first, then predict. This predicts the saved recipe without replacing it or reserving another experiment. Insufficient training data produces no invented probability.')
    render_sample_editor(store, sample, methods, "pending")


def current_pending_probability(sample):
    prediction = sample.get('manual_prediction') or sample.get('proposal') or {}
    return None if prediction.get('prediction_stale') else prediction.get('p_screen_pass')


def render_feedback(store, samples, methods):
    origin = store.origin
    notice("feedback", origin)
    st.subheader("Result feedback")
    st.caption('Edit the table and save. Unmeasured results remain pending; amended recipes require a new forecast.')
    if not samples:
        st.info("Register or add pending samples before returning results.")
        return
    scope = st.radio("Feedback sample scope", ["Pending experiments", "All experiment records"], horizontal=True, key=f"feedback_scope_{origin}")
    pool = pending_samples(samples, store.all("recommendation"), methods) if scope == "Pending experiments" else samples
    if not pool:
        st.info("No pending experiments. Select All experiment records to correct an earlier return.")
        return
    filter_key=f'feedback_filters_{origin}_{scope}'
    batch_options=sorted({s['batch_id'] for s in pool})
    if filter_key in st.session_state:
        st.session_state[filter_key]=[b for b in st.session_state[filter_key] if b in batch_options]
    batch_filters=st.multiselect('Recorded batches (empty = all)',batch_options,key=filter_key)
    batch_filter=';'.join(batch_filters) if batch_filters else 'All batches'
    pool=[s for s in pool if not batch_filters or s['batch_id'] in batch_filters]
    selection_key = f"feedback_samples_{origin}_{scope}_{batch_filter}"
    if st.button("Select all samples in this view", key=f"feedback_select_all_{origin}_{scope}_{batch_filter}"):
        st.session_state[selection_key] = [s["sample_id"] for s in pool]
    valid_ids = {s["sample_id"] for s in pool}
    if selection_key in st.session_state:
        retained = [ident for ident in st.session_state[selection_key] if ident in valid_ids]
        if retained != st.session_state[selection_key]:
            st.session_state[selection_key] = retained
    ids = st.multiselect("Samples to return", [s["sample_id"] for s in pool], default=None if selection_key in st.session_state else [pool[0]["sample_id"]], key=selection_key)
    selected = [s for s in pool if s["sample_id"] in ids]
    if not selected:
        st.info("Select at least one sample for this return.")
        return
    signature = fingerprint(selected)
    with st.form(f"feedback_form_{origin}_{signature}"):
        shared_batch = st.text_input("Actual batch ID for this return (optional)", help="Overrides the batch_id column for every selected sample when filled.", key=f"feedback_batch_{origin}_{signature}")
        rows, assays = [], []
        for s in selected:
            state = s.get("icp_review", {}).get("state", "PENDING")
            rows.append({"sample_id": s["sample_id"], "batch_id": s["batch_id"], "pulse_count": s["pulse_count"], "pulse_limit_C": s.get("pulse_limit_C"),
                "temperature_C": s.get("temperature_C"), "voltage_V": s.get("voltage_V"), "current_A": s.get("current_A"),
                "sample_condition": s.get("sample_condition") or "NOT_RECORDED", "icp_state": state if state != "AUTO" else "Keep numeric criterion", "xrd_state": s["xrd"].get("state", "PENDING"),
                "stem_state": s.get("stem", {}).get("state", "NOT_TESTED"), "notes": s.get("notes", "")})
            assays.extend({"sample_id": s["sample_id"], **r} for r in element_rows(s))
        outcomes = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", disabled=["sample_id"],
            column_order=["sample_id", "sample_condition", "icp_state", "xrd_state", "stem_state", "batch_id", "pulse_count", "pulse_limit_C", "temperature_C", "voltage_V", "current_A", "notes"],
            column_config={"sample_condition": st.column_config.SelectboxColumn("Sample condition", options=list(SAMPLE_CONDITIONS)), "icp_state": st.column_config.SelectboxColumn("ICP judgment", options=["PENDING", "PASS", "FAIL", "Keep numeric criterion"]),
                "xrd_state": st.column_config.SelectboxColumn("XRD judgment", options=["PENDING", "PASS", "FAIL"]),
                "stem_state": st.column_config.SelectboxColumn("Optional STEM", options=["NOT_TESTED", "CONFIRMED", "NOT_CONFIRMED", "INCONCLUSIVE"]),
                "pulse_count": st.column_config.NumberColumn("Pulses", min_value=1, step=1),
                "pulse_limit_C": st.column_config.NumberColumn("Controlled cutoff / C"),
                "temperature_C": st.column_config.NumberColumn("Measured peak / C")}, key=f"feedback_outcomes_{origin}_{signature}")
        st.write("**Element concentrations and returned ICP content**")
        st.caption("W is mass percent: 0.25 means 0.25 wt%. Numerical ICP, flags, methods and sources are optional when you select PASS/FAIL. Precursor concentration is recorded separately in mM.")
        details = st.data_editor(pd.DataFrame(assays), disabled=["sample_id", "element", "unit"], hide_index=True, width="stretch",
            column_config={"value": st.column_config.TextColumn("W / wt%"), "precursor_mM": st.column_config.NumberColumn("Precursor / mM", min_value=.000001, format="%.6f"),
                "flag": st.column_config.SelectboxColumn(options=["", "MEASURED", "ND", "BLQ", "MISSING"]), **chemistry_columns()}, key=f"feedback_icp_{origin}_{signature}")
        stem_source = st.text_input("Shared STEM source (optional)", key=f"feedback_stem_source_{origin}_{signature}")
        stem_reviewer = st.text_input("Shared STEM reviewer (optional)", key=f"feedback_stem_reviewer_{origin}_{signature}")
        confirm_catalog = st.checkbox('Confirm catalogue precursor identities for this returned batch', value=False,
            help='Optional. Check exact compounds/hydrates before confirming. This does not confirm solution coordination or final catalyst oxidation states.', key=f'feedback_confirm_catalog_{origin}_{signature}')
        with st.expander("Shared process metadata for this return"):
            st.caption("Filled values apply to every selected sample; blanks keep existing metadata. Known precursor, support and protocol are required for model eligibility, not for saving.")
            precursor = st.text_input("Shared precursor system (optional)", key=f"feedback_precursor_{origin}_{signature}")
            support = st.text_input("Shared support system (optional)", key=f"feedback_support_{origin}_{signature}")
            protocol = st.text_input("Shared protocol (optional)", key=f"feedback_protocol_{origin}_{signature}")
        submitted = st.form_submit_button("Save returned results", type="primary")
    if submitted:
        try:
            originals = {s["sample_id"]: s for s in selected}
            detail_rows = clean_cells(details.to_dict("records"))
            updated = []
            for row in clean_cells(outcomes.to_dict("records")):
                s = copy.deepcopy(originals[row["sample_id"]])
                s.update({k: row[k] for k in ("batch_id", "pulse_count", "pulse_limit_C", "temperature_C", "voltage_V", "current_A", "notes", "sample_condition")})
                if shared_batch.strip():
                    s["batch_id"] = shared_batch.strip()
                for key, value in (("precursor_family", precursor), ("support_batch", support), ("protocol_id", protocol)):
                    if value.strip():
                        s[key] = value.strip()
                s = apply_element_rows(s, [a for a in detail_rows if a["sample_id"] == s["sample_id"]])
                if confirm_catalog:
                    s = confirm_catalog_precursors(s)
                if row["icp_state"] != "Keep numeric criterion":
                    s["icp_review"] = {**s.get("icp_review", {}), "state": row["icp_state"], "source": "Manual batch result feedback"}
                s["xrd"] = {**s["xrd"], "state": row["xrd_state"], "mode": "MANUAL"}
                s["stem"] = {**s.get("stem", {}), "state": row["stem_state"]}
                if stem_source.strip():
                    s["stem"]["source"] = stem_source.strip()
                if stem_reviewer.strip():
                    s["stem"]["reviewer"] = stem_reviewer.strip()
                updated.append(s)
            result = save_manual_revisions(store, updated, methods, "feedback", {s["sample_id"]: fingerprint(s) for s in selected})
            saved(store, result, "feedback")
        except (ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))
