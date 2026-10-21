from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from sacfactory.bootstrap import seed
from sacfactory.dialogue import reply, diagnose, execute_intent, Answer
from sacfactory.language import CompatibleAdapter, LanguageError, shared_context
from sacfactory.providers import MIT_PARLEY_BASE, environment_config, setup_candidate, list_available_models, probe_chat_connection, same_base
from sacfactory.research_chat import ResearchAdapter, ResearchTools, research_context
from sacfactory.domain import TARGETS, RULE_VERSION, STATUS_NAMES, cohort_key, base_concentration, finite_number, parse_elements, recipe, screen, summary
from sacfactory.importers import csv_bytes, recommendation_workbook, SAMPLE_COLUMNS, ICP_COLUMNS
from sacfactory.planning import fit_model, recommend, recommendation_table, fingerprint, validate_space, concentration_bounds, ModelState
from sacfactory.storage import Store
from sacfactory.operating_space import restore_planning_space
from sacfactory.ingestion import preview_import, commit_import
from sacfactory.ui_batch import render_batch_panel, periodic_selector
from sacfactory.batching import compact_reporting_zip
from sacfactory.ui_records import render_records
from sacfactory.ui_analysis import render_analysis
from sacfactory.ui_manual import render_pending, render_feedback
from sacfactory.ui_chat import render_data_selection
from sacfactory.ui_materials import render_materials, planning_precursors
from sacfactory.precursors import PROFILES, normalize_precursors
from sacfactory.precursor_catalog import register_defaults
from sacfactory.ui_chemistry import render_precursor_editor
from sacfactory.ui_elements import element_selectbox, element_text_area
from sacfactory import __version__

st.set_page_config(page_title="SACFactory · Experiment decision workbench", page_icon="⚛", layout="wide", initial_sidebar_state="auto")
if "page_jump" in st.session_state:
    st.session_state["page"] = st.session_state.pop("page_jump")
st.markdown("""<style>
:root {--ink:#12283d;--teal:#097f7a;}
.stApp {background:#f5f7fa;}
.block-container {padding-top:3.5rem; max-width:1500px;}
h1,h2,h3 {color:var(--ink);letter-spacing:-.025em;}
[data-testid="stSidebar"] {background:#102b40;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] small {color:#edf7fb;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#bcd1df;}
[data-testid="stMetric"] {background:#fff;border:1px solid #e0e7ee;border-radius:12px;padding:14px 18px;}
[data-testid="stChatMessage"] {background:#fff;border:1px solid #e1e8ee;border-radius:12px;margin:8px 0;}
div[class*="st-key-matrix_scroll_"] {max-width:100%; overflow-x:auto;}
div[class*="st-key-matrix_scroll_"] [data-testid="stVegaLiteChart"] {overflow-x:auto;}
.eyebrow {color:#097f7a;font-size:12px;font-weight:700;letter-spacing:.15em;}
.hero {display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:15px;}
.hero h1 {font-size:31px;margin:0;padding:0;}
.hero p {color:#63788a;margin:6px 0 0;font-size:14px;}
.badge {background:#e4f4ef;color:#06776d;border:1px solid #bbe1d5;padding:7px 13px;border-radius:30px;font-size:12px;white-space:nowrap;}
.note {border-left:3px solid #10a29b;padding:8px 13px;background:#eaf5f3;color:#234b52;font-size:14px;border-radius:0 5px 5px 0;margin:15px 0;}
div[class*="st-key-periodic_grid_"] {gap:4px;}
div[class*="st-key-periodic_grid_"] button {padding:2px;min-width:0;min-height:32px;}
div[class*="st-key-periodic_grid_"] [data-testid="stHorizontalBlock"] {gap:2px !important;}
@media (max-width:760px) {
  .block-container {padding-left:1rem;padding-right:1rem;}
  .hero h1 {font-size:25px;}
  [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
  [data-testid="stColumn"] {min-width:min(100%,280px);flex:1 1 280px;}
  [data-testid="stMetric"] {padding:10px 14px;}
  div[class*="st-key-periodic_grid_"] [data-testid="stColumn"],
  div[class*="st-key-periodic_controls_"] [data-testid="stColumn"],
  div[class*="st-key-periodic_actions_"] [data-testid="stColumn"] {min-width:0;flex:1 1 0;}
  div[class*="st-key-periodic_grid_"] [data-testid="stHorizontalBlock"],
  div[class*="st-key-periodic_controls_"] [data-testid="stHorizontalBlock"],
  div[class*="st-key-periodic_actions_"] [data-testid="stHorizontalBlock"] {flex-wrap:nowrap;}
  div[class*="st-key-periodic_grid_"] button p {font-size:12px;white-space:nowrap;}
}
</style>""", unsafe_allow_html=True)

if st.session_state.get("page") == "Analytical methods":
    st.session_state["page"] = "Data analysis"

with st.sidebar:
    st.markdown("## ⚛ SACFactory")
    st.caption("INTELLIGENT SYNTHESIS WORKBENCH")
    origin_label = st.radio("Data space", ["Real experiments · REAL", "Simulated demo · DEMO"], key="mode")
    origin = "REAL" if "REAL" in origin_label else "DEMO"
    page = st.radio("Workspace", ["Experiment chat", "Samples & data", "Materials & consumables", "Next batch", "Data analysis", "Language model", "About"], key="page")
    st.divider()
    api_active = bool(st.session_state.get("api_config") and st.session_state.get(f"api_permission_{origin}", False))
    research_mode = st.session_state.get('assistant_mode_setting', 'Task interpretation') == 'Research conversation'
    st.caption("Objective: candidate screening success\n\nChat: " + ("Research conversation + local tools" if api_active and research_mode else "API intent parser + local engine" if api_active else "Local task assistant"))
    st.caption("Controlled cutoff temperature and measured peak are separate. Pulse duration is not controlled.")

store = Store(os.environ.get("SACFACTORY_DATA_DIR", str(ROOT / "data")), origin)
seed(store, ROOT)
register_defaults(store)
epoch = (store.get("meta", "data-epoch") or {}).get("value", "initial")
epoch_key = f"seen_data_epoch_{origin}"
if st.session_state.get(epoch_key) != epoch:
    for key in list(st.session_state):
        if origin in key and not key.startswith(("api_", "seen_data_epoch_", "record_action_")):
            del st.session_state[key]
    st.session_state[epoch_key] = epoch
samples = store.all("sample")
methods = {m["method_id"]: m for m in store.all("method")}
reservations = store.all("recommendation")
context_record = store.get("meta", "plan_context")
if context_record:
    context = context_record["context"]
elif samples:
    source = next((s for s in samples if origin == "DEMO" and len(s["elements"]) == 50), samples[0])
    context = copy.deepcopy(source)
else:
    context = {"sample_id": "EMPTY_CONTEXT", "origin": origin, "elements": list(TARGETS),
        "concentrations_mM": recipe(TARGETS), "pulse_count": 1, "temperature_C": None, "pulse_limit_C": None,
        "precursor_family": "UNKNOWN", "support_batch": "UNKNOWN", "protocol_id": "UNKNOWN", "icp": []}
context = {**context, 'precursors': planning_precursors(store, context, context['elements'])}
space_record = store.get("meta", "space")
from sacfactory.element_campaign import campaign_training_samples
campaign_policy = store.get('meta', 'element_campaign') or {}
campaign_enabled = bool(campaign_policy.get('enabled'))
space = restore_planning_space(space_record['space'] if space_record else None,
    context, samples, methods, origin, campaign_policy)
if campaign_enabled:
    covered = space['_single_element_campaign']['covered']


def concentration_text(values):
    return "; ".join(f"{e}: {v:.6g}" if v is not None else f"{e}: pending" for e, v in values.items())


def display_history_table(rows):
    # Older chat snapshots retain their audit data; display actual stored concentrations.
    out = []
    for row in rows:
        values = json.loads(row.get("concentrations_mM_json", "{}"))
        out.append({**row, "concentration_mM": row.get("concentration_mM") or concentration_text(values) or "Not recorded"})
    return pd.DataFrame(out)


@st.cache_resource(show_spinner='Learning the selected model and checking independent batches...', max_entries=8)
def train_cached(serialized):
    payload = json.loads(serialized)
    return fit_model(**payload)


payload = {"samples": campaign_training_samples(samples, campaign_enabled, context), "methods": methods, "context": context, "origin": origin, "transfer": space.get("element_transfer", True),
           'engine': space.get('model_engine', 'adaptive'), 'settings': {'learning_iterations': space.get('learning_iterations', 18), 'batch_effect': space.get('batch_effect', False),
               'descriptor_profile': space.get('descriptor_profile', 'chemistry_pulseicp') if space.get('model_engine', 'adaptive') != 'baseline' else 'periodic'}}
planning_signature=fingerprint(payload)
last_state=st.session_state.get(f'planning_state_{origin}')
active_sac_pointer={}
active_sac_path=store.root/'sac_gen'/origin.lower()/'latest.json'
if active_sac_path.exists():
    try:active_sac_pointer=json.loads(active_sac_path.read_text())
    except (ValueError,OSError):pass
if page=='Next batch' and active_sac_pointer and st.session_state.get(f'next_sac_gen_{origin}',True):
    state=ModelState(None,{'engine':'sac_gen','mode':'SAC_GEN_DEPLOYMENT','model_id':active_sac_pointer['model_id'],
        'n_labeled':active_sac_pointer.get('training_samples',0),'n_batches':len({s.get('batch_id') for s in samples}),
        'reason':'Trained SAC-Gen checkpoint available. XRD probability and continuous ICP forecasts are evaluated when generating a batch. Training-record changes require retraining.'},[])
    st.session_state[f'planning_state_{origin}']=(planning_signature,state)
elif page=='Next batch':
    state = train_cached(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    st.session_state[f'planning_state_{origin}']=(planning_signature,state)
elif last_state and last_state[0]==planning_signature:
    state=last_state[1]
else:
    # Analysis performs its own explicitly triggered, protected fits. Opening
    # records/settings must not trigger a large campaign model refit.
    state = ModelState(None, {'deferred': True, 'mode':'DEFERRED', 'model_id':'DEFERRED', 'n_labeled': 0, 'n_batches': 0,
        'reason': 'Planning model fitting is deferred until a chat request or Next batch.'}, [])
preview = st.session_state.get(f'preview_{origin}', [])
if preview:
    signature = fingerprint({'context': context, 'space': space, 'samples': samples, 'methods': methods, 'rule_version': RULE_VERSION})
    reserved_ids = {r['recommendation_id'] for r in reservations}
    if not all(r['recommendation_id'] in reserved_ids for r in preview) and any(
            r.get('snapshot') != signature or (not state.info.get('deferred') and r.get('model_id') !=
                (active_sac_pointer.get('model_id') if r.get('policy')=='SAC_GEN_XRD_ACTIVE_LEARNING' else state.info.get('model_id','NO_MODEL'))) for r in preview):
        st.session_state.pop(f'preview_{origin}', None)
        st.session_state.pop(f'plan_note_{origin}', None)
        st.session_state[f'preview_refresh_notice_{origin}'] = 'Earlier unreserved preview cleared because data, chemistry, settings or model version changed. Generate a new preview with the current model.'
stats = summary(samples, methods)
failed = sum(stats["counts"].get(k, 0) for k in ("ICP_FAIL", "XRD_FAIL", "BOTH_FAIL", "BURNED_OUT"))
pending = stats["counts"].get("PENDING", 0) + stats["counts"].get("RETEST", 0)
compact_page = page in {'Next batch', 'Materials & consumables'}
if compact_page:
    st.caption(f"SACFactory · {origin} · {stats['total']} recorded samples")
summary_container = st.expander('Workspace summary and screening criteria') if compact_page else st.container()
with summary_container:
    st.markdown(f'<div class="hero"><div><div class="eyebrow">SACFACTORY / LOCAL RESEARCH PLATFORM</div><h1>Plan the next experiment with evidence</h1><p>Record → Screen → Learn → Recommend → Report</p></div><div class="badge">{"Real experimental data" if origin == "REAL" else "Simulated demo · Not experimental"}</div></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for c, label, value in ((c1, "Samples", stats["total"]), (c2, "Screen pass", stats["counts"].get("PASS", 0)), (c3, "Screen fail", failed), (c4, "Evidence pending", pending)):
        c.metric(label, value)
    st.markdown('<div class="note">Candidate screening uses ICP PASS + XRD PASS. Record complete burnout separately as a synthesis failure; handling/device exceptions require retesting. Unperformed tests remain pending. STEM is separate. Screening does not independently establish atomic dispersion.</div>', unsafe_allow_html=True)
if origin == "DEMO":
    st.warning("DEMO contains explicitly simulated synthesis outcomes, optional content readings and saved forward predictions. Assumed response functions illustrate the workflow, not experimental performance. DEMO data and training are isolated from REAL.")


def render_recommendations(rows, key):
    if not rows:
        return
    table = recommendation_table(rows)
    sac_gen_rows=any(r.get('sac_gen_prediction') for r in rows)
    if sac_gen_rows:
        st.write('**SAC-Gen: trained loading and XRD forecasts**')
        st.dataframe(pd.DataFrame(table)[['target_elements','p_xrd_pass','p_xrd_fail','predicted_total_wt_percent',
            'concentration_mM','pulse_count','pulse_limit_C','extrapolation']],hide_index=True,width='stretch')
        st.caption('XRD probabilities are trained forecasts, not measured atomic dispersion. ICP is predicted as W / wt%, without a pass classifier. Correctly predicted XRD FAIL experiments count as accurate model predictions.')
    if not sac_gen_rows:st.dataframe(pd.DataFrame(table)[["target_elements", "p_screen_pass", "concentration_mM", "precursor_compounds", "pulse_count", "pulse_limit_C", "total_concentration_mM", "purpose", "extrapolation"]], hide_index=True, width="stretch",
        column_config={"target_elements": st.column_config.TextColumn("Composition", width="small"),
                       "p_screen_pass": st.column_config.NumberColumn("Predicted p", format="%.3f", width="small", help="Uncalibrated candidate screening estimate, not a measurement."),
                       "concentration_mM": st.column_config.TextColumn("Precursor / mM", width="medium"),
                       'precursor_compounds': st.column_config.TextColumn('Declared precursors', width='medium'),
                       "pulse_count": st.column_config.NumberColumn("Pulses", width="small")})
    if not sac_gen_rows:st.caption('Predicted screening probabilities; extrapolated conditions are flagged.')
    st.write('**Mixing thermodynamics in these predictions**')
    mixing_columns = ['target_elements', 'p_xrd_pass' if sac_gen_rows else 'p_screen_pass', 'h_mix_kJ_mol_atoms', 's_mix_J_mol_K',
        'g_mix_proxy_kJ_mol_atoms', 'mix_pair_weight_coverage', 'mixing_descriptors_used_by_model', 'mix_input_status']
    st.dataframe(pd.DataFrame(table)[mixing_columns], hide_index=True, width='stretch')
    st.caption('Mixing descriptors use feed molar fractions; missing binary parameters remain missing.')
    with st.expander('Element loading predictions'):
        if sac_gen_rows:
            values=[{'recommendation_id':r['recommendation_id'],**v} for r in rows for v in r.get('sac_gen_prediction',{}).get('icp_predictions',[])]
            st.dataframe(pd.DataFrame(values),hide_index=True,width='stretch')
        elif st.button('Calculate ICP forecasts for this batch', key=f'picp_forecast_{key}'):
            from sacfactory.pulseicp_bridge import forecast, identity
            try:
                recipes = [{**r['recipe'], 'sample_id': r['recommendation_id']} for r in rows]
                forecasts = forecast(recipes, origin, store.root, current_samples=samples)
                for r in rows:
                    part = forecasts.loc[forecasts.sample_id == r['recommendation_id']]
                    r['icp_prediction'] = {'values':part.astype(object).where(pd.notna(part), None).to_dict('records'), 'snapshot':identity(), 'basis':'PRE_SYNTHESIS_RECIPE_FORECAST'}
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                st.error(str(exc))
        values = [v for r in rows for v in r.get('icp_prediction', {}).get('values', [])]
        if values:
            st.dataframe(pd.DataFrame(values), hide_index=True, width='stretch')
    if any(r.get('policy') in {'ADAPTIVE_JOINT_INFORMATION', 'RELIABILITY_INFORMATION_AND_REPEAT_BUDGET'} for r in rows):
        st.write('**Adaptive prediction and batch information**')
        st.dataframe(pd.DataFrame(table)[['p_screen_pass', 'p_screen_low', 'p_screen_high', 'bald_bits', 'joint_information_bits', 'model_disagreement', 'calibration_state', 'information_weight']], hide_index=True, width='stretch')
        st.caption('Intervals describe model uncertainty; they are not structural confirmation.')
    if any(r.get('repeat_rationale') for r in rows):
        st.write('**Repeat allocation and stopping assumptions**')
        st.dataframe(pd.DataFrame([{k: row.get(k) for k in ('recommendation_id', 'is_repeat', 'repeat_strategy', 'planned_additional_repeats', 'further_repeats_after_this')} for row in table if row.get('is_repeat')]), hide_index=True, width='stretch')
    if any(r.get('robustness') for r in rows):
        pass
    if any(r.get("prediction_mode") in {"TRANSFER_HYPOTHESIS", "TARGET_AND_TRANSFER"} for r in rows):
        st.caption("Transfer outputs share member information and retain pair/composition residuals. Source-only mixtures are unvalidated hypotheses. Source affinity is kernel similarity; an unseen target remains a transfer hypothesis.")
        st.dataframe(pd.DataFrame(table)[["prediction_mode", "target_labeled_n", "source_compositions", "source_sample_ids", "source_affinity", "support_level", "expansion_step", "unobserved_pair_count", "predictive_entropy"]], hide_index=True, width="stretch")
    linked = [r for r in store.all("recommendation") if r["recommendation_id"] in {row["recommendation_id"] for row in rows} and r.get("pending_sample_id")]
    if st.button("Add batch to pending", disabled=len(linked) == len(rows), key=f"pending_{key}", type="primary"):
        try:
            ids = store.enqueue_pending(rows, context, space)
            st.session_state[f"pending_added_{origin}"] = ids
            st.session_state.pop(f"entry_cells_{origin}", None)
            st.session_state.pop(f"entry_preview_data_{origin}", None)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if linked:
        st.success("Pending samples: " + ", ".join(r["pending_sample_id"] for r in linked) + ". Edit plans under Pending experiments; return measurements under Result feedback.")
    st.caption("One click saves the full preview with automatic SAC IDs, pending ICP/XRD and separate prediction metadata. Measured peak, voltage/current and actual synthesis batch remain unrecorded.")
    a, b, c = st.columns([1, 1, 2])
    a.download_button("Export batch CSV", csv_bytes(table), file_name=f"{origin.lower()}_next_batch.csv", mime="text/csv", key=f"download_{key}")
    a.download_button("Results workbook", recommendation_workbook(rows), file_name=f"{origin.lower()}_batch_results.xlsx", key=f"workbook_{key}")
    a.download_button("Compact reporting tables", compact_reporting_zip(rows), file_name=f"{origin.lower()}_compact_batch.zip", mime="application/zip", key=f"compact_{key}", help="Unzip, fill batch/ICP/settings CSVs in Excel, then upload all three together. Actual peak temperatures and assay results are blank.")
    existing_ids = {r["recommendation_id"] for r in reservations}
    new = [r for r in rows if r["recommendation_id"] not in existing_ids]
    if b.button("Reserve batch", disabled=not new, key=f"reserve_{key}"):
        # Recommendations are snapshots: reject if inputs changed since preview.
        current = fingerprint({"context": context, "space": space, "samples": samples, "methods": methods, "rule_version": RULE_VERSION})
        if any(r.get("snapshot") != current for r in new):
            st.error("Data or settings changed. Generate a new preview before reserving.")
        else:
            current_reserved = store.all("recommendation")
            from sacfactory.domain import recipe_key
            open_keys = {recipe_key(r["recipe"]) for r in current_reserved if r["state"] == "OPEN"}
            if any(recipe_key(r["recipe"]) in open_keys for r in new):
                st.error("An identical recipe is already reserved. Refresh and generate a new batch.")
            else:
                store.put_many("recommendation", new, "recommendation_id")
                st.rerun()
    with c.expander("Inspect element-specific precursor concentrations"):
        selected = st.selectbox("Proposed recipe", range(len(rows)), format_func=lambda i: rows[i]["recommendation_id"], key=f"detail_{key}")
        st.dataframe(pd.DataFrame([{"Element": e, "Concentration / mM": v} for e, v in rows[selected]["recipe"]["concentrations_mM"].items()]), hide_index=True, width="stretch")


def snapshot_rows(rows):
    sig = fingerprint({"context": context, "space": space, "samples": samples, "methods": methods, "rule_version": RULE_VERSION})
    for r in rows:
        r["snapshot"] = sig
    return rows


if page == "Experiment chat":
    left, right = st.columns([2.25, 1], gap="large")
    with right:
        st.subheader("Current context")
        st.write(f"**{len(context['elements'])} target elements**")
        st.caption(f"Controlled cutoff: {context.get('pulse_limit_C') or 'Not recorded'} °C. Recorded temperature: {context.get('temperature_C') or 'Not recorded'} °C.")
        st.caption(" · ".join(context["elements"]))
        entered = context["concentrations_mM"]
        if all(v is not None for v in entered.values()):
            st.write(f"Reference total concentration **{sum(entered.values()):.3g} mM**")
        else:
            st.caption("Reference concentrations are incomplete; enter them on Next batch.")
        st.caption(f"Precursor: {context['precursor_family']}\n\nSupport: {context['support_batch']}\n\nProtocol: {context['protocol_id']}")
        st.divider()
        st.write("**Recommendation engine**")
        st.write('Planning model fits on demand' if state.info.get('deferred') else "Classification active learning" if state.model is not None else "Initial experimental design")
        if not state.info.get('deferred'):
            st.caption(f"Training pool: {state.info['n_labeled']} valid records / {state.info['n_batches']} batches")
        if state.model is None:
            st.caption("Enter ICP/XRD judgments directly or supplement numeric evidence. Collect independent recipes with complete synthesis conditions for model training." if origin == "REAL" and len(samples) == 1 else state.info["reason"])
        if st.button("Configure composition and range", width="stretch"):
            st.session_state["page_jump"] = "Next batch"
            st.rerun()
        with st.expander("Local dialogue capabilities"):
            st.write("Overview, missing evidence, sample IDs, batch size, only N pulses, model explanation and composition expansion. Configure an optional provider on Language model for free-form interpretation. Quick shortcuts always run locally.")
        if api_active:
            st.caption("Research conversation uses selected recorded data, recent conversation and local tools. Enable sharing on Language model." if research_mode else "Typed requests use the enabled API. Scientific outputs are computed locally.")
        if research_mode:
            st.divider()
            st.write('**Conversation data**')
            render_data_selection(samples, origin, context)
            if st.button('Start new research discussion', key=f'research_new_{origin}'):
                from uuid import uuid4
                st.session_state[f'research_discussion_{origin}'] = uuid4().hex
                st.success('New research discussion started. Earlier chat remains in local history but is excluded from the new API context.')
    with left:
        st.subheader('Research conversation' if research_mode else 'Experiment assistant')
        if st.button('Clear chat history', key=f'clear_chat_{origin}'):
            store.clear_messages()
            from uuid import uuid4
            st.session_state[f'research_discussion_{origin}'] = uuid4().hex
            st.rerun()
        research_ready = api_active and research_mode and st.session_state.get(f'research_permission_{origin}', False)
        if research_ready:
            shared_count = len(st.session_state.get(f'research_samples_{origin}', []))
            st.caption(f'{origin} · {shared_count} selected records · Research conversation and local calculations enabled')
        elif research_mode:
            st.info('Configure the provider and enable API/data sharing on Language model to discuss your experiments. Local shortcuts remain available.')
        else:
            st.caption('Local shortcuts calculate summaries, check recorded evidence and preview the next batch. Enable Research conversation for scientific discussion.')
        if not research_ready and st.button('Open language model settings', key='chat_open_settings'):
            st.session_state['page_jump'] = 'Language model'
            st.rerun()
        research_prompt = None
        if research_mode:
            st.write('**Start a research discussion**')
            prompt_cols = st.columns(3)
            for col, label, prompt in (
                    (prompt_cols[0], 'Explain failures', 'Analyze failures in my shared experiments. Use local calculations, distinguish observed evidence from possible causes, and suggest discriminating experiments.'),
                    (prompt_cols[1], 'Compare synthesis settings', 'Compare concentration, pulse count and controlled cutoff temperature in my shared selection. Report sample and batch counts, and explain confounding and missing data.'),
                    (prompt_cols[2], 'Plan element expansion', 'Discuss how the shared single-element data could guide the next element or a multi-element synthesis. Use the registered composition and range for any preview, and distinguish validated evidence from transfer hypotheses.')):
                if col.button(label, disabled=not research_ready, width='stretch'):
                    research_prompt = prompt
        st.caption('Local shortcuts · no API request')
        quick = st.columns(3)
        question = None
        for col, label, prompt in ((quick[0], "Summary", "Summarize experimental results"), (quick[1], "Check evidence", "Check missing evidence"), (quick[2], "Next batch", "Recommend 6 experiments")):
            if col.button(label, width="stretch"):
                question = prompt
        chat = st.container(height=520, border=True)
        with chat:
            messages = store.messages(30)
            if not messages:
                st.caption('No conversation yet. Enter a message below or use an available action.')
            for message in messages:
                with st.chat_message(message["role"], avatar="⚛" if message["role"] == "assistant" else None):
                    st.markdown(message["content"])
                    if message["payload"].get("numbering_note"):
                        st.caption(message["payload"]["numbering_note"])
                    research = message['payload'].get('language', {})
                    if research.get('mode') == 'API_RESEARCH':
                        counts = research.get('usage', {})
                        input_count = counts.get('input_tokens') if counts.get('input_tokens') is not None else 'Unavailable'
                        output_count = counts.get('output_tokens') if counts.get('output_tokens') is not None else 'Unavailable'
                        st.caption(f"Research interpretation · {research.get('api_calls', 0)} API calls · Provider-reported input/output tokens: {input_count} / {output_count}")
                        if research.get('tools'):
                            pass
                    parsed = message["payload"].get("language", {}).get("intent")
                    if parsed:
                        details = [f"Interpreted action: {parsed['intent']}"]
                        if parsed.get("sample_id"): details.append(f"sample {parsed['sample_id']}")
                        if parsed.get("batch_size"): details.append(f"requested batch size {parsed['batch_size']}")
                        if parsed.get("pulse_count_filter"): details.append(f"pulses {parsed['pulse_count_filter']}")
                        if parsed.get("topic"): details.append(f"topic {parsed['topic']}")
                        st.caption(" · ".join(details) + " · Results calculated locally")
                    if message["payload"].get("table"):
                        st.dataframe(display_history_table(message["payload"]["table"])[["concentration_mM", "pulse_count", "p_screen_pass", "purpose"]], hide_index=True, width="stretch")
        typed = st.chat_input("Ask about failures, compare batches, or discuss the next synthesis" if research_mode else "Try: Diagnose SAC_0001; Recommend 6 experiments", key="experiment_chat") or research_prompt
        question = typed or question
        if question:
            config = st.session_state.get("api_config")
            pending_config = st.session_state.get('api_pending_config')
            known_keys = [c.api_key for c in (config, pending_config, environment_config()) if c and c.api_key]
            if any(key in question for key in known_keys):
                st.error("Do not put the API key in chat. Use the password field on Language model. This request was not saved or sent.")
            else:
                if state.info.get('deferred'):
                    state=train_cached(json.dumps(payload,ensure_ascii=False,sort_keys=True))
                    st.session_state[f'planning_state_{origin}']=(planning_signature,state)
                language_record = {"mode": "LOCAL"}
                if typed and api_active and research_mode:
                    selected_ids = st.session_state.get(f'research_samples_{origin}', [])
                    scope = fingerprint({'base': config.base_url, 'model': config.model, 'api_format': config.api_format, 'origin': origin, 'selection': sorted(selected_ids), 'discussion': st.session_state.get(f'research_discussion_{origin}', 'initial')})
                    history = [m for m in messages if m['payload'].get('language', {}).get('mode') == 'API_RESEARCH' and m['payload']['language'].get('scope') == scope]
                    try:
                        enabled = bool(st.session_state.get(f'research_permission_{origin}', False))
                        if not enabled:
                            raise LanguageError('Enable research-chat data and conversation sharing on Language model before sending a free-form discussion.')
                        shared = research_context(samples, methods, state, context, space, origin, selected_ids)
                        tools = ResearchTools(samples, methods, state, context, space, origin, selected_ids, reservations)
                        with st.spinner('Discussing results and running requested local calculations...'):
                            result = ResearchAdapter(config, enabled=True, max_output_tokens=st.session_state.get('research_token_budget', 3000)).converse(question, history, shared, tools)
                        answer = result.answer
                        language_record = {'mode': 'API_RESEARCH', 'scope': scope, 'model': config.model, 'usage': result.usage, 'api_calls': result.api_calls, 'tools': result.trace}
                    except LanguageError as exc:
                        answer = Answer(str(exc) + '\n\nLocal shortcuts remain available. No experiment was reserved.', [], [])
                        language_record = {'mode': 'API_ERROR'}
                elif typed and api_active:
                    shared = shared_context(samples, methods, state, context, space, origin)
                    scope = fingerprint({"origin": origin, "cohort": list(cohort_key(context)), "sample_id": context.get("sample_id")})
                    history = [m["payload"]["language"]["intent"] for m in messages
                               if m["payload"].get("language", {}).get("scope") == scope and "intent" in m["payload"].get("language", {})]
                    try:
                        with st.spinner("Interpreting request; scientific calculations stay local..."):
                            intent = CompatibleAdapter(config, enabled=True).interpret(question, history, shared)
                            answer = execute_intent(intent, samples, methods, state, context, space, origin, reservations)
                        language_record = {"mode": "API_INTENT", "intent": intent, "scope": scope, "model": config.model}
                    except LanguageError as exc:
                        answer = Answer(str(exc) + "\n\nUse the local shortcuts, or disable the API on Language model. No batch was reserved.", [], [])
                        language_record = {"mode": "API_ERROR"}
                else:
                    answer = reply(question, samples, methods, state, context, space, origin, reservations)
                snapshot_rows(answer.recommendations)
                store.message("user", question, {'language': {'mode': language_record['mode'], 'scope': language_record.get('scope')}})
                store.message("assistant", answer.text, {"table": answer.table, "recommendations": answer.recommendations, "context": context, "model": state.info, "language": language_record})
                if answer.recommendations:
                    st.session_state[f"preview_{origin}"] = answer.recommendations
                st.rerun()
        rows = st.session_state.get(f"preview_{origin}", [])
        if rows:
            st.write("**Most recent successful batch preview**")
            render_recommendations(rows, "chat")

elif page == "Samples & data":
    tab4, tab1, tab_pending, tab_feedback, tab2, tab3 = st.tabs(["Fill batch table", "Experiment records", "Pending experiments", "Result feedback", "Bulk import / report", "Register sample"])
    with tab4:
        render_batch_panel(store, context, methods, ROOT)
    with tab1:
        render_records(store, samples, methods, concentration_text)
    with tab_pending:
        render_pending(store, samples, methods)
    with tab_feedback:
        render_feedback(store, samples, methods)
    with tab2:
        st.write("Upload a compact **batch / icp / settings** workbook or its three CSV tables. The advanced **samples / icp / methods** format remains supported for legacy imports. Optional ICP values are recorded in W / wt%.")
        compact = ROOT / "examples/compact_batch_template.xlsx"
        if compact.exists(): st.download_button("Download compact batch template", compact.read_bytes(), "SACFactory_batch_template.xlsx")
        st.download_button("Download bulk Excel template", (ROOT / "examples/bulk_import_template.xlsx").read_bytes(), "SACFactory_bulk_template.xlsx")
        upload_nonce = st.session_state.get(f"upload_nonce_{origin}", 0)
        uploads = st.file_uploader("Standard Excel, or sample and ICP CSV files", type=["xlsx", "csv"], accept_multiple_files=True, key=f"bulk_files_{origin}_{upload_nonce}")
        overwrite = st.checkbox("Revise existing sample or method IDs (preserve history)", value=False, key=f"import_revision_{origin}")
        if uploads:
            try:
                preview = preview_import([(file.name, file.getvalue()) for file in uploads], store, overwrite)
                if preview.rows:
                    st.dataframe(pd.DataFrame(preview.rows), hide_index=True, width="stretch")
                if preview.method_changes:
                    st.write("**Analytical method changes**")
                    st.dataframe(pd.DataFrame(preview.method_changes), hide_index=True, width="stretch")
                if preview.queue_updates:
                    st.caption(f"{len(preview.queue_updates)} matching reservations will be marked completed and linked to reported samples.")
                st.caption("Revision replaces each complete sample. Start from an export and retain all ICP rows. Source files are archived.")
                if st.button("Commit import to local database", type="primary"):
                    result = commit_import(preview, store)
                    st.session_state[f"last_import_{origin}"] = result
                    st.session_state[f"upload_nonce_{origin}"] = upload_nonce + 1
                    st.rerun()
            except (ValueError, UnicodeError, KeyError) as exc:
                st.error(str(exc))
        if st.session_state.get(f"last_import_{origin}"):
            result = st.session_state[f"last_import_{origin}"]
            st.success(f"Last import: {result['samples']} samples, {result['methods']} method updates, {result['completed_tasks']} completed reservations.")
    with tab3:
        st.caption("Register recipes here; no measured values are generated. Report results using the standard template.")
        id_mode = st.radio("Sample ID numbering", ["Automatic", "Manual"], key=f"register_id_mode_{origin}", horizontal=True)
        elements_text = element_text_area("Target elements, separated by semicolons",
            value=";".join(context["elements"]), key=f'register_elements_{origin}')
        try:
            register_elements = parse_elements(elements_text)
        except ValueError as exc:
            register_elements = []
            st.error(str(exc))
        from sacfactory.ui_preparation import preparation_inputs
        from sacfactory.preparation import from_mM, to_mM
        register_unit, register_materials = preparation_inputs(f'register_preparation_{origin}')
        from sacfactory.experiment_defaults import shared_process_values
        registration_process = shared_process_values(context)
        with st.form("register_sample"):
            cols = st.columns(2)
            sample_id = cols[0].text_input("New sample ID", value=store.next_sample_ids()[0] if id_mode == "Automatic" else "", disabled=id_mode == "Automatic",
                key=f"new_sample_id_{origin}_{id_mode}_{store.next_sample_ids()[0] if id_mode == 'Automatic' else 'manual'}", placeholder="SAC_0001 or your own ID")
            batch_id = cols[1].text_input("Independent synthesis batch ID", placeholder="BATCH_001")
            st.caption("Enter precursor salt concentration or amount in the selected unit. Each element may have a different value; blanks remain pending.")
            entered_concentrations = st.data_editor(pd.DataFrame([{"element": e, "precursor_mM": from_mM(context["concentrations_mM"].get(e) or base_concentration(e), register_unit, register_materials['solvent_volume_mL'])} for e in register_elements], columns=["element", "precursor_mM"]),
                disabled=["element"], hide_index=True, width="stretch", column_config={"precursor_mM": st.column_config.NumberColumn("Precursor / " + register_unit, min_value=.000001, format="%.6f")}, key=f"register_concentrations_{origin}_{fingerprint([register_elements, register_unit, register_materials['solvent_volume_mL']])}")
            pulse = cols[1].number_input("Pulse count", min_value=1, value=1, step=1)
            cutoff = cols[0].text_input("Controlled cutoff temperature / °C", help="Enter the configured shutoff temperature; leave blank if unknown.")
            measured = cols[1].text_input("Measured peak temperature / °C", help="Leave blank until measured; do not copy a setpoint into this field.")
            voltage = cols[0].text_input("Voltage / V (optional)", value="32")
            current = cols[1].text_input("Current / A (optional)", value="80")
            precursor = st.text_input("Precursor system / recipe version", value=registration_process["precursor_family"])
            support = st.text_input("Support batch / fixed system", value=registration_process["support_batch"])
            protocol = st.text_input("Protocol version", value=registration_process["protocol_id"])
            with st.expander('Per-element precursor defaults'):
                registration_chemistry = render_precursor_editor(register_elements,
                    planning_precursors(store, context, register_elements),
                    f"register_chemistry_{origin}_{fingerprint(register_elements)}")
            if st.form_submit_button("Register pending sample"):
                try:
                    from sacfactory.domain import validate_sample
                    new = {"sample_id": sample_id, "batch_id": batch_id, "origin": origin, "elements": parse_elements(elements_text), "pulse_count": pulse, "temperature_C": measured, "pulse_limit_C": cutoff, "voltage_V": voltage, "current_A": current,
                           "precursor_family": precursor, "support_batch": support, "protocol_id": protocol, "xrd": {"state": "PENDING", "source": "", "criteria": "", "reviewer": ""}, "notes": "Registered through the UI; measurements pending", "icp": []}
                    new.update(register_materials, preparation_basis='User-declared preparation; protocol defaults if unchanged', concentration_input_unit=register_unit)
                    new['electrical_input_kind'] = 'SETTING'
                    new["concentrations_mM"] = {row["element"]: to_mM(None if pd.isna(row["precursor_mM"]) else row["precursor_mM"], register_unit, register_materials['solvent_volume_mL']) for row in entered_concentrations.to_dict("records")}
                    new['precursors'] = normalize_precursors(registration_chemistry, new['elements'])
                    store.put_many("sample", [validate_sample(new)], "sample_id")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

elif page == "Materials & consumables":
    render_materials(store, context, samples)

elif page == "Next batch":
    st.subheader("Next batch")
    st.caption("Target: " + ", ".join(context["elements"]))
    with st.expander("Change target elements"):
        chosen_elements = periodic_selector(f"plan_elements_{origin}", context["elements"])
    if state.model is None:
        st.info(state.info["reason"])
    count = st.number_input("Maximum experiments in the next batch", min_value=1, max_value=12, value=6, step=1)
    sac_gen_exists=(store.root/'sac_gen'/origin.lower()/'latest.json').exists()
    use_sac_gen=st.checkbox('Use trained SAC-Gen for XRD acquisition and ICP forecasts',value=sac_gen_exists,
        disabled=not sac_gen_exists,key=f'next_sac_gen_{origin}')
    sac_gen_strategy=st.selectbox('SAC-Gen proposal objective',['Balanced boundary learning',
        'High-loading XRD failure mapping','XRD pass region validation'],disabled=not use_sac_gen,key=f'sac_gen_strategy_{origin}')
    generate_saved = st.button("Generate batch preview", type="primary",
        help='Uses the saved operating range. After editing fields below, use Save range and generate batch.')
    generate_after_save = st.session_state.pop(f'generate_after_save_{origin}', False)
    if generate_saved or generate_after_save:
        try:
            with st.spinner('Evaluating the candidate pool and joint batch information...'):
                from sacfactory.sac_gen import latest as latest_sac_gen
                deployed=latest_sac_gen(store.root,origin,samples) if use_sac_gen else None
                result, note = recommend(state, samples, context,{**space,'sac_gen_strategy':sac_gen_strategy}, origin, count, reservations,sac_gen=deployed)
            st.session_state[f"preview_{origin}"] = snapshot_rows(result)
            st.session_state[f"plan_note_{origin}"] = note
        except ValueError as exc:
            st.error(str(exc))
    left, right = st.columns([1.15, 1.5], gap="large")
    with left:
        st.subheader("Operating range")
        st.caption('Edits below are a draft until saved. Use Save range and generate batch to apply them and refresh the preview in one step.')
        if st.button('Edit materials & consumables', key=f'plan_to_materials_{origin}'):
            st.session_state['page_jump'] = 'Materials & consumables'
            st.rerun()
        engine = st.selectbox('Prediction engine', ['reliability', 'adaptive', 'baseline'],
                              index=['reliability', 'adaptive', 'baseline'].index(space.get('model_engine', 'adaptive')),
                              format_func={'reliability': 'Recipe reliability and adaptive transfer', 'adaptive': 'Adaptive hierarchical GP', 'baseline': 'Fixed-kernel baseline'}.get,
                              key=f'prediction_engine_{origin}')
        policy = st.selectbox('Next-batch strategy', ['balanced', 'success', 'exploration'],
                             index=['balanced', 'success', 'exploration'].index(space.get('acquisition_policy', 'balanced')),
                             format_func={'balanced': 'Balanced learning', 'success': 'Success focused', 'exploration': 'Exploration focused'}.get,
                             key=f'acquisition_policy_{origin}')
        if engine != space.get('model_engine', 'adaptive') or policy != space.get('acquisition_policy', 'balanced'):
            space.update(model_engine=engine, acquisition_policy=policy)
            store.put_many('meta', [{'id': 'space', 'origin': origin, 'space': space}], 'id', overwrite=True)
            st.session_state.pop(f'preview_{origin}', None)
            st.session_state.pop(f'plan_note_{origin}', None)
            st.rerun()
        with st.form("plan_config"):
            from sacfactory.experiment_defaults import shared_process_values
            process_values = shared_process_values(context)
            precursor, support, protocol = (process_values[k] for k in ("precursor_family", "support_batch", "protocol_id"))
            planned_chemistry = planning_precursors(store, context, chosen_elements)
            use_cutoff = st.checkbox("Include controlled cutoff temperature in the model", value=context.get("pulse_limit_C") is not None)
            tlo, thi = st.columns(2)
            cutoff_min = tlo.number_input("Cutoff lower bound / °C", min_value=1.0, value=float(space.get("pulse_limit_min_C") or 1000), step=25.)
            cutoff_max = thi.number_input("Cutoff upper bound / °C", min_value=1.0, value=float(space.get("pulse_limit_max_C") or 1000), step=25.)
            temperature_step = st.number_input('Temperature step / °C', min_value=1, value=int(space.get('temperature_step_C', 10)), step=1,
                help='10 proposes only integer multiples of ten within the allowed bounds. Probabilities are calculated at the actual proposed setpoints.')
            stored_bounds = space.get("concentration_bounds_mM", {})
            legacy_bounds = {}
            if not stored_bounds and chosen_elements and 'concentration_bounds_mM' not in space:
                try:
                    legacy_bounds = concentration_bounds(space, chosen_elements)
                except ValueError as exc:
                    st.warning(str(exc))
            range_rows = []
            for e in chosen_elements:
                ref = context["concentrations_mM"].get(e) or base_concentration(e)
                bounds = stored_bounds.get(e) or legacy_bounds.get(e) or {"min_mM": ref, "max_mM": ref}
                range_rows.append({"element": e, "reference_mM": ref, **bounds})
            st.write("**Precursor concentrations / mM**")
            st.caption("Enter each element directly. Set lower = upper to keep its concentration fixed. The reference is a recipe anchor, not a measured outcome.")
            concentration_ranges = st.data_editor(pd.DataFrame(range_rows, columns=["element", "reference_mM", "min_mM", "max_mM"]), hide_index=True, disabled=["element"], width="stretch",
                column_config={k: st.column_config.NumberColumn(label, min_value=.000001, format="%.6f", required=True) for k, label in (("reference_mM", "Reference / mM"), ("min_mM", "Lower / mM"), ("max_mM", "Upper / mM"))},
                key=f"plan_concentrations_{origin}_{fingerprint({'elements': chosen_elements, 'context': context['concentrations_mM'], 'space': space})}")
            pulse_text = st.text_input("Available pulse counts, comma-separated", value=",".join(str(n) for n in space["pulses"]))
            st.caption('Enter 1,2,3 to compare three pulse settings. Entering 10 means one setting of ten pulses; it does not request ten samples.')
            max_total = st.number_input("Maximum total metal precursor concentration / mM", min_value=.0001, value=float(space["max_total_mM"]), step=.1)
            with st.expander('Planned preparation basis'):
                plan_pan = st.number_input('Planned PAN mass / g', min_value=.000001, value=float(context.get('pan_mass_g') or 2.))
                plan_volume = st.number_input('Planned DMF volume / mL', min_value=.000001, value=float(context.get('solvent_volume_mL') or 20.))
            range_source = st.text_input("Operating-range validation source", value=space["range_source"])
            verified = st.checkbox("The entered operating range has been validated on the device" if origin == "REAL" else "This range is for simulation only", value=bool(space.get('verified', False)))
            with st.expander("Advanced learning, repeats & sensitivity"):
                transfer_enabled = st.checkbox("Use existing single-element and mixture data for transfer", value=space.get("element_transfer", True))
                repeat = st.checkbox("Allow one explicitly labeled independent repeat", value=space["baseline_repeat"])
                repeat_strategy = st.selectbox('Repeat selection', ['reference', 'critical', 'allocation', 'none'],
                    index=['reference', 'critical', 'allocation', 'none'].index(space.get('repeat_strategy', 'reference')),
                    format_func={'reference': 'Reference recipe', 'critical': 'Uncertain or unstable existing condition', 'allocation': 'Adaptive repeat-budget allocation', 'none': 'No repeat'}.get)
                descriptor_profile = st.selectbox('Prediction descriptor profile', list(PROFILES),
                    index=list(PROFILES).index(space.get('descriptor_profile', 'chemistry_pulseicp')), format_func=PROFILES.get)
                st.caption('Transfer uses declared precursor chemistry, support and synthesis conditions.')
                repeat_fraction = st.slider('Maximum next-batch repeat fraction', 0., 1., float(space.get('repeat_budget_fraction', .5)), .1)
                repeat_goal = st.slider('Working repeat half-width target', .05, .45, float(space.get('repeat_goal_half_width', .15)), .05)
                repeat_cap = st.number_input('Additional-repeat planning cap per recipe', min_value=1, max_value=100, value=int(space.get('repeat_cap', 20)))
                concentration_tolerance = st.number_input('Observed relative concentration tolerance (0 = unknown)', min_value=0., max_value=.5, value=float(space.get('concentration_tolerance_fraction', 0)), step=.01)
                cutoff_tolerance = st.number_input('Observed cutoff tolerance / C (0 = unknown)', min_value=0., max_value=200., value=float(space.get('cutoff_tolerance_C', 0)), step=1.)
                robustness_weight = st.slider('Finite-scenario robustness utility weight', 0., 1., float(space.get('robustness_weight', 0)), .1)
                st.caption('Tolerances start unknown. Supply measured estimates to inspect input sensitivity; pulse count is unchanged. Robust utility requires enough in-bound scenarios and is not a certified safe-region calculation.')
                st.caption('Repeat allocation uses observed variability and the available experiment budget.')
                batch_effect = st.checkbox('Model shared synthesis-batch variation', value=space.get('batch_effect', False),
                    help='Off by default: compatible batches train together without an extra batch offset. Batch IDs still support provenance and held-out evaluation. Enable this only when shared batch variation needs modeling.')
                st.caption('Adaptive sampling balances uncertainty, predicted outcomes and recipe diversity.')
            save_range = st.form_submit_button("Save operating range")
            save_generate = st.form_submit_button('Save range and generate batch', type='primary')
            if save_range or save_generate:
                try:
                    elements = parse_elements(";".join(chosen_elements))
                    concentrations = concentration_ranges.to_dict("records")
                    refs = {r["element"]: finite_number(r["reference_mM"], f"{r['element']} reference concentration", True) for r in concentrations}
                    updated = {"concentration_bounds_mM": {r["element"]: {k: r[k] for k in ("min_mM", "max_mM")} for r in concentrations}, "pulses": [int(n.strip()) for n in pulse_text.replace("，", ",").split(",") if n.strip()],
                               "max_total_mM": max_total, "element_transfer": transfer_enabled, "verified": verified, "baseline_repeat": repeat, "range_source": range_source,
                               'model_engine': engine, 'acquisition_policy': policy, 'learning_iterations': space.get('learning_iterations', 18), 'temperature_step_C': temperature_step,
                               'repeat_strategy': repeat_strategy, 'batch_effect': batch_effect, 'descriptor_profile': descriptor_profile,
                               'repeat_budget_fraction': repeat_fraction, 'repeat_goal_half_width': repeat_goal, 'repeat_cap': repeat_cap,
                               'concentration_tolerance_fraction': concentration_tolerance, 'cutoff_tolerance_C': cutoff_tolerance, 'robustness_weight': robustness_weight}
                    if use_cutoff: updated.update(pulse_limit_min_C=cutoff_min, pulse_limit_max_C=cutoff_max)
                    validate_space(updated, origin)
                    if not range_source or not all(v.strip() for v in (precursor, support, protocol)):
                        raise ValueError("Provide range evidence and process metadata.")
                    ctx = {**context, "sample_id": "CUSTOM_CONTEXT", "elements": elements, "concentrations_mM": refs, "precursor_family": precursor,
                           "support_batch": support, "protocol_id": protocol, "temperature_C": None if use_cutoff else 1000, "pulse_limit_C": cutoff_min if use_cutoff else None, "pulse_count": 1, "icp": []}
                    ctx['precursors'] = normalize_precursors(planned_chemistry, elements)
                    ctx.update(pan_mass_g=plan_pan, solvent_volume_mL=plan_volume, preparation_basis='Declared next-batch preparation')
                    store.put_many("meta", [{"id": "space", "origin": origin, "space": updated}, {"id": "plan_context", "origin": origin, "context": ctx}], "id", overwrite=True)
                    if save_generate:
                        st.session_state[f'generate_after_save_{origin}'] = True
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    with right:
        st.subheader("Batch preview")
        if st.session_state.get(f'preview_refresh_notice_{origin}'):
            st.info(st.session_state.pop(f'preview_refresh_notice_{origin}'))
        st.write(f"Current **{len(context['elements'])} elements** · Pulses {space['pulses']}")
        gate = state.info.get('fitting_gate')
        if gate:
            st.caption(f"Fitting inputs: {gate['eligible_samples']}/12 eligible samples; "
                f"{gate['pass_recipes']}/3 distinct PASS recipes; {gate['fail_recipes']}/3 distinct FAIL recipes. "
                'These gates are separate from the requested experiment count.')
        if state.info.get('exclusions'):
            with st.expander('Why records were excluded from fitting'):
                st.dataframe(pd.DataFrame([{'reason': k, 'records': v} for k, v in state.info['exclusions'].items()]),
                    hide_index=True, width='stretch')
        from sacfactory.experiment_defaults import fixed_search_space
        try:
            fixed_message = fixed_search_space(space, context['elements'])
            if fixed_message:
                st.info(fixed_message)
        except (ValueError, KeyError):
            pass
        if campaign_enabled:
            st.caption(f"Single-element-first campaign: {covered} / {len(TARGETS)} elements have screening candidates. Single-element planning learns only from other singles; mixture proposals require complete single-element coverage. Review Element campaign under Data analysis.")
        if st.session_state.get(f"plan_note_{origin}"):
            st.caption(st.session_state[f"plan_note_{origin}"])
        render_recommendations(st.session_state.get(f"preview_{origin}", []), "plan")
        with st.expander("Model evidence and transfer diagnostics"):
            if state.info.get("feature_kind") == "COMPOSITION_TRANSFER":
                st.write(f"**Training source pool:** {state.info['n_labeled']} records across {len(state.info['source_compositions'])} compositions.")
                st.caption(f"Target {state.info['target_element']}: {state.info['target_labeled_n']} exact-composition labeled records. Source-only predictions require bridge experiments on this composition.")
                st.caption("Source stages: " + (", ".join(f"{int(n)} elements ({count} records)" for n, count in sorted(state.info["source_stages"].items(), key=lambda item: int(item[0]))) or "No eligible observations"))
                st.caption(f"{len(state.info['unseen_elements'])} unseen target elements. Pairs without prior co-synthesis records: {state.info['unobserved_pair_count']} / {state.info['target_pair_count']}.")
                with st.container():
                    st.dataframe(pd.DataFrame([{"composition": name, "source_records": count} for name, count in state.info["source_compositions"].items()]), hide_index=True, width="stretch")
                    st.dataframe(pd.DataFrame([{"target_element": e, "source_occurrences": state.info["source_elements"].get(e, 0)} for e in context["elements"]]), hide_index=True, width="stretch")
                if state.info["highest_source_stage"] and len(context["elements"]) > state.info["highest_source_stage"]:
                    st.warning(f"Expansion from source stage {state.info['highest_source_stage']} to {len(context['elements'])} elements. Numerical predictions are transfer hypotheses; a large jump can have very low source support.")
            with st.container():
                try:
                    st.dataframe(pd.DataFrame([{"element": e, **b} for e, b in concentration_bounds(space, context["elements"]).items()]), hide_index=True, width="stretch")
                except ValueError as exc:
                    st.caption(str(exc))
            if context.get("pulse_limit_C") is not None: st.caption(f"Controlled cutoff range: {space.get('pulse_limit_min_C')}–{space.get('pulse_limit_max_C')} °C")
            st.caption(space["range_source"])
            st.info(state.info["reason"])
            st.caption(f"Trainable records with a direct ICP or XRD judgment: {state.info.get('reported_label_count', 0)}. These labels are user-reported; numeric results and evidence files can be added later.")
            if state.info.get("validation"):
                v = state.info["validation"]
                cols = st.columns(2)
                cols[0].metric("Grouped balanced accuracy", f"{v['balanced_accuracy']:.3f}")
                cols[1].metric("Grouped Brier score", f"{v['brier']:.3f}")
                st.caption(f"{v['n']} / {v.get('total_n', v['n'])} records covered; {v['n_groups']} groups. {v['type']}. Probabilities are not prospectively validated.")
            elif state.model:
                st.caption(state.info.get("validation_note", "No independent validation yet"))
            if state.info.get('engine') in {'adaptive', 'reliability'}:
                st.write('**Adaptive learning and validation**')
                st.caption('Kernel parameters are learned again after eligible records change. The joint acquisition weight adapts to target novelty, posterior information and eligible prior forecast errors. Device bounds remain required.')
                if state.model:
                    st.caption(f"GP fit: {state.info['gp_fit_n']} records; separate calibration: {state.info['calibration_n']}. {state.info['calibration_status']}. New compositions remain uncalibrated transfer hypotheses.")
                    with st.container():
                        if state.info.get('engine') == 'reliability':
                            st.write('**Target/source development weights**')
                            st.caption('Weights are development OOF fits. Pooled kernel diagnostics below describe one expert; the combined predictor is explicitly uncalibrated. Independent model/descriptor benchmarks are on Data analysis.')
                        st.dataframe(pd.DataFrame([{'parameter': k, 'initial': state.info['learning']['initial_parameters'][k], 'learned': v} for k, v in state.info['learning']['parameters'].items()]), hide_index=True, width='stretch')
                        st.caption('Batch-effect covariance is fitted only with cross-batch repeated recipes. Prediction integrates variation for a new batch; joint acquisition assumes the proposed samples share one new batch.')
                        if state.info['model_comparison']:
                            st.dataframe(pd.DataFrame(state.info['model_comparison']), hide_index=True, width='stretch')
                        st.caption('Comparison uses outer batch holdouts; calibration batches never enter GP fitting. It does not establish new-element or prospective superiority.')
            if state.info.get("feature_kind") == "COMPOSITION_TRANSFER":
                tv = state.info.get("transfer_validation")
                if tv:
                    st.write(f"**Held-out-composition check:** balanced accuracy {tv['balanced_accuracy']:.3f}; Brier {tv['brier']:.3f}; {tv['n']} / {tv['total_n']} records covered.")
                st.caption(state.info["transfer_validation_note"])
                ev = state.info.get("expansion_validation")
                if ev:
                    st.write(f"**Lower-to-higher-stage check:** balanced accuracy {ev['balanced_accuracy']:.3f}; Brier {ev['brier']:.3f}; {ev['n']} / {ev['total_n']} records covered.")
                st.caption(state.info["expansion_validation_note"])
                st.caption("Source-pool, held-out-composition and lower-to-higher-stage validation answer different questions. None establishes prospective reliability for a new composition.")
        with st.expander("Observed experiments"):
            observed = []
            observed_element = element_selectbox("Plot element concentration", context["elements"], key=f"observed_element_{origin}")
            for s in samples:
                if cohort_key(s) == cohort_key(context) and s["concentrations_mM"].get(observed_element) is not None:
                    observed.append({"Concentration / mM": s["concentrations_mM"][observed_element], "Pulse count": s["pulse_count"], "Status": STATUS_NAMES[screen(s, methods)["status"]], "Sample": s["sample_id"]})
            if observed:
                st.write("**Observed experiments in this cohort**")
                chart = alt.Chart(pd.DataFrame(observed)).mark_circle(size=90).encode(
                    x=alt.X("Concentration / mM:Q", scale=alt.Scale(type="log"), title=f"{observed_element} concentration / mM"), y=alt.Y("Pulse count:O", axis=alt.Axis(labelOverlap=False)),
                    color=alt.Color("Status:N"), tooltip=["Sample:N", "Concentration / mM:Q", "Pulse count:O", "Status:N"]
                ).properties(height=max(180, 28 * len({r["Pulse count"] for r in observed})))
                st.altair_chart(chart, width="stretch")
        with st.expander("Reserved work queue"):
            open_rows = [r for r in reservations if r["state"] == "OPEN"]
            if open_rows:
                st.dataframe(pd.DataFrame(recommendation_table(open_rows))[["recommendation_id", "concentration_mM", "pulse_count", "purpose"]], hide_index=True, width="stretch")
                close_id = st.selectbox("Complete or cancel a task", [r["recommendation_id"] for r in open_rows])
                close_state = st.selectbox("New task status", ["COMPLETED", "CANCELLED"])
                if st.button("Record task status"):
                    row = copy.deepcopy(next(r for r in open_rows if r["recommendation_id"] == close_id))
                    row["state"] = close_state
                    store.put_many("recommendation", [row], "recommendation_id", overwrite=True)
                    st.rerun()
            else:
                st.caption("No experiments are currently reserved.")

elif page == "Data analysis":
    render_analysis(samples, methods, origin, reservations, context, space, state.info, store)

elif page == "Language model":
    st.subheader("Optional conversational API")
    st.write('Use a compatible Responses API or Chat Completions provider, including institutional gateways. Research chat can discuss recorded data in your language and call local analysis/planning tools. Task interpretation remains available.')
    st.caption('Use a key issued by your selected provider or institution. Access, quotas and billing follow that provider. ChatGPT website access is separate from API access.')
    current = st.session_state.get("api_config")
    if st.session_state.pop('clear_api_key_input', False):
        st.session_state['api_key_input'] = ''

    def revoke_api_sharing():
        for mode_name in ('REAL', 'DEMO'):
            for prefix in ('api_permission', 'api_checkbox', 'research_permission', 'research_checkbox'):
                st.session_state[f'{prefix}_{mode_name}'] = False

    preset_columns = st.columns(2)
    openai_preset = preset_columns[0].button('Use OpenAI preset')
    parley_preset = preset_columns[1].button('Use MIT Parley preset')
    if openai_preset or parley_preset:
        st.session_state['api_base_url'] = MIT_PARLEY_BASE if parley_preset else 'https://api.openai.com/v1'
        st.session_state['api_model_id'] = ''
        st.session_state['api_format_choice'] = 'responses'
        st.session_state['api_assistant_mode'] = 'Research conversation'
        st.session_state['assistant_mode_setting'] = 'Research conversation'
        st.session_state['api_key_input'] = ''
        st.session_state.pop('api_model_catalog', None)
        revoke_api_sharing()
    draft = st.session_state.get('api_pending_config') or current or environment_config()
    for key, default in {'api_base_url': draft.base_url, 'api_model_id': draft.model, 'api_format_choice': draft.api_format}.items():
        if st.session_state.get(key) is None:
            st.session_state[key] = default
    catalog = st.session_state.get('api_model_catalog')
    if catalog and same_base(catalog['base_url'], st.session_state['api_base_url']):
        def select_catalog_model():
            selected = st.session_state.get('api_catalog_choice')
            if selected:
                st.session_state['api_model_id'] = selected
        st.selectbox('Available models', catalog['models'], index=None, key='api_catalog_choice', on_change=select_catalog_model,
                     help='Select a provider model to fill Model ID, then apply session settings. A listed model may not support every API feature.')
    def clear_key_on_endpoint_edit():
        st.session_state['api_key_input'] = ''
        revoke_api_sharing()

    with st.container():
        base_url = st.text_input("API base URL", value=None, key='api_base_url', on_change=clear_key_on_endpoint_edit)
        model = st.text_input("Model ID", value=None, help="Enter a model ID available in your provider account. No model is assumed.", key='api_model_id')
        api_format = st.selectbox('API format', ['chat_completions', 'responses'], index=None, format_func={'chat_completions': 'Compatible Chat Completions', 'responses': 'Responses API (OpenAI-compatible)'}.get, key='api_format_choice')
        entered_key = st.text_input("API key", type="password", key='api_key_input', help="Session memory only. Leave blank to keep the current or pending key for this same endpoint, or use matching environment credentials. Never paste a key into chat.")
        apply_settings = st.button("Apply session settings")
        load_models = st.button('Load available models')
        if apply_settings or load_models:
            revoke_api_sharing()
            candidate = setup_candidate(base_url, model, entered_key, api_format, st.session_state.get('api_pending_config'), current)
            try:
                candidate.credential_base()
                st.session_state['api_pending_config'] = candidate
                st.session_state['clear_api_key_input'] = True
                if load_models:
                    st.session_state.pop('api_model_catalog', None)
                    models = list_available_models(candidate)
                    st.session_state['api_model_catalog'] = {'base_url': base_url, 'models': models, 'setup': candidate}
                    st.session_state.pop('api_catalog_choice', None)
                    st.session_state['api_setup_notice'] = f'Loaded {len(models)} available models. Select one, then apply settings. No experimental data was sent.'
                    st.rerun()
                candidate.endpoint()
                st.session_state["api_config"] = candidate
                st.session_state.pop('api_pending_config', None)
                current = candidate
                if candidate.api_format == 'responses':
                    st.session_state['api_assistant_mode'] = 'Research conversation'
                    st.session_state['assistant_mode_setting'] = 'Research conversation'
                st.success("Settings applied for this browser session. Requests remain disabled until enabled below. No connection test was sent.")
            except LanguageError as exc:
                st.error(str(exc))
    if st.session_state.get('api_setup_notice'):
        st.info(st.session_state.pop('api_setup_notice'))
    if st.session_state.get('api_pending_config'):
        st.caption('A setup key is retained in this browser session. Leave API key blank while correcting the model for the same endpoint.')
    catalog_setup = (st.session_state.get('api_model_catalog') or {}).get('setup')
    if catalog_setup:
        if current and same_base(current.base_url, catalog_setup.base_url) and current.api_key == catalog_setup.api_key:
            st.success('Model-catalog authentication succeeded for the applied endpoint and key. Chat inference and model permissions have not been tested.')
        else:
            st.warning('Model-catalog authentication succeeded for the entered settings. Select a model and apply session settings to use this key for chat.')
    st.caption('For MIT Parley, choose its preset, enter your institutional key, load available models, select a model, then apply. If model discovery is unavailable, enter the exact ID supplied by your institution. Model listing does not verify research-chat or tool compatibility.')
    st.caption("Settings and keys are held in session memory, never in SQLite or exported records. Environment variables are optional. Applying settings revokes both REAL and DEMO permission.")
    def save_assistant_mode():
        st.session_state['assistant_mode_setting'] = st.session_state['api_assistant_mode']
    if st.session_state.get('api_assistant_mode') is None:
        st.session_state['api_assistant_mode'] = st.session_state.get('assistant_mode_setting', 'Task interpretation')
    mode = st.radio('Assistant mode', ['Task interpretation', 'Research conversation'], index=None, key='api_assistant_mode', horizontal=True, on_change=save_assistant_mode)
    st.session_state['assistant_mode_setting'] = mode
    if mode == 'Task interpretation' and current and current.api_format == 'responses':
        st.warning('The applied provider uses /responses. Select Research conversation to use it. Task interpretation will not send a request to a different endpoint.')
    if mode == 'Research conversation':
        budget = st.number_input('Maximum output tokens per research API call', min_value=512, max_value=8192, value=st.session_state.get('research_token_budget', 3000), step=256, key='research_output_budget')
        st.session_state['research_token_budget'] = budget
        render_data_selection(samples, origin, context)
    preview = shared_context(samples, methods, state, context, space, origin)
    if mode == 'Research conversation':
        preview = research_context(samples, methods, state, context, space, origin, st.session_state.get(f'research_samples_{origin}', []))
    with st.expander("Preview experimental context sent to the provider", expanded=True):
        if mode == 'Research conversation':
            st.caption(f"{preview['selected_record_count']} selected records · {len(preview['shared_records'])} detail rows in initial context · {preview['sharing_mode']}")
        st.dataframe(pd.DataFrame([{'field': key, 'content': json.dumps(value, ensure_ascii=True)}
            for key,value in preview.items() if key != 'shared_records']),hide_index=True,width='stretch')
        if preview.get('shared_records'):
            st.dataframe(pd.json_normalize(preview['shared_records']),hide_index=True,width='stretch')
        st.download_button('Download shared context',json.dumps(preview,indent=2).encode(),
            'shared_context.json','application/json',key=f'shared_context_download_{origin}')
    st.write('Research mode sends this context, your message, up to 12 recent research messages in the same selection/discussion, and any requested local tool outputs. Only selected recorded conditions and entered ICP W are shared individually; aggregate tools can summarize this data space. Notes, source paths and raw files are excluded. Changing the selection/provider or starting a new discussion excludes earlier conversation from the API context.' if mode == 'Research conversation' else 'Task mode sends the displayed minimal context, your current message and up to six validated intents. Raw ICP values, files, notes, source paths and full chat history are excluded.')

    def update_api_permission():
        st.session_state[f"api_permission_{origin}"] = st.session_state[f"api_checkbox_{origin}"]

    st.checkbox(f"Enable API requests for {origin} and share the displayed context with this endpoint",
                value=st.session_state.get(f"api_permission_{origin}", False), disabled=current is None,
                key=f"api_checkbox_{origin}", on_change=update_api_permission)
    if mode == 'Research conversation':
        def update_research_permission():
            st.session_state[f'research_permission_{origin}'] = st.session_state[f'research_checkbox_{origin}']
        st.checkbox(f'Share selected experimental records and recent research conversation for {origin}', value=st.session_state.get(f'research_permission_{origin}', False), disabled=current is None or not st.session_state.get(f'api_permission_{origin}'), key=f'research_checkbox_{origin}', on_change=update_research_permission)
    st.caption('REAL and DEMO permissions are separate. Research mode uses at most three API calls per turn, up to 60 seconds each, with no automatic retries. Token usage is displayed after replies. Task mode uses one 30-second JSON call. Shortcuts remain local. Apply settings again when changing endpoint or model.')
    if current:
        route = '/responses' if current.api_format == 'responses' else '/chat/completions'
        st.info(f"Configured model: {current.model} · API base: {current.base_url} · Selected route: {route} · Mode: {mode} · Key available in memory")
        st.caption('The connection test sends only a short OK prompt with the applied key/model. It includes no experiments, chat history or tools, makes one request, and may count toward provider usage. It does not enable data sharing.')
        if st.button('Test connection (no experiment data)'):
            try:
                probe = probe_chat_connection(current)
                if probe['complete_text_reply']:
                    st.success(f"Connection verified on {probe['route']}: a text reply was received. Research tool support remains untested.")
                else:
                    st.info(f"The provider accepted the request on {probe['route']}, but no complete text reply was received within the small test budget. Research tool support remains untested.")
            except LanguageError as exc:
                st.error(str(exc))
    if st.button("Disconnect and forget session credentials"):
        st.session_state.pop("api_config", None)
        for key in ('api_pending_config', 'api_model_catalog', 'api_catalog_choice', 'api_key_input', 'api_setup_notice', 'clear_api_key_input'):
            st.session_state.pop(key, None)
        for mode_name in ("REAL", "DEMO"):
            st.session_state[f"api_permission_{mode_name}"] = False
            st.session_state.pop(f"api_checkbox_{mode_name}", None)
            st.session_state[f'research_permission_{mode_name}'] = False
            st.session_state.pop(f'research_checkbox_{mode_name}', None)
        st.rerun()
    st.markdown('Compatibility reference: [Model listing](https://developers.openai.com/api/reference/resources/models/methods/list) · [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state) · [Function calling](https://developers.openai.com/api/docs/guides/function-calling). Obtain credentials and model IDs from your chosen provider or institution.')

elif page == "About":
    st.subheader("SACFactory")
    st.write("An experimental workbench for single-atom catalyst synthesis, data management, SAC-Gen prediction and batch planning.")
    st.dataframe(pd.DataFrame([
        {"Module": "Experiments", "Capabilities": "Element selection, precursor quantities, preparation conditions, editable records and pending results"},
        {"Module": "SAC-Gen", "Capabilities": "Mass-conserving loading prediction and XRD PASS/FAIL probabilities"},
        {"Module": "Validation", "Capabilities": "Masked batch and element tests, endpoint metrics and prediction agreement"},
        {"Module": "Benchmarks", "Capabilities": "Matched statistical baselines and recorded-pool acquisition replay"},
        {"Module": "Planning", "Capabilities": "Constrained recommendations, manual amendment and pending experiment management"},
        {"Module": "Conversation", "Capabilities": "Local commands and optional API conversation with selected experimental context"},
    ]),hide_index=True,width="stretch")
    for name,label in [('USER_GUIDE.md','User guide'),('METHODS.md','Scientific methods'),
                       ('BENCHMARKS.md','Benchmark design'),('CREDITS.md','Sources and acknowledgments')]:
        st.download_button(label,(Path(__file__).parent/name).read_bytes(),name,'text/markdown',key='about_'+name)

st.caption(f"SACFactory {__version__}")
