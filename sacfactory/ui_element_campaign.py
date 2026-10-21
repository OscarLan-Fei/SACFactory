"""Single-element campaign progress and cross-element prediction diagnostics."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from .domain import TARGETS, ELEMENTS
from .element_campaign import campaign_progress, element_transfer_validation
from .element_loading import combined_element_loading
from .loading import MODELS
from .planning import fingerprint
from .precursors import PROFILES
from .ui_elements import element_multiselect
from .ui_physics import render_physics_options, physical_comparison_ready


@st.cache_data(show_spinner=False, max_entries=8)
def cached_element_transfer(payload):
    return element_transfer_validation(**json.loads(payload))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_combined_loading(payload):
    return combined_element_loading(**json.loads(payload))


def render_combined_loading(samples, origin, singles):
    st.write('**Combined cross-element loading plot**')
    st.caption('Plot all selected elements together. Each element is predicted by a separate fit that excludes every result for that element and all its shared batches. Mixture observations do not enter this single-element phase.')
    observed = list(dict.fromkeys(s['elements'][0] for s in singles))
    key = f'combined_loading_elements_{origin}'
    st.session_state.setdefault(key, observed)
    columns = st.columns(2)
    if columns[0].button('Select all observed single elements', key=f'combined_loading_observed_{origin}'):
        st.session_state[key] = observed
    if columns[1].button('Select all 50 campaign elements', key=f'combined_loading_all50_{origin}'):
        st.session_state[key] = list(TARGETS)
    selected = element_multiselect('Elements in combined loading plot', ELEMENTS, key=key)
    family = st.selectbox('Combined loading regression model', MODELS, key=f'combined_loading_model_{origin}')
    profile = st.selectbox('Combined loading descriptor profile', list(PROFILES), index=2, format_func=PROFILES.get, key=f'combined_loading_profile_{origin}')
    physics = render_physics_options(f'combined_loading_physics_{origin}')
    payload = json.dumps({'samples': samples, 'origin': origin, 'targets': selected, 'family': family, 'profile': profile, 'physics': physics}, sort_keys=True)
    signature = fingerprint(payload)
    if st.button('Evaluate and plot selected elements', disabled=not selected or not physical_comparison_ready(physics), key=f'combined_loading_run_{origin}'):
        with st.spinner('Fitting one protected cross-element model per target and process context...'):
            st.session_state[f'combined_loading_result_{origin}'] = (signature, cached_combined_loading(payload))
    saved = st.session_state.get(f'combined_loading_result_{origin}')
    if not saved or saved[0] != signature:
        return
    frame, report, info = saved[1]
    if frame.empty:
        st.info('No evaluable held-out loading points. Inspect target measurements, compatible source data and batch requirements below.')
    else:
        score = info['metrics']
        cols = st.columns(3)
        cols[0].metric('Plotted elements', frame.element.nunique())
        cols[1].metric('Combined held-out samples', len(frame))
        cols[2].metric('Combined loading MAE / wt%', f"{score['mae_wt_percent']:.4f}")
        if info.get('paired_comparison'):
            st.dataframe(pd.DataFrame(info['paired_comparison']), hide_index=True, width='stretch')
        elements = frame.element.drop_duplicates().tolist()
        limit = max(.01, float(frame[['predicted_w_wt_percent', 'observed_w_wt_percent']].max().max()) * 1.08)
        data = alt.Data(values=frame.astype(object).where(pd.notna(frame), None).to_dict('records'))
        dots = alt.Chart(data).mark_point(filled=True, size=65, opacity=.7).encode(
            x=alt.X('predicted_w_wt_percent:Q', title='Predicted element W / wt%', scale=alt.Scale(domain=[0, limit])),
            y=alt.Y('observed_w_wt_percent:Q', title='Measured element W / wt%', scale=alt.Scale(domain=[0, limit])),
            color=alt.Color('element:N', title='Element', sort=elements, scale=alt.Scale(domain=elements, scheme='turbo'),
                legend=alt.Legend(orient='bottom', columns=5,symbolLimit=len(elements))),
            shape=alt.Shape('outside_training_range:N', title='Outside source ranges'),
            tooltip=[alt.Tooltip(c, type='nominal' if c in {'sample_id', 'batch_id', 'element', 'composition', 'process_context', 'outside_training_range'} else 'quantitative') for c in frame.columns])
        ideal = alt.Chart(pd.DataFrame({'v': [0, limit]})).mark_line(strokeDash=[4, 4], color='#8a98a6').encode(x='v:Q', y='v:Q')
        st.altair_chart((dots + ideal).properties(height=400 + 20 * ((len(elements) + 4) // 5)), width='stretch')
        st.download_button('Download combined loading points CSV', frame.to_csv(index=False).encode(), f'{origin.lower()}_combined_element_loading.csv', 'text/csv', key=f'combined_loading_points_{origin}')
    st.dataframe(report, hide_index=True, width='stretch')
    st.download_button('Download combined element evaluation report', report.to_csv(index=False).encode(), f'{origin.lower()}_combined_element_report.csv', 'text/csv', key=f'combined_loading_report_{origin}')
    st.download_button('Download combined loading provenance', json.dumps(info, indent=2, allow_nan=False).encode(), f'{origin.lower()}_combined_element_loading.json', 'application/json', key=f'combined_loading_provenance_{origin}')


def render_element_campaign(samples, methods, origin, store=None):
    st.write('**Single-element-first campaign**')
    st.caption('Evaluate selected elements and process contexts separately; single-element screening does not establish mixture performance.')
    if store is not None:
        policy = store.get('meta', 'element_campaign') or {}
        enabled = st.checkbox('Complete 50 single-element candidates before mixtures', value=bool(policy.get('enabled')), key=f'element_campaign_policy_{origin}')
        if st.button('Save campaign strategy', key=f'element_campaign_save_{origin}'):
            store.put_many('meta', [{'id': 'element_campaign', 'origin': origin, 'enabled': enabled,
                'strategy': 'SINGLE_ELEMENT_FIRST' if enabled else 'OPEN_COMPOSITION_PLANNING'}], 'id', overwrite=True)
            st.rerun()
    progress = campaign_progress(samples, methods, origin)
    covered = int(progress.candidate_covered.sum())
    st.metric('Single elements with screening candidates', f'{covered} / {len(TARGETS)}')
    st.progress(covered / len(TARGETS))
    st.dataframe(progress, hide_index=True, width='stretch')
    st.download_button('Download single-element campaign CSV', progress.to_csv(index=False).encode(), f'{origin.lower()}_single_element_campaign.csv', 'text/csv', key=f'element_progress_{origin}')
    singles = [s for s in samples if s.get('origin') == origin and len(s['elements']) == 1]
    combined, individual = st.tabs(['All-element loading plot', 'Selected-element transfer'])
    with combined:
        render_combined_loading(samples, origin, singles)
    with individual:
        render_individual_transfer(samples, methods, origin, singles)


def render_individual_transfer(samples, methods, origin, singles):
    from .composition import process_key
    from .ui_calculation_batch import run_jobs,detail
    observed=list(dict.fromkeys(s['elements'][0] for s in singles))
    key=f'element_targets_{origin}'
    if st.button('Select all observed transfer targets',key=f'element_targets_all_{origin}'):
        st.session_state[key]=observed
    targets=element_multiselect('Cross-element targets',ELEMENTS,default=['Co'],key=key)
    sources=element_multiselect('Single-element source elements (empty = all observed)',observed,key=f'element_sources_{origin}')
    profile=st.selectbox('Cross-element descriptor profile',list(PROFILES),index=2,format_func=PROFILES.get,key=f'element_profile_{origin}')
    protocol=st.radio('Cross-element success validation',['Strict unseen element','Sequential element replay'],horizontal=True,key=f'element_protocol_{origin}')
    iterations=st.number_input('Element-transfer learning iterations',2,30,4,key=f'element_iterations_{origin}')
    folds=st.number_input('Element replay batch budget',1,12,3,key=f'element_folds_{origin}')
    jobs=[]
    for target in targets:
        contexts={}
        for s in singles:
            if s['elements']==[target]:contexts.setdefault(process_key(s),s)
        if not contexts:
            jobs.append((target,None))
        for index,ctx in enumerate(contexts.values(),1):
            payload=json.dumps({'samples':samples,'methods':methods,'context':ctx,'origin':origin,'target':target,
                'sources':[e for e in sources if e!=target],'profile':profile,'protocol':protocol,
                'iterations':int(iterations),'max_folds':int(folds)},sort_keys=True)
            jobs.append((f'{target} / context {index}',payload))
    signature=fingerprint([jobs,profile,protocol,iterations,folds])
    if st.button('Evaluate cross-element success transfer',disabled=not targets,key=f'element_success_run_{origin}'):
        result=run_jobs(jobs,lambda payload:cached_element_transfer(payload) if payload else
            (pd.DataFrame(),{'reason':'No measured single-element target records','batch_status':'Unavailable'}))
        st.session_state[f'element_success_{origin}']=(signature,result)
    saved=st.session_state.get(f'element_success_{origin}')
    if saved and saved[0]==signature:
        frame,info=detail(saved[1],f'element_transfer_batch_{origin}')
        if frame.empty:
            st.info(info.get('reason','No evaluable source/target folds; inspect class and batch diversity.'))
        else:
            st.dataframe(pd.DataFrame([info['metrics']]),hide_index=True,width='stretch')
            st.dataframe(frame,hide_index=True,width='stretch')
            st.download_button('Download cross-element success predictions',frame.to_csv(index=False).encode(),
                f'{origin.lower()}_element_success.csv','text/csv',key=f'element_success_csv_{origin}')
