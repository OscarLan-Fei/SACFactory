"""Auxiliary ICP mass-content prediction and protected validation interface."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from .composition import process_key
from .loading import MODELS, PROTOCOLS, loading_records, loading_validation
from .planning import fingerprint
from .precursors import PROFILES
from .ui_elements import element_selectbox
from .ui_physics import render_physics_options, physical_comparison_ready


@st.cache_data(show_spinner=False, max_entries=8)
def cached_loading(payload):
    return loading_validation(**json.loads(payload))


def render_loading(samples, origin, root=None, all_samples=None):
    st.write('**ICP loading prediction / W (wt%)**')
    st.caption('Evaluate continuous ICP loading on held-out observations. Units: W / wt%.')
    options=['Statistical models','SAC-Gen loading models']
    if st.session_state.get(f'loading_engine_{origin}') not in options:st.session_state[f'loading_engine_{origin}']=options[0]
    engine = st.radio('ICP prediction engine', ['Statistical models', 'SAC-Gen loading models'], key=f'loading_engine_{origin}')
    if engine == 'SAC-Gen loading models':
        import os
        from pathlib import Path
        from .ui_pulseicp import render_pulseicp
        render_pulseicp(samples, origin, root or os.environ.get('SACFACTORY_DATA_DIR', str(Path(__file__).resolve().parents[1]/'data')))
        return
    modes=['All selected elements: shared model','Single-element model']
    if st.session_state.get(f'loading_data_mode_{origin}') not in modes:st.session_state[f'loading_data_mode_{origin}']=modes[0]
    mode=st.radio('Regression data mode',modes,
        key=f'loading_data_mode_{origin}')
    if mode=='All selected elements: shared model':
        from .ui_pooled_loading import render_pooled_loading
        render_pooled_loading(samples if all_samples is None else all_samples,origin)
        return
    elements = list(dict.fromkeys(e for s in samples for e in s['elements']))
    element = element_selectbox('Loading target element', elements, key=f'loading_element_{origin}')
    contexts = {}
    for sample in samples:
        if element in sample['elements']:
            contexts.setdefault(process_key(sample), sample)
    choices = list(contexts.values())
    if not choices:
        st.info('Add experimental records for the selected element to begin.')
        return
    index = 0 if len(choices)==1 else st.selectbox('Loading process context', range(len(choices)),
        format_func=lambda i: ' / '.join(str(choices[i].get(k, 'UNKNOWN')) for k in ('precursor_family', 'support_batch', 'protocol_id')),
        key=f'loading_context_{origin}_{element}')
    family = st.selectbox('Loading regression model', MODELS, key=f'loading_model_{origin}')
    profile = st.selectbox('Loading descriptor profile', list(PROFILES), index=2, format_func=PROFILES.get, key=f'loading_profile_{origin}')
    protocol = st.selectbox('Loading validation protocol', PROTOCOLS, key=f'loading_protocol_{origin}')
    physics = render_physics_options(f'loading_physics_{origin}')
    with st.expander('Loading validation settings and interpretation'):
        folds = st.number_input('Loading fold budget', min_value=2, max_value=6, value=3, key=f'loading_folds_{origin}')
    _, audit = loading_records(samples, origin, element, profile)
    if not audit.empty:
        st.caption(f'{int(audit.eligible.sum())} quantified records with complete controls and context before process selection; {int((~audit.eligible).sum())} excluded. Direct ICP PASS without numeric W is valid for screening but cannot supply a regression target.')
    with st.expander('ICP regression eligibility and missing values'):
        st.dataframe(audit, hide_index=True, width='stretch')
        st.download_button('Download loading eligibility CSV', audit.to_csv(index=False).encode(), f'{origin.lower()}_{element}_loading_eligibility.csv', 'text/csv', key=f'loading_audit_{origin}')
    payload = json.dumps({'samples': samples, 'origin': origin, 'element': element, 'context': choices[index],
        'family': family, 'profile': profile, 'protocol': protocol, 'max_folds': int(folds), 'physics': physics}, sort_keys=True)
    signature = fingerprint(payload)
    if st.button('Validate ICP loading predictions', disabled=not physical_comparison_ready(physics), key=f'loading_run_{origin}'):
        with st.spinner('Fitting regressors inside protected training folds...'):
            st.session_state[f'loading_result_{origin}'] = (signature, cached_loading(payload))
    saved = st.session_state.get(f'loading_result_{origin}')
    if not saved or saved[0] != signature:
        st.info('Run validation to obtain held-out predictions. Data, element, filter or settings changes invalidate previous results; no in-sample fit is presented as validation.')
        return
    predicted, audit, info = saved[1]
    if predicted.empty:
        st.info('No evaluable loading folds. At least three actual batches are required; each training fold needs eight quantified specimens, two batches and three recipes. Inspect exclusions and protocol-specific requirements below.')
    else:
        score = info['metrics']
        columns = st.columns(4)
        for col, label, value in zip(columns, ['Held-out loading samples', 'Loading MAE / wt%', 'Loading RMSE / wt%', 'Loading R-squared'],
            [str(score['n']), f"{score['mae_wt_percent']:.4f}", f"{score['rmse_wt_percent']:.4f}", f"{score['r_squared']:.3f}" if score['r_squared'] is not None else 'Unavailable']):
            col.metric(label, value)
        limit = max(.01, float(predicted[['observed_w_wt_percent', 'predicted_w_wt_percent']].max().max()) * 1.08)
        dots = alt.Chart(predicted).mark_circle(size=70, opacity=.7).encode(
            x=alt.X('predicted_w_wt_percent:Q', title=f'Predicted {element} W / wt%', scale=alt.Scale(domain=[0, limit])),
            y=alt.Y('observed_w_wt_percent:Q', title=f'Measured {element} W / wt%', scale=alt.Scale(domain=[0, limit])),
            color=alt.Color('element_count:N', title='Elements in recipe'), shape='outside_training_range:N',
            tooltip=list(predicted.columns))
        ideal = alt.Chart(pd.DataFrame({'value': [0, limit]})).mark_line(color='#8a98a6', strokeDash=[4, 4]).encode(x='value:Q', y='value:Q')
        st.altair_chart((dots + ideal).properties(height=370), width='stretch')
        if score['interval_coverage'] is not None:
            st.metric('GP nominal 95% interval observed coverage', f"{score['interval_coverage']:.1%}")
            st.caption('Coverage is measured on these held-out samples, not assumed to equal 95%. Conditional intervals are included in the exported point table.')
        st.dataframe(predicted, hide_index=True, width='stretch')
        st.dataframe(pd.DataFrame(info['stage_metrics']), hide_index=True, width='stretch')
        if info.get('paired_comparison'):
            st.write('**Paired loading models: identical held-out samples**')
            st.dataframe(pd.DataFrame(info['paired_comparison']), hide_index=True, width='stretch')
            st.caption('The scatter shows the residual model. Lower errors favor that model on this split only; no automatic model promotion. DEMO results test implementation and assumed relationships, not experimental validity.')
        st.download_button('Download held-out loading predictions CSV', predicted.to_csv(index=False).encode(), f'{origin.lower()}_{element}_loading_predictions.csv', 'text/csv', key=f'loading_predictions_{origin}')
    st.download_button('Download loading validation provenance', json.dumps(info, indent=2, allow_nan=False).encode(), f'{origin.lower()}_{element}_loading_validation.json', 'application/json', key=f'loading_provenance_{origin}')
