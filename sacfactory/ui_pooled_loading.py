"""Multi-select inputs for shared GP, forest and Ridge ICP regression."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from .domain import ELEMENTS
from .loading import MODELS,PROTOCOLS
from .planning import fingerprint
from .pooled_loading import collect,pooled_loading_validation
from .precursors import PROFILES
from .ui_elements import element_multiselect
from .ui_physics import render_physics_options,physical_comparison_ready


@st.cache_data(show_spinner=False,max_entries=8)
def cached_pooled_loading(payload):
    return pooled_loading_validation(**json.loads(payload))


@st.cache_data(show_spinner=False,max_entries=8)
def cached_pooled_eligibility(payload):
    rows,audit=collect(**json.loads(payload))
    return len(rows),len({r['element'] for r in rows}),audit


def render_pooled_loading(samples,origin):
    observed=list(dict.fromkeys(e for s in samples if s.get('origin')==origin for e in s['elements']))
    key=f'pooled_loading_elements_{origin}'
    st.session_state.setdefault(key,observed)
    if st.button('Select all available ICP elements',key=f'pooled_loading_all_{origin}'):
        st.session_state[key]=observed
    targets=element_multiselect('Elements in the shared ICP model',ELEMENTS,key=key)
    st.caption(f'{len(samples)} experimental records received from the current data selection. Selected elements train together; incompatible process contexts get separate fold models. Each target row includes the identity and descriptors of the element being predicted.')
    family=st.selectbox('Shared loading regression model',MODELS,key=f'pooled_loading_model_{origin}')
    profile=st.selectbox('Shared loading descriptor profile',list(PROFILES),index=2,
        format_func=PROFILES.get,key=f'pooled_loading_profile_{origin}')
    protocol=st.selectbox('Shared loading validation protocol',PROTOCOLS,key=f'pooled_loading_protocol_{origin}')
    folds=st.number_input('Shared loading fold budget',2,6,3,key=f'pooled_loading_folds_{origin}')
    physics=render_physics_options(f'pooled_loading_physics_{origin}')
    if not targets:
        st.info('Select one or more elements to inspect eligibility and run the shared model.')
        return
    eligible_count,element_count,audit=cached_pooled_eligibility(json.dumps(
        {'samples':samples,'origin':origin,'elements':targets,'profile':profile},sort_keys=True))
    cols=st.columns(3)
    cols[0].metric('Shared ICP input records',len(samples))
    cols[1].metric('Quantified ICP target pairs',eligible_count)
    cols[2].metric('Elements with quantified ICP',element_count)
    with st.expander('Shared ICP input eligibility'):
        st.dataframe(audit,hide_index=True,width='stretch')
    payload=json.dumps({'samples':samples,'origin':origin,'elements':targets,'family':family,
        'profile':profile,'protocol':protocol,'max_folds':int(folds),'physics':physics},sort_keys=True)
    signature=fingerprint(payload)
    if st.button('Validate shared ICP model',disabled=not physical_comparison_ready(physics),key=f'pooled_loading_run_{origin}'):
        with st.spinner('Fitting shared target-aware models inside protected batch folds...'):
            st.session_state[f'pooled_loading_result_{origin}']=(signature,cached_pooled_loading(payload))
    saved=st.session_state.get(f'pooled_loading_result_{origin}')
    if not saved or saved[0]!=signature:
        st.info('Run shared validation to predict all eligible selected sample-element pairs. Changing selection or settings invalidates an old report.')
        return
    points,audit,info=saved[1]
    metrics=info['metrics']
    cols=st.columns(4)
    for col,label,value in zip(cols,['Shared held-out ICP pairs','Shared ICP MAE / wt%','Shared ICP RMSE / wt%','Shared ICP R squared'],
        [metrics['n'],metrics['mae_wt_percent'],metrics['rmse_wt_percent'],metrics['r_squared']]):
        col.metric(label,f'{value:.4f}' if isinstance(value,float) else value if value is not None else 'Unavailable')
    st.caption(f"{metrics['physical_specimens']} physical specimens predicted. Equal-element MAE: {metrics['equal_element_mae_wt_percent']}. Batch masks keep every member of each sample together. Pooled R-squared includes between-element loading differences; inspect the per-element results as well.")
    if len(points):
        limit=max(.01,float(points[['observed_w_wt_percent','predicted_w_wt_percent']].max().max())*1.08)
        members=points.element.drop_duplicates().tolist()
        values=alt.Data(values=points.astype(object).where(pd.notna(points),None).to_dict('records'))
        dots=alt.Chart(values).mark_point(filled=True,size=65,opacity=.7).encode(
            x=alt.X('predicted_w_wt_percent:Q',title='Predicted element W / wt%',scale=alt.Scale(domain=[0,limit])),
            y=alt.Y('observed_w_wt_percent:Q',title='Measured element W / wt%',scale=alt.Scale(domain=[0,limit])),
            color=alt.Color('element:N',legend=alt.Legend(orient='bottom',columns=5,symbolLimit=len(members))),
            tooltip=['sample_id:N','batch_id:N','element:N','composition:N','fold:N','observed_w_wt_percent:Q',
                'predicted_w_wt_percent:Q','target_seen_in_training:N','outside_training_range:N'])
        ideal=alt.Chart(pd.DataFrame({'v':[0,limit]})).mark_line(strokeDash=[4,4],color='#8a98a6').encode(x='v:Q',y='v:Q')
        st.altair_chart((dots+ideal).properties(height=380+20*((len(members)+4)//5)),width='stretch')
        st.dataframe(points,hide_index=True,width='stretch')
        st.download_button('Download shared ICP predictions',points.to_csv(index=False).encode(),
            f'{origin.lower()}_shared_icp_predictions.csv','text/csv',key=f'pooled_loading_csv_{origin}')
    else:
        st.info('No evaluable shared folds. Inspect training sample/batch/recipe gates and exclusions.')
    st.write('**Per-element results from the shared model**')
    st.dataframe(pd.DataFrame(info['per_element']),hide_index=True,width='stretch')
    if info['paired_comparison']:
        st.dataframe(pd.DataFrame(info['paired_comparison']),hide_index=True,width='stretch')
    with st.expander('Shared ICP coverage and provenance'):
        st.dataframe(audit,hide_index=True,width='stretch')
        st.download_button('Download shared ICP provenance',json.dumps(info,indent=2,allow_nan=False).encode(),
            f'{origin.lower()}_shared_icp_provenance.json','application/json',key=f'pooled_loading_provenance_{origin}')
