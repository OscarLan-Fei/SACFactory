"""Read-only same-recipe comparisons and saved-forecast follow-up charts."""
import copy

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .forecast_review import compare_batch, follow_up, metrics
from .planning import fingerprint
from .ui_elements import element_multiselect
from .domain import ELEMENTS


def render_batch_comparison(rows, store, samples, key):
    from .planning_models import options
    choices = options(store.root, store.origin)
    if len(choices) < 2:
        return
    prefix = f'forecast_compare_{store.origin}_{key}'
    with st.expander('Compare saved models on these recipes'):
        defaults = [name for name in choices if name.startswith('Pretrained ')] or list(choices)[:2]
        if prefix+'_models' in st.session_state:
            st.session_state[prefix+'_models'] = [n for n in st.session_state[prefix+'_models'] if n in choices]
        kwargs = {} if prefix+'_models' in st.session_state else {'default':defaults}
        names = st.multiselect('Models for the same-recipe comparison', list(choices), key=prefix+'_models', **kwargs)
        signature = fingerprint([[(r['recommendation_id'], r['recipe']) for r in rows],
                                 {n: choices[n]['model_id'] for n in names}, samples])
        result = st.session_state.get(prefix+'_result')
        if result and result[0] != signature:
            st.session_state.pop(prefix+'_result', None)
            result = None
            for row in rows:
                row.pop('comparison_predictions', None)
        saved_ids = {r['recommendation_id'] for r in store.all('recommendation')}
        unreserved = all(r['recommendation_id'] not in saved_ids for r in rows)
        if st.button('Compare recipe forecasts', disabled=len(names)<2 or not unreserved, key=prefix+'_run'):
            try:
                with st.spinner('Forecasting the same recipes with each saved model...'):
                    captured, table = compare_batch(rows,store.root,store.origin,names,samples)
                # These are immutable pre-result comparison forecasts when the
                # preview is subsequently enqueued. Existing records are untouched.
                for row in rows:
                    row['comparison_predictions'] = copy.deepcopy(captured[row['recommendation_id']])
                result = (signature, table)
                st.session_state[prefix+'_result'] = result
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                st.error(str(exc))
        st.caption('Compares identical recipes without training. Model disagreement is a difference in predictions, not a calibrated uncertainty interval. Comparison forecasts are kept with the batch when added to Pending.')
        if not unreserved:
            st.caption('This batch is already saved. Its original comparison forecasts remain frozen.')
        if result:
            table = result[1]
            st.dataframe(table, hide_index=True, width='stretch')
            spread = table.groupby('recommendation_id',sort=False).agg(
                structural_probability_range=('p_xrd_pass',lambda x:float(x.max()-x.min())),
                total_loading_range_wt_percent=('predicted_total_wt_percent',lambda x:float(x.max()-x.min())))
            st.dataframe(spread.reset_index(), hide_index=True, width='stretch')
            st.download_button('Export model comparison CSV', table.to_csv(index=False).encode(),
                               'next_batch_model_comparison.csv','text/csv',key=prefix+'_export')


def render_follow_up(samples, origin, recommendations=()):
    st.subheader('Saved forecast follow-up')
    st.caption('Compare forecasts saved before result entry with recorded ICP and structural assessments. These calculations do not refit a model or alter observations.')
    elements = sorted({e for sample in samples for e in sample['elements']})
    chosen = element_multiselect('Follow-up elements', ELEMENTS, default=elements, key='followup_elements_'+origin)
    selected = [s for s in samples if set(s['elements']) & set(chosen)]
    pairs, structures, audit = follow_up(selected,origin,recommendations)
    if audit.empty:
        st.info('No saved SAC-Gen forecasts in this selection. Generate Next batch, optionally compare models, add to Pending, then return results. Historical imports can be evaluated through masked Weighted validation.')
        return
    names = {r.model_id: f'{r.model} ({str(r.model_id)[:8]})' for r in audit.itertuples() if r.model_id}
    model_key = 'followup_models_'+origin
    if model_key in st.session_state:
        st.session_state[model_key] = [m for m in st.session_state[model_key] if m in names]
    kwargs = {} if model_key in st.session_state else {'default':list(names)}
    chosen_models = st.multiselect('Saved model versions',list(names),format_func=names.get,key=model_key,**kwargs)
    audit = audit.loc[audit.model_id.isin(chosen_models) | audit.model_id.isna()]
    if len(pairs):
        pairs = pairs.loc[pairs.model_id.isin(chosen_models) & pairs.element.isin(chosen)]
        # Reweight the selected quantified members equally within each specimen.
        pairs = pairs.copy()
        pairs['specimen_weight'] = 1/pairs.groupby(['sample_id','model_id']).element.transform('count')
    if len(structures):
        structures = structures.loc[structures.model_id.isin(chosen_models)]
    verified = st.checkbox('Require verified local training separation',key='followup_separation_'+origin)
    if verified:
        if len(pairs):pairs = pairs.loc[pairs.overlap_verified]
        if len(structures):structures = structures.loc[structures.overlap_verified]
    st.caption('Public pretrained training membership is withheld. Its results describe saved-forecast agreement; they do not establish an independent test of the original training data. Local snapshot checks verify sample/batch separation only. Repeated recipes can still share information across batches.')
    st.caption('Each endpoint uses the latest matching forecast saved before that result was recorded, once per specimen and model version. Earlier or changed-control revisions remain in the audit.')
    summary = metrics(pairs,structures)
    a,b,c = st.columns(3)
    a.metric('Quantified specimens',pairs.sample_id.nunique() if len(pairs) else 0)
    b.metric('Structural specimens',structures.sample_id.nunique() if len(structures) else 0)
    c.metric('Evaluated model versions',len(summary))
    if len(summary):
        st.dataframe(summary,hide_index=True,width='stretch')
        st.caption('ICP errors use equal weight per specimen and equal weight per selected quantified element within a specimen. R-squared evaluates predictions against measurements; the displayed fitted line is a separate agreement diagnostic. Correct FAIL forecasts contribute to FAIL recall and Brier score.')
        st.download_button('Export forecast metrics CSV',summary.to_csv(index=False).encode(),
                           'saved_forecast_metrics.csv','text/csv',key='followup_metrics_'+origin)
    if len(pairs):
        chart_data = pairs.copy()
        chart_data['model_version'] = [names[m] for m in chart_data.model_id]
        lo = min(chart_data.predicted_wt_percent.min(), chart_data.observed_wt_percent.min())
        hi = max(chart_data.predicted_wt_percent.max(), chart_data.observed_wt_percent.max())
        if hi == lo:hi = lo+1
        points = alt.Chart(chart_data).mark_circle(size=50,opacity=.65).encode(
            x=alt.X('predicted_wt_percent:Q',title='Saved predicted ICP W / wt%'),
            y=alt.Y('observed_wt_percent:Q',title='Measured ICP W / wt%'),
            color=alt.Color('model_version:N',title='Model version'),
            tooltip=['sample_id:N','batch_id:N','element:N','model_version:N','predicted_wt_percent:Q','observed_wt_percent:Q'])
        reference = alt.Chart(pd.DataFrame({'x':[lo,hi],'y':[lo,hi]})).mark_line(color='#8293a5',strokeDash=[5,4]).encode(x='x:Q',y='y:Q')
        lines = [{'model_version':names[row['model_id']],'x':x,'y':row['fit_slope']*x+row['fit_intercept']}
                 for row in summary.to_dict('records') if pd.notna(row.get('fit_slope')) for x in (lo,hi)]
        chart = points+reference
        if lines:
            chart += alt.Chart(pd.DataFrame(lines)).mark_line().encode(x='x:Q',y='y:Q',color='model_version:N')
        st.write('**Saved prediction versus measured ICP**')
        with alt.data_transformers.enable('default',max_rows=None):
            st.altair_chart(chart.properties(height=390),width='stretch')
        st.download_button('Export ICP prediction pairs CSV',pairs.to_csv(index=False).encode(),
                           'saved_icp_pairs.csv','text/csv',key='followup_icp_'+origin)
    if len(structures):
        calibration = []
        for (name, ident), group in structures.groupby(['model','model_id'],sort=False):
            group = group.copy()
            group['bin'] = pd.cut(group.p_pass,np.linspace(0,1,6),include_lowest=True)
            for _, values in group.groupby('bin',observed=True):
                calibration.append({'model_version':names[ident],'mean_prediction':values.p_pass.mean(),
                                    'observed_pass_rate':values.label.mean(),'specimens':len(values),
                                    'batches':values.batch_id.nunique()})
        st.write('**Structural probability calibration**')
        st.caption('Each point aggregates a probability bin. Individual structural outcomes remain PASS=1 and FAIL=0.')
        chart = alt.Chart(pd.DataFrame(calibration)).mark_circle(size=90).encode(
            x=alt.X('mean_prediction:Q',title='Mean saved structural PASS probability',scale=alt.Scale(domain=[0,1])),
            y=alt.Y('observed_pass_rate:Q',title='Observed structural PASS fraction',scale=alt.Scale(domain=[0,1])),
            color='model_version:N',tooltip=['model_version:N','specimens:Q','batches:Q','mean_prediction:Q','observed_pass_rate:Q'])
        reference = alt.Chart(pd.DataFrame({'x':[0,1],'y':[0,1]})).mark_line(color='#8293a5',strokeDash=[5,4]).encode(x='x:Q',y='y:Q')
        st.altair_chart((chart+reference).properties(height=320),width='stretch')
        st.download_button('Export structural prediction pairs CSV',structures.to_csv(index=False).encode(),
                           'saved_structural_pairs.csv','text/csv',key='followup_structure_'+origin)
    with st.expander('Forecast eligibility and missing results',expanded=summary.empty):
        st.dataframe(audit,hide_index=True,width='stretch')
        st.download_button('Export forecast audit CSV',audit.to_csv(index=False).encode(),
                           'saved_forecast_audit.csv','text/csv',key='followup_audit_'+origin)
