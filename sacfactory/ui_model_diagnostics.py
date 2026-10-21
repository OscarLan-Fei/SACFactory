"""Detailed predictive comparisons and independently defined outcome-score parity."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from .outcome_index import score_pairs,shared_score_support
from .validation_diagnostics import regression_summary,parity_data,common_rows,probability_diagnostics
from .ui_elements import element_multiselect


def inline_data(frame):
    """Preserve every diagnostic row, including selections above 5,000 points."""
    return alt.Data(values=frame.astype(object).where(pd.notna(frame),None).to_dict('records'))


def parity_plot(frame,summary,predicted,observed,label,limits=None):
    values=parity_data(frame,summary,predicted,observed,limits)
    if values.empty:return None
    data=inline_data(values)
    base=alt.Chart(data)
    x=alt.X('prediction:Q',title='Predicted '+label,scale=alt.Scale(domain=list(limits)) if limits else alt.Undefined)
    y=alt.Y('measurement:Q',title='Measured '+label,scale=alt.Scale(domain=list(limits)) if limits else alt.Undefined)
    points=base.transform_filter(alt.datum.kind=='Observation').mark_point(filled=True,size=38,opacity=.6).encode(x=x,y=y,
        color=alt.Color('recorded_xrd:N',title='Recorded XRD',scale=alt.Scale(domain=['PASS','FAIL','Unknown'],range=['#268d88','#d46a27','#71808c'])),
        tooltip=['sample_id:N','batch_id:N','element:N','prediction:Q','measurement:Q','recorded_xrd:N'])
    identity=base.transform_filter(alt.datum.kind=='Identity').mark_line(strokeDash=[5,4],color='#8795a5').encode(x=x,y=y)
    fitted=base.transform_filter(alt.datum.kind=='OLS').mark_line(color='#7951a1',clip=True).encode(x=x,y=y)
    chart=(points+identity+fitted).properties(height=300,width=360)
    if frame.model.nunique()>1:
        return chart.facet(facet=alt.Facet('model:N',title=None),columns=2).resolve_scale(x='shared',y='shared')
    return chart.properties(width='container',height=390)


def render_outcome_score(points,origin,key):
    st.subheader('Composite outcome score: predicted vs measured')
    cols=st.columns(2)
    weight=cols[0].number_input('ICP weight in outcome score',min_value=0.,max_value=1.,value=.8,step=.05,key=key+'_score_weight')
    reference=cols[1].number_input('Loading reference / wt%',min_value=.001,max_value=100.,value=2.5,format='%.3f',key=key+'_score_reference')
    scores=shared_score_support(score_pairs(points,float(weight),float(reference)))
    common=scores.loc[scores.comparison_common]
    summary=regression_summary(common,'predicted_outcome_score','observed_outcome_score')
    paired=common.dropna(subset=['predicted_outcome_score','observed_outcome_score'])
    if len(paired):
        if paired.model.nunique()==1:
            row=summary.iloc[0]
            for col,label,value in zip(st.columns(4),['Score pairs','Score MAE / points','Score RMSE / points','Score predictive R²'],
                [int(row['n']),row.mae,row.rmse,row.predictive_r2]):
                col.metric(label,f'{value:.4f}' if isinstance(value,float) and pd.notna(value) else str(value) if pd.notna(value) else 'Unavailable')
        chart=parity_plot(paired,summary,'predicted_outcome_score','observed_outcome_score','outcome score / 100',(0.,100.))
        st.altair_chart(chart,width='stretch')
        st.caption('Purple: descriptive linear fit. Dashed: y = x. Correctly predicted failures lie near the identity line, even when both outcome scores are low.')
        if paired.model.nunique()==1 and pd.notna(summary.iloc[0].fit_slope):
            row=summary.iloc[0]
            st.write(f"Linear fit: y = {row.fit_slope:.4f}x {row.fit_intercept:+.4f}; fit R² = {row.fit_r2:.4f}" if pd.notna(row.fit_r2) else f"Linear fit: y = {row.fit_slope:.4f}x {row.fit_intercept:+.4f}")
        st.dataframe(summary,hide_index=True,width='stretch')
    else:st.info('A score pair requires quantified whole-recipe ICP and an XRD result, together with held-out predictions for both endpoints.')
    st.caption(f'{len(paired)} paired model-sample scores; {len(scores)-len(paired)} unavailable or outside common support. Multi-model fits use identical specimens. This is a result index, not percent accuracy or a single-atom purity measurement. Score settings change this display without retraining.')
    with st.expander('Outcome score definition'):
        st.latex(r'g(W)=\frac{W}{W+k}\frac{100+k}{100},\qquad S_{obs}=100[a\,g(W_{obs})+(1-a)y_{XRD}],\qquad S_{pred}=100[a\,g(\widehat W)+(1-a)p_{XRD}]')
        st.caption('W is total whole-recipe ICP in wt%; k is a reference scale, not a target or cutoff. The same fixed mapping is used on both sides. Predictions use held-out ICP means and XRD probabilities; this plug-in score is not a draw-integrated expected score. Priors enter the model, not the observed score. Missing endpoints and unphysical totals are excluded without reweighting.')
    st.download_button('Download outcome score pairs',scores.to_csv(index=False).encode(),f'{origin.lower()}_outcome_scores.csv','text/csv',key=key+'_score_pairs')
    st.download_button('Download score regression parameters',summary.to_csv(index=False).encode(),f'{origin.lower()}_score_regression.csv','text/csv',key=key+'_score_fits')
    return scores,summary


def render_model_diagnostics(points,origin,models=None,key_prefix='comparison',include_score=True):
    if points.empty:return
    elements=list(dict.fromkeys(points.element))
    selected=element_multiselect('Elements in comparison plots',elements,default=elements,key=f'{key_prefix}_plot_elements_{origin}')
    # Compute the shared support before display-only filtering or score aggregation.
    aligned=common_rows(points,models)
    frame=aligned.loc[aligned.element.isin(selected)].copy()
    if frame.empty:st.info('Select elements to display the diagnostics.');return
    frame['recorded_xrd']=frame.xrd_pass_observed.map({1:'PASS',0:'FAIL'}).fillna('Unknown')
    tabs=st.tabs(['Regression fits','Residuals and element errors','XRD probabilities']+(['Composite score fits'] if include_score else []))
    regression,residuals,classification=tabs[:3]
    with regression:
        st.subheader('ICP regression fits by model')
        summary=regression_summary(frame,'predicted_w_wt_percent','observed_w_wt_percent')
        chart=parity_plot(frame,summary,'predicted_w_wt_percent','observed_w_wt_percent','ICP W / wt%')
        if chart is not None:st.altair_chart(chart,width='stretch')
        st.caption('Identical paired rows across models. Purple: weighted OLS fit; dashed: y = x. Predictive R² evaluates the original predictions; fit R² evaluates the descriptive fitted line. Each specimen has equal metric weight, with equal member weight within a specimen.')
        st.dataframe(summary,hide_index=True,width='stretch')
        st.download_button('Download all regression parameters',summary.to_csv(index=False).encode(),f'{origin.lower()}_model_regression.csv','text/csv',key=f'{key_prefix}_regression_csv_{origin}')
    with residuals:
        numeric=frame.dropna(subset=['predicted_w_wt_percent','observed_w_wt_percent']).copy()
        if len(numeric):
            numeric['residual']=numeric.observed_w_wt_percent-numeric.predicted_w_wt_percent
            numeric['absolute_error']=numeric.residual.abs()
            base=alt.Chart(inline_data(numeric))
            dots=base.mark_circle(size=35,opacity=.55).encode(x=alt.X('predicted_w_wt_percent:Q',title='Predicted ICP / wt%'),
                y=alt.Y('residual:Q',title='Measured minus predicted / wt%'),color='recorded_xrd:N',tooltip=['sample_id:N','element:N','model:N','residual:Q'])
            zero=base.transform_aggregate(groupby=['model']).mark_rule(color='#8795a5',strokeDash=[4,4]).encode(y=alt.datum(0))
            st.altair_chart((dots+zero).properties(height=230,width=360).facet(facet=alt.Facet('model:N',title=None),columns=2),width='stretch')
            box=alt.Chart(inline_data(numeric)).mark_boxplot(extent=1.5).encode(x=alt.X('absolute_error:Q',title='Absolute ICP error / wt%'),y='model:N',color=alt.Color('model:N',legend=None)).properties(height=alt.Step(34))
            st.altair_chart(box,width='stretch')
            element_table=regression_summary(numeric,'predicted_w_wt_percent','observed_w_wt_percent',('model','element'))
            st.write('**Per-element regression statistics**')
            st.dataframe(element_table,hide_index=True,width='stretch')
            st.download_button('Download per-element errors',element_table.to_csv(index=False).encode(),f'{origin.lower()}_per_element_errors.csv','text/csv',key=f'{key_prefix}_elements_csv_{origin}')
        else:st.info('No common paired numeric ICP rows.')
    with classification:
        metrics,calibration,curves=probability_diagnostics(frame)
        st.dataframe(metrics,hide_index=True,width='stretch')
        if len(calibration):
            base=alt.Chart(calibration)
            interval=base.mark_rule(opacity=.4).encode(x='predicted_probability:Q',y='lower:Q',y2='upper:Q',color='model:N')
            dots=base.mark_line(point=True).encode(x=alt.X('predicted_probability:Q',title='Predicted XRD PASS probability',scale=alt.Scale(domain=[0,1])),
                y=alt.Y('observed_fraction:Q',title='Observed XRD PASS fraction',scale=alt.Scale(domain=[0,1])),color='model:N',tooltip=['model','n','predicted_probability','observed_fraction'])
            ideal=alt.Chart(pd.DataFrame({'v':[0.,1.]})).mark_line(strokeDash=[4,4],color='#8795a5').encode(x='v:Q',y='v:Q')
            st.altair_chart((interval+dots+ideal).properties(height=330),width='stretch')
            st.caption('Ten fixed probability bins; bars are approximate 95% Wilson intervals. Each physical specimen contributes once. Both-class test outcomes are required for ROC-AUC and precision-recall summaries.')
        if len(curves):
            for name,labels in [('ROC',('False positive rate','True positive rate')),('Precision-recall',('Recall','Precision'))]:
                part=curves.loc[curves.curve==name]
                curve=alt.Chart(inline_data(part)).mark_line().encode(x=alt.X('x:Q',title=labels[0],scale=alt.Scale(domain=[0,1])),y=alt.Y('y:Q',title=labels[1],scale=alt.Scale(domain=[0,1])),color='model:N',order=alt.Order('order:Q')).properties(height=300,title=name)
                st.altair_chart(curve,width='stretch')
        st.download_button('Download probability metrics',metrics.to_csv(index=False).encode(),f'{origin.lower()}_xrd_metrics.csv','text/csv',key=f'{key_prefix}_probability_csv_{origin}')
        st.download_button('Download calibration bins',calibration.to_csv(index=False).encode(),f'{origin.lower()}_calibration_bins.csv','text/csv',key=f'{key_prefix}_calibration_csv_{origin}')
        st.download_button('Download ROC and precision-recall curves',curves.to_csv(index=False).encode(),f'{origin.lower()}_probability_curves.csv','text/csv',key=f'{key_prefix}_curves_csv_{origin}')
    if include_score:
        with tabs[3]:render_outcome_score(frame,origin,f'{key_prefix}_outcome_{origin}')
