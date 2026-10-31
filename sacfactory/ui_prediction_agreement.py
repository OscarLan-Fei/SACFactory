"""Post-result agreement diagnostics and XRD/STEM probability calibration."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def render(samples,points):
    st.write('**Per-sample prediction agreement**')
    st.caption('A post-result error score, not classification accuracy or synthesis quality. Its scale depends on the ICP error scale and endpoint weights. Correct PASS and FAIL forecasts are rewarded equally.')
    if samples.empty:
        return
    paired=samples.dropna(subset=['validation_score']).copy()
    if len(paired):
        paired['recorded_xrd']=paired.xrd_failure_observed.map({0:'PASS',1:'FAIL'}).fillna('Unknown')
        paired['sample_order']=np.arange(1,len(paired)+1)
        values=alt.Data(values=paired.astype(object).where(pd.notna(paired),None).to_dict('records'))
        chart=alt.Chart(values).mark_point(filled=True,size=40,opacity=.65).encode(
            x=alt.X('sample_order:Q',title='Evaluated sample index'),
            y=alt.Y('validation_score:Q',title='Prediction agreement score / 100',scale=alt.Scale(domain=[0,100])),
            color=alt.Color('recorded_xrd:N',title='Recorded XRD/STEM',scale=alt.Scale(domain=['PASS','FAIL','Unknown'],range=['#268D88','#D46A27','#71808C'])),
            tooltip=['sample_id:N','composition:N','batch_id:N','validation_score:Q','icp_agreement_score:Q',
                     'xrd_pass_agreement_score:Q','xrd_failure_probability:Q','recorded_xrd:N'])
        st.altair_chart(chart.properties(height=300),width='stretch')
        st.caption('The horizontal axis is display order, not a prediction, synthesis time or experimental condition. Use ICP parity and endpoint metrics to assess predictive performance.')
    columns=['sample_id','composition','batch_id','validation_score','baseline_agreement_score',
             'icp_agreement_score','xrd_pass_agreement_score','xrd_failure_probability',
             'xrd_failure_observed','missing_score_endpoints']
    st.dataframe(samples[columns],hide_index=True,width='stretch')
    binary=points.drop_duplicates(['fold','sample_id']).dropna(subset=['xrd_failure_probability','xrd_failure_observed']).copy() if len(points) else points
    st.write('**XRD/STEM failure probability calibration**')
    if not len(binary):
        st.info('No paired XRD/STEM probability predictions. Inspect XRD/STEM class counts and fold fitting gates.')
        return
    binary['probability_bin']=np.minimum((binary.xrd_failure_probability*10).astype(int),9)
    grouped=binary.groupby('probability_bin',as_index=False).agg(predicted_failure_probability=('xrd_failure_probability','mean'),
        observed_failure_fraction=('xrd_failure_observed','mean'),samples=('sample_id','size'))
    dots=alt.Chart(grouped).mark_point(filled=True,size=80).encode(
        x=alt.X('predicted_failure_probability:Q',title='Mean predicted XRD/STEM failure probability',scale=alt.Scale(domain=[0,1])),
        y=alt.Y('observed_failure_fraction:Q',title='Observed XRD/STEM failure fraction',scale=alt.Scale(domain=[0,1])),
        size=alt.Size('samples:Q',title='Samples in bin'),tooltip=['predicted_failure_probability:Q','observed_failure_fraction:Q','samples:Q'])
    ideal=alt.Chart(pd.DataFrame({'v':[0,1]})).mark_line(strokeDash=[4,4],color='#8a98a6').encode(x='v:Q',y='v:Q')
    st.altair_chart((dots+ideal).properties(height=300),width='stretch')
    st.caption(f'All {len(binary)} paired samples enter ten fixed probability bins. Each point summarizes a bin, not one experiment. Probability 1 is included in the final bin. This is a retrospective calibration diagnostic, not a recalibration fit.')
