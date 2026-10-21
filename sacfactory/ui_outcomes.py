"""Descriptive PASS/FAIL distributions of recorded chemical and process variables."""
import altair as alt
import streamlit as st

from .analysis import parameter_rates
from .outcome_analysis import outcome_groups, variable_label
from .ui_charts import category_axis, row_height


def render_outcome_comparison(frame, origin, comparable):
    st.write('**Success and failure: chemical context**')
    st.caption('Observed outcomes and descriptive associations; unperformed tests are excluded.')
    metrics = [c for c in frame if c.endswith(('_mean', '_sd', '_fraction')) or c in {'total_concentration_mM', 'element_count', 'pulse_count', 'pulse_limit_C'} or (c.endswith('_mM') and c != 'total_concentration_mM')]
    if not metrics:
        st.info('No conditions or chemical summaries yet.')
        return
    metric = st.selectbox('Outcome comparison metric', metrics,
        index=metrics.index('en_pauling_mean') if 'en_pauling_mean' in metrics else 0,
        format_func=variable_label, key=f'outcome_chemical_metric_{origin}')
    branches = st.radio('Outcome grouping', ['Screen pass vs screen fail', 'Screening failure branches'],
                        horizontal=True, key=f'outcome_chemical_groups_{origin}') == 'Screening failure branches'
    resolved, summary = outcome_groups(frame, metric, branches)
    cols = st.columns(3)
    cols[0].metric('Compared screening outcomes', len(resolved))
    cols[1].metric('Excluded unresolved/conflicting records', len(frame) - len(resolved))
    cols[2].metric('Compared descriptor observations', int(resolved[metric].notna().sum()))
    if not comparable:
        pass
    if resolved.empty:
        st.info('Resolved, nonconflicting screening outcomes are required for a comparison.')
    else:
        if resolved.label.nunique() < 2:
            st.info('Only one screening class is present. The recorded distribution is shown; a success-versus-failure difference cannot be estimated.')
        points = resolved.dropna(subset=[metric])
        if points.empty:
            st.info('This metric has no recorded values in the selected outcomes. Historical precursor/source/state information is required where applicable; future reagent defaults do not fill this gap.')
        else:
            chart = alt.Chart(points).mark_boxplot(extent=1.5, color='#097f7a').encode(
                y=alt.Y('outcome_group:N', title=None, axis=category_axis()),
                x=alt.X(metric + ':Q', title=variable_label(metric), scale=alt.Scale(zero=False)))
            st.altair_chart(chart.properties(height=row_height(points.outcome_group.nunique())), width='stretch')
            st.caption('Boxes: median and interquartile range; whiskers follow the 1.5-IQR rule. Sample-level distributions are descriptive and may include dependent repeats.')
        st.dataframe(summary, hide_index=True, width='stretch')
        st.download_button('Download chemical outcome comparison', summary.to_csv(index=False).encode('utf-8-sig'),
                           f'{origin.lower()}_chemical_outcomes.csv', 'text/csv', key=f'chemical_outcome_export_{origin}')
        rates = parameter_rates(frame, metric)
        with st.expander('Observed pass rate across this chemical metric'):
            st.dataframe(rates, hide_index=True, width='stretch')
            st.download_button('Download chemical metric pass rates', rates.to_csv(index=False).encode('utf-8-sig'),
                               f'{origin.lower()}_chemical_metric_rates.csv', 'text/csv', key=f'chemical_metric_rates_{origin}')
