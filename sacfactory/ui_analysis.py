"""Interactive descriptive analysis of experimental conditions and saved outcomes."""
import altair as alt
import pandas as pd
import streamlit as st

from .analysis import PARAMETERS, experiment_frame, metrics, grouped_rates, parameter_rates, correlations
from .domain import cohort_key, screen, ELEMENTS
from .outcome_analysis import enriched_frame, include_elements, variable_groups, variable_label, FAMILIES
from .planning import fingerprint
from .descriptor_analysis import association_availability
from .ui_charts import category_axis, row_height, correlation_grid, matrix_chart, render_matrix, chart_page
from .ui_elements import element_selectbox, element_multiselect


def download(frame, label, filename, key):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), filename, mime="text/csv", key=key)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_analysis_frame(samples, methods, origin):
    """Cache only data; any recorded edit or namespace change changes the key."""
    return enriched_frame(samples, methods, origin)


def render_analysis(samples, methods, origin, recommendations=(), planning_context=None, planning_space=None, model_info=None, store=None):
    from .ui_navigation import preserve_analysis_controls
    preserve_analysis_controls()
    st.subheader("Data analysis")
    st.caption("Quick mathematics on recorded conditions and screening outcomes. No ICP assay or XRD/STEM spectrum analysis. No data are changed.")
    if not samples:
        st.info("No experiments yet. Save a batch under Samples & data to begin analysis. The separate DEMO space can preview this page.")
        return
    from .ui_data_selection import select_experiments
    samples=select_experiments(samples,origin)
    if not samples:
        st.info('No experiments match the shared calculation selection.')
        return
    groups = {}
    for sample in samples:
        groups.setdefault(cohort_key(sample), []).append(sample)
    choices = list(groups)
    scope = st.radio("Analysis scope", ["One comparable cohort", "All cohorts: overview only", "Cross-cohort exploratory associations"], horizontal=True, key=f"analysis_scope_{origin}")
    comparable = scope == "One comparable cohort"
    if comparable:
        cohort_widget=f'analysis_cohort_{origin}'
        if cohort_widget in st.session_state and st.session_state[cohort_widget] not in range(len(choices)):
            st.session_state[cohort_widget]=0
        def describe(index):
            key = choices[index]
            members = groups[key]
            elements, precursor, support, protocol, basis = key
            symbols = ", ".join(elements) if len(elements) <= 6 else ", ".join(elements[:4]) + ", ..."
            return f"{len(elements)} elements ({symbols}) | {precursor} / {support} / {protocol} | {basis[0]} | {len(members)} records"
        selected = st.selectbox("Comparable composition and process", range(len(choices)), format_func=describe, key=f"analysis_cohort_{origin}")
        selected_samples = groups[choices[selected]]
        st.caption("Same target elements, precursor system, support, protocol and temperature-control basis. Conditions may vary within this cohort.")
    else:
        selected_samples = samples
        st.caption("Cross-cohort totals and associations are exploratory. Composition, support, protocol and batch differences can confound relationships; select one comparable cohort for matched process analysis.")
    batches = sorted({s.get("batch_id") or "UNASSIGNED" for s in selected_samples})
    batch_widget=f'analysis_batches_{origin}'
    if batch_widget in st.session_state:
        st.session_state[batch_widget]=[v for v in st.session_state[batch_widget] if v in batches]
    chosen = st.multiselect("Filter by batch (empty = all)", batches, key=batch_widget)
    selected_samples = [s for s in selected_samples if not chosen or (s.get("batch_id") or "UNASSIGNED") in chosen]
    if not selected_samples:
        st.info("No records match this filter.")
        return
    # Reuse the workbench's existing screening rule; this page never interprets assays.
    outcomes = {s["sample_id"]: screen(s, methods) for s in selected_samples}
    frame = experiment_frame(selected_samples, outcomes)
    info = metrics(frame)
    columns = st.columns(4)
    columns[0].metric("Selected records", info["samples"])
    columns[1].metric("Resolved outcomes", info["labeled"])
    columns[2].metric("Candidate pass rate", f"{info['pass_rate']:.1%}" if info["pass_rate"] is not None else "Unavailable")
    columns[3].metric("Distinct complete recipes", info["distinct_recipes"])
    st.caption(f"{info['passed']} passed / {info['failed']} failed; {info['pending']} pending; {info['conflicts']} conflicts excluded. Pass rate uses resolved, nonconflicting outcomes only. Screening PASS is a candidate label; STEM remains separate.")
    st.caption('Correlation analysis: open Correlations & coverage below. Outcome Maps contains the two-parameter interaction map.')
    overview, trends, associations, chemical, outcome_chemistry, loading, campaign, weighted, benchmarks = st.tabs(["Batch overview", "Parameter relationships", "Correlations & coverage", "Chemical descriptors", "Success & failure", "ICP loading prediction", "Element campaign", "Weighted validation", "Model comparison"],
        default='Correlations & coverage', key=f'analysis_views_{origin}', on_change='rerun')
    chemical_frame = cached_analysis_frame(selected_samples, methods, origin)
    if benchmarks.open:
        with benchmarks:
            from .ui_benchmarks import render_benchmarks
            render_benchmarks(samples,methods,origin)
    if weighted.open:
        with weighted:
            from .ui_weighted_validation import render_weighted_validation
            render_weighted_validation(samples,methods,origin,store.root if store is not None else None)
    if campaign.open:
        with campaign:
            from .ui_element_campaign import render_element_campaign
            render_element_campaign(samples, methods, origin, store)
    if loading.open:
        with loading:
            from .ui_loading import render_loading
            render_loading(selected_samples, origin, store.root if store is not None else None, all_samples=samples)
    if outcome_chemistry.open:
        with outcome_chemistry:
            from .ui_outcomes import render_outcome_comparison
            render_outcome_comparison(chemical_frame, origin, comparable)
    if chemical.open:
        with chemical:
            from .ui_descriptor_analysis import render_descriptor_analysis
            render_descriptor_analysis(selected_samples, methods, origin, planning_context, planning_space, model_info)
    if overview.open:
        with overview:
            rates = grouped_rates(frame, "batch_id")
            st.write("**Success rate by batch**")
            if rates["labeled"].sum():
                chart_data = chart_page(rates.loc[rates["labeled"] > 0], f"batch_chart_page_{origin}")
                bars = alt.Chart(chart_data).mark_bar(color="#097f7a", size=22).encode(
                    y=alt.Y("batch_id:N", title="Batch", sort=None, axis=category_axis()), x=alt.X("pass_rate:Q", title="Candidate pass rate", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    tooltip=["batch_id:N", "samples:Q", "labeled:Q", "passed:Q", "failed:Q", alt.Tooltip("pass_rate:Q", format=".1%")])
                intervals = alt.Chart(chart_data).mark_rule(color="#233e54", strokeWidth=2).encode(y=alt.Y("batch_id:N", sort=None, axis=category_axis()), x="lower_95:Q", x2="upper_95:Q")
                st.altair_chart((bars + intervals).properties(height=row_height(len(chart_data))), width="stretch")
            else:
                st.info("Pass rates become available when outcomes are resolved.")
            st.caption('Descriptive statistics use recorded observations; batch dependence can affect intervals and associations.')
            st.dataframe(rates, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("pass_rate", "lower_95", "upper_95")})
            left, right = st.columns(2)
            with left:
                st.write("**Recorded screening distribution**")
                st.dataframe(frame["status"].value_counts().rename_axis("status").reset_index(name="records"), hide_index=True, width="stretch")
            with right:
                st.write("**Optional STEM reports**")
                st.dataframe(frame["stem"].value_counts().rename_axis("reported_state").reset_index(name="records"), hide_index=True, width="stretch")
                st.caption("Reported states are displayed without changing screening labels or checking evidence files.")
            download(rates, "Download batch summary CSV", f"{origin.lower()}_batch_analysis.csv", f"analysis_rates_{origin}")
    if trends.open:
        with trends:
            if not comparable:
                st.info("Select One comparable cohort above to view parameter relationships.")
            else:
                elements = selected_samples[0]["elements"]
                element = element_selectbox("Element concentration to analyze", elements, key=f"analysis_element_{origin}_{selected_samples[0]['sample_id']}")
                concentration = f"{element}_mM"
                labels = {concentration: f"{element} precursor / mM", **PARAMETERS}
                parameter = st.selectbox("Parameter to summarize", list(labels), format_func=labels.get, key=f"analysis_parameter_{origin}_{element}")
                trend = parameter_rates(frame, parameter)
                if frame[parameter].notna().sum():
                    y_parameter = st.selectbox("Plot vertical axis", list(PARAMETERS), format_func=PARAMETERS.get, key=f"analysis_y_{origin}")
                    points = frame.dropna(subset=[parameter, y_parameter])
                    if not points.empty:
                        palette = {"PASS": "#097f7a", "ICP_FAIL": "#df923b", "XRD_FAIL": "#c55763", "BOTH_FAIL": "#8d4055", "BURNED_OUT": "#a62639", "RETEST": "#7a68a6", "PENDING": "#8a98a6"}
                        states = [status for status in palette if status in set(points["status"])]
                        chart = alt.Chart(points).mark_point(filled=True, size=85, opacity=.75).encode(
                            x=alt.X(f"{parameter}:Q", title=labels[parameter], scale=alt.Scale(zero=False)),
                            y=alt.Y(f"{y_parameter}:Q", title=PARAMETERS[y_parameter], scale=alt.Scale(zero=False)),
                            color=alt.Color("status:N", title="Recorded outcome", scale=alt.Scale(domain=states, range=[palette[status] for status in states])), shape=alt.Shape("conflict:N", title="Conflicting record"),
                            tooltip=["sample_id:N", "batch_id:N", "status:N", f"{parameter}:Q", f"{y_parameter}:Q"])
                        st.altair_chart(chart.properties(height=280), width="stretch")
                    else:
                        st.info("No paired observations for these axes; select another vertical axis.")
                    st.write("**Pass rates across the selected parameter**")
                    st.dataframe(trend, hide_index=True, width="stretch")
                    st.caption(f"{int(frame[parameter].isna().sum())} records missing this parameter. Up to five equal-frequency bins are used when there are many distinct values. Other conditions may co-vary; the table is descriptive, not a causal effect or a fitted optimum.")
                    download(trend, "Download parameter summary CSV", f"{origin.lower()}_{parameter}_analysis.csv", f"analysis_trend_{origin}")
                else:
                    st.info("This parameter has no recorded values in the current selection.")
    if associations.open:
        with associations:
            st.write("**Correlation analysis**")
            st.caption(f"{info['complete_conditions']} / {info['samples']} records have concentrations, pulse count and a temperature basis; {info['batches']} named batches. Completeness here describes data coverage, not model eligibility.")
            selected_elements = list(dict.fromkeys(e for sample in selected_samples for e in sample['elements']))
            recorded_elements = list(dict.fromkeys(e for sample in samples for e in sample['elements']))
            element_key = f"correlation_elements_{origin}_{fingerprint([s['sample_id'] for s in selected_samples])}"
            st.session_state.setdefault(element_key, selected_elements)
            if st.button('Include all recorded concentration elements', key=f'correlation_all_elements_{origin}'):
                st.session_state[element_key] = recorded_elements
            included_elements = element_multiselect('Concentration elements to include', ELEMENTS, key=element_key)
            st.caption('All 118 elements are selectable. Missing concentrations and elements absent from a recipe stay unavailable, never zero-filled. For multiple compositions choose Cross-cohort exploratory associations above. Recorded elements in this data space: ' + ', '.join(recorded_elements))
            association_frame = include_elements(chemical_frame, selected_samples, included_elements)
            association_groups = variable_groups(association_frame, included_elements)
            families = st.multiselect('Association variable groups', FAMILIES,
                default=None if f'correlation_families_{origin}' in st.session_state else FAMILIES[:4], key=f'correlation_families_{origin}')
            numeric_columns = list(dict.fromkeys(c for family in families for c in association_groups[family] if c in association_frame))
            coverage = pd.DataFrame([{'parameter': c, 'definition': variable_label(c), 'observed': int(association_frame[c].notna().sum()),
                'missing': int(association_frame[c].isna().sum()), 'distinct_values': int(association_frame[c].nunique()),
                'minimum': association_frame[c].min(), 'median': association_frame[c].median() if association_frame[c].notna().any() else None,
                'maximum': association_frame[c].max()} for c in numeric_columns])
            if comparable or scope == 'Cross-cohort exploratory associations':
                st.write('**Spearman rank associations**')
                available = numeric_columns + ['label']
                defaults = [c for c in [*[e + '_mM' for e in included_elements][:3], 'pulse_count', 'pulse_limit_C',
                    'total_concentration_mM', 'element_count', 'en_pauling_mean', 'covalent_radius_cordero_mean', 'label'] if c in available]
                selector_key = f"analysis_correlations_{origin}_{fingerprint([scope, [s['sample_id'] for s in selected_samples], included_elements, families])}"
                if selector_key in st.session_state:
                    st.session_state[selector_key] = [c for c in st.session_state[selector_key] if c in available]
                selected_columns = st.multiselect('Correlation variables', available, default=defaults if selector_key not in st.session_state else None, key=selector_key)
                availability = association_availability(association_frame, selected_columns)
                unavailable = availability.loc[availability.status != 'Available'] if not availability.empty else availability
                if not unavailable.empty:
                    st.info('Unavailable correlations: ' + '; '.join(f"{r.variable}: {r.status} ({r.observed} observations)" for r in unavailable.itertuples()))
                    st.dataframe(unavailable, hide_index=True, width='stretch')
                pairs = correlations(association_frame, selected_columns)
                if len(selected_columns) >= 2:
                    matrix = correlation_grid(pairs, selected_columns)
                    chart = matrix_chart(matrix, 'parameter_1', 'parameter_2', 'spearman_rho', selected_columns, selected_columns,
                        'Spearman rho', scheme='redblue', domain=[-1, 1])
                    render_matrix(chart, len(selected_columns), f'correlations_{origin}')
                    st.dataframe(pairs, hide_index=True, width='stretch')
                    download(pairs, 'Download correlations CSV', f'{origin.lower()}_correlations.csv', f'analysis_corr_{origin}')
            else:
                st.info('Correlations are available in matched-cohort or exploratory cross-cohort mode. Overview-only mode retains summaries without pooling correlation coefficients.')
                def open_associations():
                    st.session_state[f'analysis_scope_{origin}'] = 'Cross-cohort exploratory associations'
                st.button('Show cross-cohort correlations', on_click=open_associations, key=f'open_associations_{origin}')
            with st.expander('Variable coverage and missing values', expanded=False):
                st.dataframe(coverage, hide_index=True, width='stretch')
                download(coverage, "Download coverage CSV", f"{origin.lower()}_coverage.csv", f"analysis_coverage_{origin}")
    download(chemical_frame.drop(columns=["cohort", "recipe"]), "Download selected analysis records CSV", f"{origin.lower()}_analysis_records.csv", f"analysis_records_{origin}")
    from .ui_analysis_extended import render_extended
    render_extended(samples, selected_samples, methods, origin, comparable, recommendations)
