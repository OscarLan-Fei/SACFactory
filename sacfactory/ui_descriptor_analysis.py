"""Chemical descriptor audit, recipe summaries and descriptive associations."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from .analysis import correlations
from .descriptor_analysis import descriptor_report, LABELS, association_availability
from .precursors import PROFILES, precursor_coverage
from .ui_charts import chart_page, matrix_chart, render_matrix, correlation_grid


def export(frame, label, name, origin):
    st.download_button(label, frame.to_csv(index=False).encode('utf-8-sig'),
                       f'{origin.lower()}_{name}.csv', 'text/csv', key=f'descriptors_{name}_{origin}')


def render_descriptor_analysis(samples, methods, origin, context=None, space=None, model_info=None):
    space, model_info = space or {}, model_info or {}
    engine = model_info.get('engine', space.get('model_engine', 'adaptive'))
    profile = model_info.get('descriptor_profile', space.get('descriptor_profile', 'chemistry_pulseicp') if engine != 'baseline' else 'periodic')
    st.write('**Chemical descriptor audit**')
    st.caption(f'Selected planning engine: {engine}. Descriptor configuration: {PROFILES[profile]}. This panel follows the cohort/batch filter above and does not change data or retrain the model.')
    report = descriptor_report(samples, methods, origin, profile)
    members, recipes = report['members'], report['recipes']
    if members.empty:
        st.info('No selected records in this data space.')
        return
    columns = st.columns(3)
    columns[0].metric('Elements in descriptor audit', members.element.nunique())
    columns[1].metric('Source-backed oxidation memberships', int(members.formal_oxidation_state.notna().sum()))
    columns[2].metric('Matched ionic-radius memberships', int(members.ionic_radius_pm.notna().sum()))
    if members.reference_state_assumed.any():
        st.caption(f'{int(members.reference_state_assumed.sum())} memberships use explicitly declared reference-state assumptions. Precursor formal charges and square-planar reference radii do not establish catalyst oxidation states or measured coordination.')
    st.caption(f'{len(members)} sample-element memberships in {len(recipes)} selected records. A membership is not an independent synthesis; weighted summaries require complete positive concentrations.')
    cov, summaries, relationships, mixing = st.tabs(['Descriptor coverage', 'Recipe chemistry', 'Chemical associations', 'Mixing thermodynamics'])
    with mixing:
        from .ui_mixing import render_mixing
        render_mixing(samples, recipes, origin, profile)
    with cov:
        heat = report['coverage']
        elements = heat.element.drop_duplicates().to_frame()
        shown = chart_page(elements, f'descriptor_coverage_page_{origin}')['element'].tolist()
        visible = heat.loc[heat.element.isin(shown)]
        chart = matrix_chart(visible, 'descriptor', 'element', 'available_fraction', heat.descriptor.drop_duplicates().tolist(), shown, 'Available fraction', domain=[0, 1])
        render_matrix(chart, len(shown), f'descriptor_coverage_{origin}')
        st.caption('Descriptors retain their physical units and source coverage. Missing or constant values are excluded from correlations.')
        with st.expander('Enabled descriptors, definitions and sources'):
            st.dataframe(report['manifest'], hide_index=True, width='stretch')
            export(report['manifest'], 'Download descriptor definitions', 'descriptor_definitions', origin)
        with st.expander('Recorded element-level inputs'):
            st.dataframe(members, hide_index=True, width='stretch')
            st.dataframe(report['elemental'], hide_index=True, width='stretch')
        export(heat, 'Download chemical coverage CSV', 'chemical_coverage', origin)
        export(members, 'Download recorded descriptor inputs', 'recorded_descriptor_inputs', origin)
        if context:
            with st.expander('Saved planning chemistry, separate from recorded-data coverage'):
                st.dataframe(pd.DataFrame(precursor_coverage(context)), hide_index=True, width='stretch')
                st.caption('These future planning declarations are not inserted into historical samples or the coverage counts above.')
    with summaries:
        descriptor = st.selectbox('Recipe chemistry summary', list(LABELS), format_func=LABELS.get, key=f'descriptor_summary_{origin}')
        cols = ['sample_id', 'batch_id', 'composition', 'element_count', 'status', 'complete_concentrations', 'total_concentration_mM'] + [descriptor + suffix for suffix in ['_mean', '_sd', '_coverage']]
        st.dataframe(recipes[cols], hide_index=True, width='stretch')
        points = recipes.dropna(subset=[descriptor + '_mean'])
        if not points.empty:
            chart = alt.Chart(points).mark_circle(size=65, opacity=.65).encode(
                x=alt.X('element_count:Q', title='Elements per recipe', scale=alt.Scale(zero=False)),
                y=alt.Y(descriptor + '_mean:Q', title='Weighted ' + LABELS[descriptor], scale=alt.Scale(zero=False)),
                color='status:N', tooltip=['sample_id:N', 'batch_id:N', 'composition:N', descriptor + '_mean:Q', descriptor + '_sd:Q', descriptor + '_coverage:Q'])
            st.altair_chart(chart.properties(height=280), width='stretch')
        else:
            st.info('This summary is unavailable: supply recorded chemistry and complete concentrations where applicable.')
        export(recipes, 'Download recipe chemistry summaries', 'recipe_chemistry', origin)
    with relationships:
        candidates = [key + '_mean' for key in LABELS] + ['element_count', 'total_concentration_mM', 'label']
        variables = st.multiselect('Chemical association variables', candidates,
            default=['en_pauling_mean', 'covalent_radius_cordero_mean', 'element_count', 'label'], key=f'descriptor_associations_{origin}')
        availability = association_availability(recipes, variables)
        if not availability.empty and (availability.status != 'Available').any():
            st.info('Some selected descriptors have no usable variation. Gray cells are unavailable, not zero correlations.')
            st.dataframe(availability.loc[availability.status != 'Available'], hide_index=True, width='stretch')
        if len(variables) >= 2:
            pairs = correlations(recipes, variables)
            grid = correlation_grid(pairs, variables)
            render_matrix(matrix_chart(grid, 'parameter_1', 'parameter_2', 'spearman_rho', variables, variables, 'Spearman rho', 'redblue', [-1, 1]), len(variables), f'chemical_associations_{origin}')
            st.dataframe(pairs, hide_index=True, width='stretch')
            export(pairs, 'Download chemical associations', 'chemical_associations', origin)
    st.download_button('Download descriptor audit provenance', json.dumps(report['provenance'], indent=2).encode(),
                       f'{origin.lower()}_descriptor_audit.json', 'application/json', key=f'descriptor_audit_json_{origin}')
