"""English analysis dashboard for recorded outcomes and model validation."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from .analysis import PARAMETERS
from .analysis_extended import (analysis_records, calibration_table, composition_coverage,
    conditional_surface, failure_breakdown, gate_summary, interaction_rates, learning_progress,
    prediction_records, probability_metrics, repeatability, strategy_comparison, transfer_comparison)
from .batching import PERIODS
from .composition import composition_name
from .domain import cohort_key, ELEMENTS
from .planning import fingerprint
from .prediction_plot import agreement_table
from .probability_evaluation import probability_evaluation
from .ui_research_validation import render_agreement, render_research_validation
from .ui_reliability import render_recipe_reliability
from .ui_chemistry import render_chemistry_analysis
from .ui_charts import category_axis, row_height, matrix_chart, render_matrix, chart_page, interaction_chart
from .ui_elements import element_selectbox, element_multiselect

COLORS = {'PASS': '#097f7a', 'ICP_FAIL': '#df923b', 'XRD_FAIL': '#c55763', 'BOTH_FAIL': '#8d4055', 'BURNED_OUT': '#a62639', 'RETEST': '#7a68a6', 'PENDING': '#8a98a6'}


def export(frame, name, origin):
    st.download_button(f'Download {name.replace("_", " ")} CSV', frame.to_csv(index=False).encode('utf-8'), f'{origin.lower()}_{name}.csv', 'text/csv', key=f'extended_export_{origin}_{name}')


@st.cache_data(show_spinner=False, max_entries=8)
def cached_surface(serialized, element, pulse):
    data = json.loads(serialized)
    return conditional_surface(data['samples'], data['methods'], data['context'], data['origin'], element, pulse, engine=data.get('engine', 'baseline'))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_transfer(serialized):
    return transfer_comparison(**json.loads(serialized))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_repeatability(frame):
    """Reuse exact-recipe summaries when only display filters change."""
    return repeatability(frame)


def probability_panel(records, origin):
    st.write('**Prediction Validation**')
    if records.empty:
        st.info('No saved predictions yet. Generate Next batch, import it into Pending, then report actual results in a new synthesis batch.')
        return
    if (records['policy'] == 'HUMAN_ADJUSTED_RECIPE').any():
        scope = st.radio('Forecast origin for validation', ['Algorithm-selected recipes', 'Human-adjusted recipes', 'All forecasts'],
                         horizontal=True, key=f'prediction_origin_{origin}')
        st.caption('Human-adjusted forecasts remain model predictions, but the experiment selection includes human intervention. Report this group separately when assessing an autonomous recommendation policy.')
        if scope != 'All forecasts':
            manual = records['policy'] == 'HUMAN_ADJUSTED_RECIPE'
            records = records.loc[manual if scope == 'Human-adjusted recipes' else ~manual]
    valid = records.loc[records['eligible']]
    score = probability_metrics(valid)
    columns = st.columns(3)
    columns[0].metric('Auditable prediction/result pairs', score['n'])
    columns[1].metric('Brier score', f"{score['brier']:.3f}" if score['brier'] is not None else 'Unavailable')
    columns[2].metric('Balanced accuracy', f"{score['balanced_accuracy']:.1%}" if score['balanced_accuracy'] is not None else 'Both classes required')
    if not valid.empty:
        calibration = calibration_table(records)
        view = st.radio('Prediction plot points', ['Individual samples', 'Probability bins', 'Repeated recipes'],
                        horizontal=True, key=f'prediction_plot_points_{origin}')
        if view == 'Individual samples':
            chart = alt.Chart(valid).mark_circle(size=65, opacity=.6).encode(
                x=alt.X('p_screen_pass:Q', title='Saved predicted screening probability', scale=alt.Scale(domain=[0, 1])),
                y=alt.Y('label:Q', title='Observed outcome: 0 = FAIL, 1 = PASS', scale=alt.Scale(domain=[-.08, 1.08]), axis=alt.Axis(values=[0, 1])),
                color=alt.Color('status:N', scale=alt.Scale(domain=['PASS', 'ICP_FAIL', 'XRD_FAIL', 'BOTH_FAIL', 'BURNED_OUT'], range=[COLORS[c] for c in ['PASS', 'ICP_FAIL', 'XRD_FAIL', 'BOTH_FAIL', 'BURNED_OUT']])),
                tooltip=['sample_id:N', 'batch_id:N', 'composition:N', 'model_id:N', 'p_screen_pass:Q', 'label:Q', 'status:N'])
            st.altair_chart(chart.properties(height=300), width='stretch')
            st.caption(f'All {len(valid)} eligible samples are plotted without binning or jitter. Identical prediction/outcome coordinates overlap. A single outcome is binary, not an observed success probability. Records lacking an eligible saved pre-result forecast remain in the audit below.')
            export(valid, 'individual_prediction_outcomes', origin)
        else:
            table = calibration if view == 'Probability bins' else agreement_table(records, by_stage=False, recipe_level=True)
            if table.empty:
                st.info('No eligible repeated-recipe points. Each recipe needs results from at least two independent synthesis batches.')
            else:
                points = alt.Chart(table).mark_circle(size=110, color='#097f7a').encode(x=alt.X('mean_prediction:Q', title='Mean saved predicted screening probability', scale=alt.Scale(domain=[0, 1])), y=alt.Y('observed_pass_rate:Q', title='Observed candidate pass rate', scale=alt.Scale(domain=[0, 1])), size=alt.Size('n:Q', title='Samples in point'), tooltip=list(table.columns))
                ideal = alt.Chart(pd.DataFrame({'x': [0., 1.], 'y': [0., 1.]})).mark_line(strokeDash=[4, 4], color='#8a98a6').encode(x='x:Q', y='y:Q')
                st.altair_chart((points + ideal).properties(height=300), width='stretch')
                st.caption(f'{len(table)} aggregated points represent {int(table["n"].sum())} eligible samples. Each point is a probability bin.' if view == 'Probability bins' else f'{len(table)} repeated recipes represent {int(table["n"].sum())} eligible samples. Each point is one exact controlled recipe repeated in at least two batches; singleton recipes are excluded.')
                st.dataframe(table, hide_index=True, width='stretch')
                if view == 'Repeated recipes':
                    export(table, 'repeated_recipe_prediction_outcomes', origin)
        with st.expander('Probability-bin calibration table', expanded=False):
            st.dataframe(calibration, hide_index=True, width='stretch')
        confusion = valid.assign(predicted=valid['p_screen_pass'].ge(.5).map({True: 'Predicted pass', False: 'Predicted fail'}), observed=valid['label'].map({1: 'Observed pass', 0: 'Observed fail'})).groupby(['predicted', 'observed']).size().reset_index(name='records')
        st.dataframe(confusion, hide_index=True, width='stretch')
        export(calibration, 'probability_calibration', origin)
        render_agreement(records, origin)
        with st.expander('Probability quality and decision trade-offs'):
            threshold = st.slider('Screening probability decision threshold', 0., 1., .5, .05, key=f'evaluation_threshold_{origin}')
            quality, curve = probability_evaluation(records, threshold)
            st.dataframe(pd.DataFrame([quality]), hide_index=True, width='stretch')
            if not curve.empty:
                chart = alt.Chart(curve).mark_line(color='#097f7a').encode(
                    x=alt.X('recall:Q', title='Screening PASS recall', scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y('precision:Q', title='Screening PASS precision', scale=alt.Scale(domain=[0, 1])), tooltip=['threshold:Q', 'recall:Q', 'precision:Q'])
                baseline = alt.Chart(pd.DataFrame({'prevalence': [quality['observed_prevalence']]})).mark_rule(color='#8a98a6', strokeDash=[4, 4]).encode(y='prevalence:Q')
                st.altair_chart((chart + baseline).properties(height=300), width='stretch')
            else:
                st.info('ROC AUC and average precision require both observed classes.')
            st.caption('False-pass fraction is the proportion of actual failures among predicted passes.')
            export(pd.DataFrame([quality]), 'probability_quality_metrics', origin)
    st.caption('Results are descriptive unless evaluated on held-out data. Methods and sample counts are included in the exports.')
    st.dataframe(records, hide_index=True, width='stretch')
    export(records, 'prediction_audit', origin)


def render_extended(samples, selected_samples, methods, origin, comparable, recommendations=()):
    st.divider()
    st.write('**Synthesis intelligence**')
    frame = analysis_records(selected_samples, methods)
    maps, repeats, composition, validation, reliability, chemistry = st.tabs(['Outcome Maps', 'Repeatability', 'Composition & Transfer', 'Model Validation', 'Recipe Reliability', 'Chemical Transfer'],
        key=f'analysis_extended_views_{origin}', on_change='rerun')
    if chemistry.open:
        with chemistry:
            render_chemistry_analysis(samples, selected_samples, methods, origin)
    key = f'{origin}_{fingerprint([s["sample_id"] for s in selected_samples])}'
    if reliability.open:
        with reliability:
            render_recipe_reliability(selected_samples, methods, origin, key)
    if maps.open:
        with maps:
            st.write('**Synthesis Outcome Map**')
            if comparable:
                members = selected_samples[0]['elements']
                element = element_selectbox('Map element concentration', members, key=f'map_element_{key}')
                map_unit = st.radio('Outcome map precursor unit', ['mM', 'mmol'], horizontal=True, key=f'map_unit_{key}')
                concentration = f'{element}_{map_unit}'
                temperature = st.selectbox('Map temperature basis', ['pulse_limit_C', 'temperature_C'], format_func=PARAMETERS.get, key=f'map_temperature_{key}')
                points = frame.dropna(subset=[concentration, temperature, 'pulse_count'])
                if points.empty:
                    st.info('Recorded concentrations, temperature and pulse count are required for an outcome map.')
                else:
                    chart = alt.Chart(points).mark_point(filled=True, size=80, opacity=.75).encode(x=alt.X(f'{concentration}:Q', title=f'{element} precursor / {map_unit}', scale=alt.Scale(zero=False)), y=alt.Y(f'{temperature}:Q', title=PARAMETERS[temperature], scale=alt.Scale(zero=False)), color=alt.Color('status:N', scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values()))), shape='conflict:N', tooltip=['sample_id:N', 'batch_id:N', 'status:N', 'sample_condition:N', f'{concentration}:Q', f'{temperature}:Q', 'pulse_count:Q']).properties(height=240, width=250).facet(column=alt.Column('pulse_count:O', title='Pulse count'))
                    with st.container(height=370, border=False, key='matrix_scroll_outcome_facets_' + key):
                        st.altair_chart(chart, width='content')
                st.write('**Parameter Interaction**')
                variables = {concentration: f'{element} precursor / {map_unit}', **PARAMETERS}
                x = st.selectbox('Interaction first parameter', list(variables), format_func=variables.get, key=f'interaction_x_{key}')
                y_options = [v for v in variables if v != x]
                y = st.selectbox('Interaction second parameter', y_options, index=y_options.index('pulse_limit_C') if 'pulse_limit_C' in y_options else 0, format_func=variables.get, key=f'interaction_y_{key}')
                interaction = interaction_rates(frame, x, y)
                if interaction.empty:
                    st.info('No resolved paired observations for these parameters.')
                else:
                    chart = interaction_chart(interaction, variables[x], variables[y])
                    render_matrix(chart, interaction['y_range'].nunique(), f'interaction_{origin}')
                    st.dataframe(interaction, hide_index=True, width='stretch')
                    export(interaction, 'parameter_interaction', origin)
                st.caption('Cells show observed pass fractions and sample counts. Blank cells have no observations.')
                st.write('**Success Boundary & Uncertainty**')
                pulses = sorted(frame['pulse_count'].dropna().astype(int).unique())
                if pulses:
                    pulse = st.selectbox('Conditional model slice: pulse count', pulses, key=f'surface_pulse_{key}')
                    map_engine = st.selectbox('Probability map engine', ['baseline', 'adaptive'], format_func={'baseline': 'Fixed-kernel baseline', 'adaptive': 'Adaptive hierarchical GP'}.get, key=f'map_engine_{key}')
                    payload = json.dumps({'samples': selected_samples, 'methods': methods, 'context': selected_samples[0], 'origin': origin, 'engine': map_engine}, sort_keys=True)
                    signature = fingerprint({'payload': payload, 'element': element, 'pulse': int(pulse)})
                    if st.button('Calculate conditional probability map', key=f'surface_run_{key}'):
                        with st.spinner('Fitting the existing screening classifier...'):
                            st.session_state[f'surface_result_{origin}'] = (signature, cached_surface(payload, element, int(pulse)))
                    saved = st.session_state.get(f'surface_result_{origin}')
                    if saved and saved[0] == signature:
                        surface, info, anchor = saved[1]
                        if surface.empty:
                            st.info(info['reason'])
                        else:
                            st.caption(f"Model: {info['model_id']}. Conditional anchor concentrations: {anchor['concentrations_mM']}. Fixed pulse: {pulse}. Grid spans observed coordinate ranges; some combinations can still lack nearby experiments.")
                            displays = ['p_screen_pass', 'predictive_entropy_bits', 'nearest_training_distance'] + ([name for name in ('bald_bits', 'posterior_interval_width') if name in surface])
                            view = st.selectbox('Probability map display', displays, format_func={'p_screen_pass': 'Predicted candidate pass probability', 'predictive_entropy_bits': 'Predictive entropy / bits', 'nearest_training_distance': 'Nearest training distance / kernel scales', 'bald_bits': 'Posterior information / bits', 'posterior_interval_width': '90% model interval width'}.get, key=f'surface_view_{key}')
                            chart = alt.Chart(surface).mark_square(size=110).encode(x=alt.X('concentration_mM:Q', title=f'{element} precursor / mM', scale=alt.Scale(type='log')), y=alt.Y('temperature_C:Q', title=PARAMETERS['pulse_limit_C' if anchor.get('pulse_limit_C') is not None else 'temperature_C'], scale=alt.Scale(zero=False)), color=alt.Color(f'{view}:Q', scale=alt.Scale(scheme='tealblues')), tooltip=list(surface.columns))
                            observed = points.loc[points['pulse_count'] == pulse] if not points.empty else points
                            controlled_column = 'pulse_limit_C' if anchor.get('pulse_limit_C') is not None else 'temperature_C'
                            observed = frame.loc[frame['pulse_count'] == pulse].dropna(subset=[concentration, controlled_column])
                            dots = alt.Chart(observed).mark_circle(size=35, color='#172f46', filled=False, strokeWidth=1.5).encode(x=alt.X(f'{concentration}:Q', scale=alt.Scale(type='log')), y=alt.Y(f'{controlled_column}:Q', scale=alt.Scale(zero=False)), tooltip=['sample_id:N', 'status:N'])
                            st.altair_chart((chart + dots).properties(height=300), width='stretch')
                            st.dataframe(surface.sort_values('predictive_entropy_bits', ascending=False).head(12), hide_index=True, width='stretch')
                            export(surface, 'conditional_model_surface', origin)
                    else:
                        st.caption('Calculate on demand. Edited data, changed cohort or changed slice invalidates the displayed result. No probabilities are invented before the existing training gate passes.')
            else:
                st.info('Select One comparable cohort to map synthesis conditions, interactions and a conditional model slice.')
            st.write('**Failure Breakdown**')
            failures = failure_breakdown(frame)
            batch_ids = failures['batch_id'].drop_duplicates().to_frame()
            shown_batches = chart_page(batch_ids, f'failure_chart_page_{origin}', size=16)['batch_id']
            shown_failures = failures.loc[failures['batch_id'].isin(shown_batches)]
            st.altair_chart(alt.Chart(shown_failures).mark_bar().encode(x=alt.X('batch_id:N', title='Batch', sort=shown_batches.tolist(), axis=category_axis(True)), y=alt.Y('records:Q', title='Recorded samples'), color='category:N', tooltip=['batch_id:N', 'category:N', 'records:Q']).properties(height=230), width='stretch')
            st.dataframe(failures, hide_index=True, width='stretch')
            st.caption('Complete burnout is explicitly recorded. BOTH_FAIL does not identify burnout. Handling loss, equipment abort, insufficient sample, unresolved outcomes and conflicts remain separate from chemical failure labels.')
            export(failures, 'failure_breakdown', origin)
    if repeats.open:
        with repeats:
            st.write('**Repeatability & Robust Window**')
            repeated = cached_repeatability(frame)
            if not comparable:
                st.caption('Recipes include composition and process identity; cross-cohort recipes are never pooled into one condition.')
            if not repeated.empty:
                repeated = repeated.sort_values(['independent_batches', 'resolved'], ascending=False)
                recurrent = repeated.loc[repeated['independent_batches'] >= 2]
                st.metric('Recipes repeated across independent batches', len(recurrent))
                if not recurrent.empty:
                    chart = alt.Chart(recurrent.head(25)).mark_point(size=85, color='#097f7a').encode(y=alt.Y('example_sample:N', title='Recipe example', axis=category_axis()), x=alt.X('pass_rate:Q', scale=alt.Scale(domain=[0, 1]), title='Observed candidate pass rate'), tooltip=['example_sample:N', 'composition:N', 'resolved:Q', 'independent_batches:Q', 'equal_batch_pass_rate:Q', 'pass_rate:Q'])
                    intervals = alt.Chart(recurrent.head(25)).mark_rule().encode(y=alt.Y('example_sample:N', axis=category_axis()), x='lower_95:Q', x2='upper_95:Q')
                    st.altair_chart((chart + intervals).properties(height=row_height(len(recurrent.head(25)))), width='stretch')
                else:
                    st.info('Repeat the same recipe in new synthesis batches to measure reproducibility. Multiple measurements of one specimen are not independent syntheses.')
                st.dataframe(repeated.drop(columns='recipe'), hide_index=True, width='stretch')
                minimum = st.number_input('Observed stable-recipe filter: minimum independent batches', min_value=2, max_value=100, value=3, key=f'repeat_min_batches_{key}')
                threshold = st.slider('Observed stable-recipe filter: minimum pass rate', 0., 1., .8, .05, key=f'repeat_min_rate_{key}')
                candidates = repeated.loc[(repeated['independent_batches'] >= minimum) & (repeated['equal_batch_pass_rate'] >= threshold)]
                st.write('**Recipes meeting the observed repeatability filters**')
                st.dataframe(candidates.drop(columns='recipe'), hide_index=True, width='stretch')
                st.caption('This filter uses equal-batch observed pass rate and explicit repeat counts. It identifies candidates for confirmation, not a statistically certified continuous success window.')
                export(repeated, 'recipe_repeatability', origin)
    if composition.open:
        with composition:
            st.write('**Element & Composition Coverage**')
            elements, pairs, stages = composition_coverage(samples, methods)
            positions = {symbol: {'period': period, 'group': group} for period, row in enumerate(PERIODS, 1) for group, symbol in row.items() if symbol in ELEMENTS}
            periodic = elements.assign(period=[positions[e]['period'] for e in elements['element']], group=[positions[e]['group'] for e in elements['element']])
            cells = alt.Chart(periodic).mark_rect(stroke='#ffffff').encode(x=alt.X('group:O', title='Periodic-table column', axis=alt.Axis(labelOverlap=False)), y=alt.Y('period:O', title='Period / f-block row', axis=category_axis()), color=alt.Color('resolved:Q', scale=alt.Scale(scheme='tealblues')), tooltip=['element:N', 'records:Q', 'resolved:Q', 'single_element_records:Q', 'mixture_records:Q'])
            text = alt.Chart(periodic).mark_text(fontSize=10, color='#102b40').encode(x='group:O', y='period:O', text='element:N')
            st.altair_chart((cells + text).properties(height=310), width='stretch')
            st.dataframe(stages, hide_index=True, width='stretch')
            st.caption('Coverage counts recorded membership, not element-specific success or verified atomic dispersion. A 50-element specimen contributes co-membership observations for its pairs; it does not isolate pair interactions.')
            options = elements.loc[elements['records'] > 0, 'element'].tolist()
            pair_key = f'pair_elements_{origin}'
            st.session_state.setdefault(pair_key, options[:6])
            controls = st.columns(2)
            if controls[0].button('Select all observed elements', key=f'pair_observed_{origin}'):
                st.session_state[pair_key] = options
            if controls[1].button('Select all 118 elements', key=f'pair_all_{origin}'):
                st.session_state[pair_key] = list(ELEMENTS)
            selected = element_multiselect('Pair coverage elements', ELEMENTS, key=pair_key)
            if selected:
                table = pairs.loc[pairs['element_1'].isin(selected) & pairs['element_2'].isin(selected)]
                mirrored = table.rename(columns={'element_1': 'element_2', 'element_2': 'element_1'})
                table = pd.concat([table, mirrored])
                full = pd.DataFrame([{'element_1': a, 'element_2': b} for a in selected for b in selected]).merge(table, how='left').fillna({'resolved_records': 0})
                full.loc[full.element_1 == full.element_2, 'resolved_records'] = None
                chart = matrix_chart(full[['element_1', 'element_2', 'resolved_records']], 'element_1', 'element_2', 'resolved_records', selected, selected, 'Resolved co-membership records')
                render_matrix(chart, len(selected), f'pair_coverage_{origin}')
                st.caption('All 118 elements are selectable. Unobserved pairs have zero recorded co-memberships; this is not a failed synthesis. The diagonal is intentionally unevaluated.')
            export(elements, 'element_coverage', origin)
            export(pairs, 'pair_coverage', origin)
            export(stages, 'stage_coverage', origin)
            st.write('**Transfer Benefit**')
            cohorts = {}
            for s in samples:
                cohorts.setdefault(cohort_key(s), []).append(s)
            choices = list(cohorts)
            from .ui_calculation_batch import select_targets,run_jobs,detail as batch_detail
            selected=select_targets('Target compositions for transfer validation',list(range(len(choices))),
                format_func=lambda i:f"{composition_name(cohorts[choices[i]][0])} | {len(cohorts[choices[i]])} records | {choices[i][3]}",
                key=f'transfer_analysis_targets_{origin}')
            payload=json.dumps({'samples':samples,'methods':methods,'contexts':[cohorts[choices[i]][0] for i in selected],'origin':origin},sort_keys=True)
            signature = fingerprint(payload)
            if st.button('Evaluate paired transfer benefit',disabled=not selected,key=f'transfer_analysis_run_{origin}'):
                jobs=[(f"{i+1}: {composition_name(cohorts[choices[i]][0])}",json.dumps(
                    {'samples':samples,'methods':methods,'context':cohorts[choices[i]][0],'origin':origin},sort_keys=True)) for i in selected]
                st.session_state[f'transfer_analysis_result_{origin}']=(signature,run_jobs(jobs,cached_transfer))
            saved = st.session_state.get(f'transfer_analysis_result_{origin}')
            if saved and saved[0] == signature:
                result,info=batch_detail(saved[1],f'transfer_batch_{origin}')
                st.caption(str(info))
                if result.empty:
                    st.info('No paired comparison available. Collect independent target batches and both screening outcomes.')
                else:
                    scores = pd.DataFrame([{'model': name, **probability_metrics(result, column)} for name, column in [('Target only', 'p_target_only'), ('Target + sources', 'p_transfer'), ('Lower stages only', 'p_lower_stage_only')]])
                    st.dataframe(scores, hide_index=True, width='stretch')
                    if scores.loc[0, 'brier'] is not None and scores.loc[1, 'brier'] is not None:
                        st.metric('Transfer Brier improvement (positive = better)', f"{scores.loc[0, 'brier'] - scores.loc[1, 'brier']:.4f}")
                    st.dataframe(result, hide_index=True, width='stretch')
                    export(result, 'paired_transfer_validation', origin)
    if validation.open:
        with validation:
            st.write('**Training Data Readiness**')
            from .ui_calculation_batch import select_targets
            readiness_contexts={cohort_key(s):s for s in samples}
            readiness_options=list(readiness_contexts.values())
            readiness_selected=select_targets('Readiness target compositions',list(range(len(readiness_options))),
                format_func=lambda i:f'{composition_name(readiness_options[i])} | {readiness_options[i]["protocol_id"]}',
                key=f'readiness_targets_{origin}')
            mode = st.radio('Readiness model pool', ['Exact composition', 'Composition transfer'], horizontal=True, key=f'readiness_mode_{key}')
            readiness_rows=[];gates=[]
            for index in readiness_selected:
                detail,gate=gate_summary(samples,methods,readiness_options[index],origin,mode=='Composition transfer')
                name=f'{index+1}: {composition_name(readiness_options[index])}'
                readiness_rows.append(detail.assign(calculation=name));gates.append({'calculation':name,**gate})
            if gates:
                st.dataframe(pd.DataFrame(gates),hide_index=True,width='stretch')
                readiness_table=pd.concat(readiness_rows,ignore_index=True)
                st.dataframe(readiness_table,hide_index=True,width='stretch')
                export(readiness_table,'training_readiness',origin)
            st.caption('Existing engineering gates are evaluated separately for every selected target. Passing a gate does not prove reliability. Each target uses its compatible source pool from the shared experiment selection.')
            predictions = prediction_records(samples, methods, recommendations)
            probability_panel(predictions, origin)
            render_research_validation(samples, methods, origin)
            st.write('**Learning Progress & Efficiency**')
            progress = learning_progress(samples, methods)
            if progress.empty:
                st.info('Chronological progress needs recorded result-reporting timestamps and actual synthesis batches. Sample IDs are not treated as dates. New linked result feedback captures these timestamps automatically.')
            else:
                chart = alt.Chart(progress).mark_line(point=True, color='#097f7a').encode(x=alt.X('cumulative_resolved_experiments:Q', title='Cumulative resolved syntheses'), y=alt.Y('successful_compositions:Q', title='Compositions with a candidate PASS'), tooltip=['iteration:Q', 'batch_id:N', 'cumulative_resolved_experiments:Q', 'cumulative_passes:Q', 'cumulative_burnouts:Q', 'successful_compositions:Q'])
                st.altair_chart(chart.properties(height=240), width='stretch')
                st.dataframe(progress, hide_index=True, width='stretch')
                export(progress, 'learning_progress', origin)
            valid = predictions.loc[predictions['eligible']]
            if not valid.empty:
                policies = valid.groupby('policy').agg(n=('label', 'size'), passed=('label', 'sum'), independent_batches=('batch_id', 'nunique')).reset_index()
                policies['observed_pass_rate'] = policies['passed'] / policies['n']
                st.dataframe(policies, hide_index=True, width='stretch')
                export(policies, 'policy_outcomes', origin)
            with st.expander('Compare experiment strategies within the selected cohort'):
                if comparable:
                    batch_options = sorted(frame.loc[frame['label'].notna() & ~frame['batch_id'].isin(['UNKNOWN', 'UNASSIGNED']), 'batch_id'].unique())
                    guided = st.multiselect('Model-guided synthesis batches', batch_options, key=f'strategy_guided_{key}')
                    baseline_name = st.selectbox('Comparator strategy', ['Random design', 'Space-filling design', 'Manual design'], key=f'strategy_kind_{key}')
                    baseline = st.multiselect('Comparator synthesis batches', [b for b in batch_options if b not in guided], key=f'strategy_baseline_{key}')
                    if guided and baseline:
                        comparison = strategy_comparison(frame, guided, baseline, baseline_name)
                        st.dataframe(comparison, hide_index=True, width='stretch')
                        chart = alt.Chart(comparison.dropna(subset=['pass_rate'])).mark_point(size=110, color='#097f7a').encode(y=alt.Y('strategy:N', axis=category_axis()), x=alt.X('pass_rate:Q', scale=alt.Scale(domain=[0, 1]), title='Observed candidate pass rate'), tooltip=['strategy:N', 'resolved:Q', 'synthesis_batches:Q', 'passed:Q', 'burned_out:Q', 'pass_rate:Q'])
                        interval = alt.Chart(comparison).mark_rule().encode(y=alt.Y('strategy:N', axis=category_axis()), x='lower_95:Q', x2='upper_95:Q')
                        st.altair_chart((chart + interval).properties(height=160), width='stretch')
                        export(comparison, 'strategy_comparison', origin)
                else:
                    st.info('Select One comparable cohort above for a strategy comparison.')
