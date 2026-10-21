"""Shared REAL/DEMO chemistry inputs, source inspection and protected ablations."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from .precursors import TYPES, SPINS, COORDINATIONS, FIELDS, PROFILES, normalize_precursors, precursor_coverage, state_reference
from .chemical_descriptors import coverage, reference
from .ui_charts import category_axis, row_height


def chemistry_columns():
    return {
        'precursor_compound': st.column_config.TextColumn('Precursor compound / hydrate'),
        'precursor_type': st.column_config.SelectboxColumn('Precursor type', options=TYPES),
        'precursor_oxidation_state': st.column_config.NumberColumn('Precursor oxidation state', min_value=-8, max_value=8, step=1),
        'precursor_coordination': st.column_config.SelectboxColumn('Reference coordination', options=COORDINATIONS),
        'precursor_spin': st.column_config.SelectboxColumn('Reference spin', options=SPINS),
        'precursor_source': st.column_config.TextColumn('Precursor source / record'),
        'precursor_state_source': st.column_config.TextColumn('State evidence / reference'),
    }


def render_precursor_editor(elements, current, key):
    st.write('**Declared precursors per element**')
    st.caption('Record actual precursors and chemical states. Unknown properties remain missing; reference values include source information.')
    if not elements:
        st.info('Select target elements to enter their actual precursors.')
        return {}
    rows = []
    for e in elements:
        entry = current.get(e, {})
        rows.append({'element': e, **{'precursor_' + k: entry.get(k, None if k == 'oxidation_state' else 'UNKNOWN' if k in {'type', 'coordination', 'spin'} else '') for k in FIELDS}})
    frame = st.data_editor(pd.DataFrame(rows), hide_index=True, disabled=['element'],
        column_config=chemistry_columns(), width='stretch', key=key)
    return {r['element']: {k: r.get('precursor_' + k) for k in FIELDS} for r in frame.to_dict('records')}


def render_chemical_reference(context, key):
    st.write('**Elemental reference properties**')
    st.dataframe(pd.DataFrame(coverage(context['elements'])), hide_index=True, width='stretch')
    st.dataframe(pd.DataFrame(precursor_coverage(context)), hide_index=True, width='stretch')
    st.caption('An exact ionic lookup is a state-specific reference value, not a measured solution structure. Origin/estimated flags remain visible. UNKNOWN spin is never replaced by HS, LS or an unspecified reference entry.')
    provenance = {'elemental': {k: v for k, v in reference().items() if k != 'elements'},
                  'state_radii': {k: v for k, v in state_reference().items() if k not in {'elements', 'ionic_radii'}}}
    st.download_button('Download chemistry reference provenance', json.dumps(provenance, indent=2).encode(),
                       'chemistry_reference_provenance.json', 'application/json', key=key + '_provenance')
    render_sigma_catalog(context['elements'], key + '_sigma')


def render_sigma_catalog(elements, key):
    from .precursor_catalog import catalog, catalog_rows
    from .domain import TARGETS
    data = catalog()
    scope = st.radio('Sigma catalogue view', ['Selected target elements', 'Original 50 target elements'], horizontal=True, key=key + '_scope')
    rows = catalog_rows(elements if scope == 'Selected target elements' else TARGETS)
    st.write('**Sigma precursor reference candidates**')
    st.caption('Original 50 targets were reviewed on ' + data['checked_date'] + '. Chlorides are registered as the per-element default; there is no automatic nitrate fallback. FeCl2, CuCl2, InCl3 and AuCl3 follow your declared formulas. Other entries prefer lower verified formal oxidation-state chlorides. Defaults can be edited. Catalogue products do not confirm purchased grades, lots, hydrates or solution species. Unknown coordination and spin remain unknown. Catalogue/partner identity is retained; stock and globally lowest-state availability are not certified.')
    columns = ['element', 'compound', 'formal_oxidation_state', 'catalog_number', 'cas', 'seller', 'source_url', 'notes', 'status']
    frame = pd.DataFrame(rows).reindex(columns=columns)
    st.dataframe(frame, hide_index=True, width='stretch',
                 column_config={'source_url': st.column_config.LinkColumn('Sigma source', display_text='Product / listing')})
    st.download_button('Download Sigma precursor comparison', frame.to_csv(index=False).encode('utf-8-sig'),
                       'sigma_precursor_candidates.csv', 'text/csv', key=key + '_csv')
    st.caption('Catalogue snapshot SHA-256: ' + data['snapshot_sha256'])


@st.cache_data(show_spinner=False, max_entries=8)
def cached_ablation(payload, protocol, element, iterations, folds, batch_effect, source_identity=None):
    from .chemistry_validation import chemistry_ablation
    return chemistry_ablation(**json.loads(payload), protocol=protocol, element=element,
                             iterations=iterations, max_folds=folds, batch_effect=batch_effect)


def render_chemistry_analysis(samples, selected_samples, methods, origin):
    from .domain import cohort_key, ELEMENTS
    from .composition import composition_name
    from .planning import fingerprint
    st.write('**Chemical information and transfer ablation**')
    st.caption('The same module is available in REAL and DEMO. Sources come from the selected namespace, with fixed support/protocol/control basis. DEMO comparisons describe simulated outcomes only.')
    contexts = {}
    for sample in selected_samples + samples:
        contexts.setdefault(cohort_key(sample), sample)
    options = list(contexts.values())
    if not options:
        st.info('Save at least one recipe to inspect chemical coverage.')
        return
    from .ui_calculation_batch import select_targets,run_jobs,detail
    selected = select_targets('Chemistry target compositions',list(range(len(options))),
        format_func=lambda i: f"{composition_name(options[i])} | {options[i]['protocol_id']} | {options[i]['sample_id']}",
        key=f'chemistry_targets_{origin}')
    if not selected:
        st.info('Select one or more target compositions.')
        return
    context = options[selected[0]]
    with st.expander('Elemental and state-specific reference coverage'):
        render_chemical_reference(context, f'analysis_reference_{origin}')
    protocols = {'single_to_mixture': 'Single-element sources to mixtures', 'lower_to_higher': 'Lower element counts to higher counts',
                 'unseen_element': 'Strict unseen-element transfer', 'target_batch': 'Outer target-batch holdout'}
    protocol = st.selectbox('Chemistry ablation protocol', list(protocols), format_func=protocols.get, key=f'chemistry_protocol_{origin}')
    from .ui_elements import element_multiselect
    elements=element_multiselect('Chemistry held-out elements',ELEMENTS,default=[context['elements'][0]],
        key=f'chemistry_elements_{origin}')
    iterations = st.number_input('Chemistry learning iterations', min_value=2, max_value=60, value=4, key=f'chemistry_iterations_{origin}')
    folds = st.number_input('Chemistry fold / stage budget', min_value=2, max_value=6, value=3, key=f'chemistry_folds_{origin}')
    batch_effect = st.checkbox('Supported batch effects in chemistry ablation', value=False, key=f'chemistry_batch_{origin}')
    payload = json.dumps({'samples': samples, 'methods': methods, 'context': context, 'origin': origin}, sort_keys=True)
    from .mixing import reference_identity
    source_identity = json.dumps(reference_identity(), sort_keys=True)
    signature = fingerprint([payload,selected,protocol,elements,iterations,folds,batch_effect,source_identity,
                             reference()['snapshot_sha256'], state_reference()['snapshot_sha256']])
    st.caption(f'{len(PROFILES)} paired variants: periodic baseline, elemental properties, declared precursor charges, salt families, matched ionic states, sourced thermal properties, unfitted loss-prior features, mixing thermodynamics and the SAC-Gen physical profile. All use identical outer rows and GP family. No model is promoted automatically.')
    if st.button('Run chemical-transfer ablation', key=f'chemistry_run_{origin}'):
        jobs=[]
        for index in selected:
            ctx=options[index]
            held=elements if protocol=='unseen_element' else [ctx['elements'][0]]
            for element in held:
                encoded=json.dumps({'samples':samples,'methods':methods,'context':ctx,'origin':origin},sort_keys=True)
                jobs.append((f"{index+1}: {composition_name(ctx)} | held out {element}",(encoded,element)))
        result=run_jobs(jobs,lambda args:cached_ablation(args[0],protocol,args[1],iterations,folds,batch_effect,source_identity))
        st.session_state[f'chemistry_result_{origin}']=(signature,result)
    saved = st.session_state.get(f'chemistry_result_{origin}')
    if saved and saved[0] == signature:
        if not saved[1]:
            st.info('Select at least one held-out element for unseen-element evaluation.')
            return
        frame, info = detail(saved[1],f'chemistry_batch_{origin}')
        if frame.empty:
            st.info('No evaluable source/target split yet. Enter resolved outcomes, controlled settings and independent batch metadata; mixtures are needed for expansion evaluation.')
        else:
            comparison = pd.DataFrame(info['comparison'])
            chart = alt.Chart(comparison).mark_bar(color='#097f7a').encode(
                y=alt.Y('profile:N', title=None, sort=list(PROFILES), axis=category_axis()),
                x=alt.X('equal_batch_brier:Q', title='Held-out Brier score, equal batch weight (lower is better)'),
                tooltip=['profile:N', 'equal_batch_brier:Q', 'test_batches:Q'])
            st.altair_chart(chart.properties(height=row_height(len(comparison))), width='stretch')
            st.dataframe(comparison, hide_index=True, width='stretch')
            st.dataframe(pd.DataFrame(info['paired_effects']), hide_index=True, width='stretch')
            with st.expander('Inspect held-out chemical-transfer predictions'):
                st.dataframe(frame, hide_index=True, width='stretch')
            st.download_button('Download chemical-transfer predictions', frame.to_csv(index=False).encode(),
                               f'{origin.lower()}_chemistry_ablation.csv', 'text/csv', key=f'chemistry_export_{origin}')
        st.download_button('Download chemical-transfer provenance', json.dumps(info, indent=2, allow_nan=False).encode(),
                           f'{origin.lower()}_chemistry_ablation.json', 'application/json', key=f'chemistry_provenance_{origin}')
