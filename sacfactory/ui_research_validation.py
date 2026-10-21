"""On-demand research validation and Python probability-agreement exports."""
from io import BytesIO
from zipfile import ZipFile
import json

import pandas as pd
import streamlit as st

from .composition import composition_name
from .domain import cohort_key
from .planning import fingerprint
from .prediction_plot import agreement_table, descriptive_line, figure_bundle
from .validation_research import forward_validation, unseen_element_validation, negative_transfer_validation
from .extra_models import reliability_benchmark


@st.cache_data(show_spinner=False, max_entries=8)
def cached_validation(payload, protocol, element, iterations, folds, novel_recipes, batch_effect, benchmark_protocol='target_batch'):
    arguments = json.loads(payload)
    if protocol == 'Reliability model benchmark':
        return reliability_benchmark(**arguments, iterations=iterations, max_folds=folds,
            protocol=benchmark_protocol, element=element, batch_effect=batch_effect)
    if protocol == 'Chronological replay':
        return forward_validation(**arguments, iterations=iterations, max_folds=folds,
                                  novel_recipes=novel_recipes, batch_effect=batch_effect)
    if protocol == 'Strict unseen element':
        return unseen_element_validation(**arguments, element=element, iterations=iterations, batch_effect=batch_effect)
    return negative_transfer_validation(**arguments, iterations=iterations, max_folds=folds, batch_effect=batch_effect)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_bundle(table_json, records_json, origin, fit):
    return figure_bundle(pd.read_json(BytesIO(table_json.encode()), orient='table'),
                         pd.read_json(BytesIO(records_json.encode()), orient='table'), origin, fit)


def render_agreement(records, origin):
    with st.expander('Predicted probability vs observed success rate', expanded=False):
        st.caption('Only eligible saved pre-result forecasts enter this figure. Single outcomes are 0/1; observed probabilities are frequencies over groups or repeated recipes. Loading is not the target.')
        valid = records.copy()
        selected = st.multiselect('Agreement compositions (empty = all)', sorted(valid.loc[valid['eligible'], 'composition'].unique()),
                                  key=f'agreement_compositions_{origin}')
        if selected:
            valid = valid.loc[valid['composition'].isin(selected)]
        processes = sorted(valid['process_identity'].unique())
        process_key=f'agreement_processes_{origin}'
        if process_key in st.session_state:
            st.session_state[process_key]=[v for v in st.session_state[process_key] if v in processes]
        chosen_processes = st.multiselect('Agreement process contexts (empty = all)', processes,key=process_key)
        if chosen_processes:
            valid = valid.loc[valid['process_identity'].isin(chosen_processes)]
        elif len(processes) > 1:
            st.caption('This overview pools process contexts. Select one context for a process-specific assessment.')
        mode = st.radio('Agreement point definition', ['Probability bins', 'Repeated recipes'], horizontal=True,
                        key=f'agreement_mode_{origin}')
        bins = st.slider('Number of probability bins', min_value=3, max_value=10, value=5, key=f'agreement_bins_{origin}')
        stage = st.checkbox('Separate element-count stages', value=True, key=f'agreement_stage_{origin}')
        fit = st.checkbox('Show descriptive linear fit', value=False, key=f'agreement_fit_{origin}')
        table = agreement_table(valid, bins, stage, mode == 'Repeated recipes')
        if table.empty:
            st.info('No eligible points in this selection. Recipe-level points require repeats in at least two actual batches.')
            return
        bundle = cached_bundle(table.to_json(orient='table'), valid.to_json(orient='table'), origin, fit)
        with ZipFile(BytesIO(bundle)) as archive:
            st.image(archive.read('prediction_agreement.png'), width='stretch')
        line = descriptive_line(table) if fit else None
        if line is not None:
            st.write(f"Descriptive fit: y = {line['slope']:.3f} x + {line['intercept']:.3f}")
        elif fit:
            st.info('A descriptive line needs at least three points with distinct mean predictions.')
        st.caption('Evaluate paired predictions on held-out batches; intervals describe the specified resampling method.')
        st.dataframe(table, hide_index=True, width='stretch')
        st.download_button('Download agreement figure and source data', bundle, f'{origin.lower()}_prediction_agreement.zip',
                           'application/zip', key=f'agreement_download_{origin}')


def render_research_validation(samples, methods, origin):
    with st.expander('Advanced prediction validation', expanded=False):
        contexts = {}
        for sample in samples:
            contexts.setdefault(cohort_key(sample), sample)
        if not contexts:
            st.info('Enter experimental recipes and returned outcomes before validation.')
            return
        choices = list(contexts.values())
        from .ui_calculation_batch import select_targets,run_jobs,detail
        indices=select_targets('Advanced validation targets',list(range(len(choices))),
            format_func=lambda i: f"{composition_name(choices[i])} | {choices[i]['protocol_id']} | {choices[i]['sample_id']}",
            key=f'advanced_targets_{origin}')
        if not indices:
            st.info('Select one or more validation targets.')
            return
        context = choices[indices[0]]
        protocol = st.selectbox('Validation protocol', ['Chronological replay', 'Strict unseen element', 'Paired adaptive transfer', 'Reliability model benchmark'],
                                key=f'advanced_protocol_{origin}')
        benchmark_protocol = 'target_batch'
        if protocol == 'Reliability model benchmark':
            benchmark_protocol = st.selectbox('Reliability benchmark split', ['target_batch', 'forward', 'unseen_element'],
                key=f'benchmark_split_{origin}', format_func={'target_batch': 'Outer target-batch holdout', 'forward': 'Chronological corrected-data replay', 'unseen_element': 'Strict unseen element'}.get)
        from .ui_elements import element_multiselect
        elements=element_multiselect('Held-out elements',sorted({e for sample in samples for e in sample['elements']}),
            default=[context['elements'][0]],key=f'advanced_elements_{origin}')
        iterations = st.number_input('Validation learning iterations', min_value=2, max_value=60, value=8, key=f'advanced_iterations_{origin}')
        folds = st.number_input('Validation fold budget', min_value=2, max_value=12, value=3, key=f'advanced_folds_{origin}')
        novel = st.checkbox('Purge exact test recipes in chronological replay', value=False, key=f'advanced_novel_{origin}')
        batch = st.checkbox('Include supported batch effects in validation', value=True, key=f'advanced_batch_{origin}')
        payload = json.dumps({'samples': samples, 'methods': methods, 'context': context, 'origin': origin}, sort_keys=True)
        signature = fingerprint([payload,indices,protocol,elements,iterations,folds,novel,batch,benchmark_protocol])
        if st.button('Run advanced validation', key=f'advanced_run_{origin}'):
            jobs=[]
            unseen=protocol=='Strict unseen element' or protocol=='Reliability model benchmark' and benchmark_protocol=='unseen_element'
            for index in indices:
                ctx=choices[index]
                for element in elements if unseen else [ctx['elements'][0]]:
                    encoded=json.dumps({'samples':samples,'methods':methods,'context':ctx,'origin':origin},sort_keys=True)
                    jobs.append((f"{index+1}: {composition_name(ctx)} | held out {element}",(encoded,element)))
            result=run_jobs(jobs,lambda args:cached_validation(args[0],protocol,args[1],iterations,folds,novel,batch,benchmark_protocol))
            st.session_state[f'advanced_result_{origin}']=(signature,result)
        saved = st.session_state.get(f'advanced_result_{origin}')
        if saved and saved[0] == signature:
            if not saved[1]:
                st.info('Select at least one held-out element for this protocol.')
                return
            frame,info=detail(saved[1],f'advanced_batch_results_{origin}')
            if info.get('comparison'):
                st.dataframe(pd.DataFrame(info['comparison']), hide_index=True, width='stretch')
            if frame.empty:
                st.info('No evaluable test rows under this protocol. Inspect exclusions, timestamps and fitting diversity.')
            else:
                st.dataframe(frame, hide_index=True, width='stretch')
                st.download_button('Download advanced validation predictions', frame.to_csv(index=False).encode(),
                    f'{origin.lower()}_advanced_validation.csv', 'text/csv', key=f'advanced_download_{origin}')
                st.download_button('Download validation provenance', json.dumps(info, indent=2, allow_nan=False).encode(),
                    f'{origin.lower()}_advanced_validation.json', 'application/json', key=f'advanced_provenance_{origin}')
