"""Shared multi-select experiment scope without rewriting sample compositions."""
import streamlit as st

from .domain import ELEMENTS
from .ui_elements import element_multiselect


def select_experiments(samples,origin):
    with st.expander('Shared experiment selection for calculations',expanded=False):
        st.caption('Empty selections include all records. Selecting an element retains complete mixed-element recipes.')
        if st.button('Include all experiments and elements',key=f'calculation_all_{origin}'):
            for suffix in ('elements','batches','samples'):
                st.session_state[f'calculation_{suffix}_{origin}']=[]
        elements=element_multiselect('Calculation elements (empty = all)',ELEMENTS,key=f'calculation_elements_{origin}')
        selected=[s for s in samples if not elements or set(s['elements'])&set(elements)]
        batches=sorted({s['batch_id'] for s in selected})
        batch_key=f'calculation_batches_{origin}'
        if batch_key in st.session_state:
            st.session_state[batch_key]=[b for b in st.session_state[batch_key] if b in batches]
        chosen=st.multiselect('Calculation batches (empty = all)',batches,key=batch_key)
        selected=[s for s in selected if not chosen or s['batch_id'] in chosen]
        ids=[s['sample_id'] for s in selected]
        sample_key=f'calculation_samples_{origin}'
        if sample_key in st.session_state:
            st.session_state[sample_key]=[i for i in st.session_state[sample_key] if i in ids]
        chosen_ids=st.multiselect('Calculation samples (empty = all)',ids,key=sample_key)
        selected=[s for s in selected if not chosen_ids or s['sample_id'] in chosen_ids]
        st.caption(f'{len(selected)} / {len(samples)} experiments selected; {len({e for s in selected for e in s["elements"]})} elements. Single-record editors and plot-axis choices remain individual views.')
    return selected
