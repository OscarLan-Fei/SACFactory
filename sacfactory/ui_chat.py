"""Shared data selection controls used identically across conversation/settings."""
import streamlit as st


def render_data_selection(samples, origin, context):
    scoped = [s for s in samples if s.get('origin') == origin]
    valid_ids = [s['sample_id'] for s in scoped]
    key = f'research_samples_{origin}'
    st.session_state[key] = [i for i in st.session_state.get(key, []) if i in valid_ids]
    all_col, composition_col, clear_col = st.columns(3)
    if all_col.button('Select all records', key=f'share_all_{origin}', width='stretch'):
        st.session_state[key] = valid_ids
    if composition_col.button('Current composition', key=f'share_composition_{origin}', width='stretch'):
        st.session_state[key] = [s['sample_id'] for s in scoped if set(s['elements']) == set(context['elements'])]
    if clear_col.button('Clear selection', key=f'share_clear_{origin}', width='stretch'):
        st.session_state[key] = []
    with st.expander('Select records by batch'):
        batches = sorted({s['batch_id'] for s in scoped})
        batch_key = f'share_batches_{origin}'
        st.session_state[batch_key] = [b for b in st.session_state.get(batch_key, []) if b in batches]
        chosen = st.multiselect('Batches to add', batches, key=batch_key)
        if st.button('Add batches to selection', key=f'share_add_batches_{origin}', disabled=not chosen):
            selected = set(st.session_state[key]) | {s['sample_id'] for s in scoped if s['batch_id'] in chosen}
            st.session_state[key] = [i for i in valid_ids if i in selected]
    with st.expander('Review individual sample selection', expanded=False):
        selected = st.multiselect('Shared experiment records', valid_ids, key=key,
            help='Choose any number of records. Selected conditions and ICP W can be shared; large selections use complete local statistics and paged details. Notes and raw files stay excluded.')
    batch_count = len({s['batch_id'] for s in scoped if s['sample_id'] in selected})
    st.caption(f'{len(selected)} of {len(valid_ids)} records selected · {batch_count} batches · No sample-count limit. Selecting records does not enable API requests.')
    return selected
