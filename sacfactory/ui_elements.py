"""Shared periodic-table controls for element-valued UI fields.

Only the active picker renders a table. Draft changes are isolated until Apply;
callbacks commit widget values before the containing page reruns.
"""
import streamlit as st

from .batching import PERIODS
from .domain import ELEMENTS, ELEMENT_ORDER, TARGETS, parse_elements


def ordered(values, allowed=ELEMENTS):
    members = set(values) & set(allowed)
    return [element for element in ELEMENT_ORDER if element in members]


def _replace(key, values):
    st.session_state[key] = list(values)


def _toggle(key, symbol, single):
    selected = st.session_state.get(key, [])
    st.session_state[key] = ([symbol] if single else
        ordered(set(selected) ^ {symbol}))


def _table(key, options, single=False, summary_limit=None):
    selected = st.session_state[key]
    allowed = set(options)
    with st.container(key='periodic_grid_' + key):
        for period, positions in enumerate(PERIODS):
            columns = st.columns(18, gap='small')
            for group, symbol in positions.items():
                enabled = symbol in allowed
                columns[group - 1].button(symbol, disabled=not enabled,
                    type='primary' if symbol in selected else 'secondary',
                    key=f'{key}_{period}_{symbol}', width='stretch',
                    help=('Selected element' if symbol in selected else
                          'Select element' if enabled else 'Unavailable for this field'),
                    on_click=_toggle, args=(key, symbol, single))
    st.caption('Highlighted elements are selected. Ln and An are detached-series labels. '
        + ('All 118 elements are available.' if len(allowed) == 118 else
           'Elements outside this field\'s valid data scope are disabled.'))
    shown = selected if summary_limit is None else selected[:summary_limit]
    remainder = len(selected) - len(shown)
    st.write('Selected: ' + (', '.join(shown) if shown else 'None') +
        (f' (+{remainder} more; {len(selected)} total)' if remainder else ''))


def _controls(key, options):
    with st.container(key='periodic_controls_' + key):
        all_elements, campaign, clear = st.columns(3)
        all_elements.button('Select all 118' if len(options) == 118 else 'Select all available',
            key=key + '_all', on_click=_replace, args=(key, ordered(options)))
        campaign.button('Use original 50', key=key + '_baseline',
            on_click=_replace, args=(key, ordered(TARGETS, options)))
        clear.button('Clear selection', key=key + '_clear',
            on_click=_replace, args=(key, []))


def periodic_selector(key, initial):
    """Inline selector retained for primary batch and planning compositions."""
    st.session_state.setdefault(key, ordered(initial))
    st.write('**Select elements from the periodic table**')
    with st.container(key=key + '_controls'):
        _controls(key, ELEMENTS)
    _table(key, ELEMENTS)
    return list(st.session_state[key])


def _dismiss():
    st.session_state.pop('_active_element_picker', None)


def _commit(key, draft_key, kind, options):
    selected = ordered(st.session_state.get(draft_key, []), options)
    if kind == 'single':
        if len(selected) != 1:
            return
        value = selected[0]
    elif kind == 'text':
        value = ';'.join(selected)
    else:
        value = selected
    st.session_state[key] = value
    _dismiss()


@st.dialog('Periodic table', width='large', on_dismiss=_dismiss)
def _picker(key, label, options, kind):
    draft_key = 'periodic_draft_' + key
    st.write('**' + label + '**')
    st.caption('Choose one element.' if kind == 'single' else
               'Choose any number of available elements. Apply to update this field.')
    if kind != 'single':
        _controls(draft_key, options)
    _table(draft_key, options, single=kind == 'single', summary_limit=12)
    with st.container(key='periodic_actions_' + key):
        apply, cancel = st.columns(2)
        if apply.button('Apply element selection', key=draft_key + '_apply', type='primary',
                disabled=kind == 'single' and len(st.session_state[draft_key]) != 1,
                on_click=_commit, args=(key, draft_key, kind, options)):
            st.rerun()
        if cancel.button('Cancel selection', key=draft_key + '_cancel', on_click=_dismiss):
            st.rerun()


def _open_picker(key, label, options, selected, kind):
    if st.button('Periodic table: ' + label, key=key + '_periodic_open',
                 disabled=not options, help='Select elements visually from the periodic table.'):
        st.session_state['periodic_draft_' + key] = ordered(selected, options)
        st.session_state['_active_element_picker'] = key
    if st.session_state.get('_active_element_picker') == key:
        _picker(key, label, list(options), kind)


def element_multiselect(label, options, *, key, **kwargs):
    if key in st.session_state:
        kwargs.pop('default', None)
        retained=[v for v in st.session_state[key] if v in options]
        if retained!=st.session_state[key]:
            st.session_state[key]=retained
    selected = st.multiselect(label, options, key=key, **kwargs)
    _open_picker(key, label, options, selected, 'multiple')
    return selected


def element_selectbox(label, options, *, key, **kwargs):
    if key in st.session_state:
        kwargs['index'] = 0
        if st.session_state[key] not in options:
            st.session_state[key]=options[0] if options else None
    selected = st.selectbox(label, options, key=key, **kwargs)
    _open_picker(key, label, options, [selected] if selected else [], 'single')
    return selected


def element_text_area(label, value, *, key):
    text = st.text_area(label, value=None if key in st.session_state else value, key=key)
    try:
        selected = parse_elements(text)
    except ValueError:
        selected = []
    _open_picker(key, label, ELEMENTS, selected, 'text')
    return text
