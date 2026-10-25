"""Inline periodic tables with immediate, isolated field updates."""
import streamlit as st

from .batching import PERIODS
from .domain import ELEMENTS, ELEMENT_ORDER, TARGETS, parse_elements


def ordered(values, allowed=ELEMENTS):
    members = set(values) & set(allowed)
    return [element for element in ELEMENT_ORDER if element in members]


def _values(key, kind):
    value = st.session_state.get(key)
    if kind == 'single':
        return [value] if value else []
    if kind == 'text':
        try:
            return parse_elements(value or '')
        except ValueError:
            return []
    return list(value or [])


def _replace(key, values, kind='multiple'):
    values = list(values)
    if kind == 'text':
        st.session_state[key] = ';'.join(values)
    elif kind == 'single':
        st.session_state[key] = values[0] if values else None
    else:
        st.session_state[key] = values


def _toggle(key, symbol, kind):
    selected = _values(key, kind)
    _replace(key, [symbol] if kind == 'single' else
        ordered(set(selected) ^ {symbol}), kind)


def _table(key, options, kind='multiple', disabled=False):
    selected = _values(key, kind)
    allowed = set(options)
    with st.container(key='periodic_grid_' + key):
        for period, positions in enumerate(PERIODS):
            columns = st.columns(18, gap='small')
            for group, symbol in positions.items():
                enabled = symbol in allowed
                columns[group - 1].button(symbol, disabled=disabled or not enabled,
                    type='primary' if symbol in selected else 'secondary',
                    key=f'{key}_{period}_{symbol}', width='stretch',
                    on_click=_toggle, args=(key, symbol, kind))
    st.caption('Highlighted elements are selected. Ln and An are detached-series labels. '
        + ('All 118 elements are available.' if len(allowed) == 118 else
           'Elements outside this field\'s valid data scope are disabled.'))
    st.write('Selected: ' + (', '.join(selected) if selected else 'None'))


def _controls(key, options, kind='multiple', disabled=False):
    with st.container(key='periodic_controls_' + key):
        all_elements, campaign, clear = st.columns(3)
        all_elements.button('Select all 118' if len(options) == 118 else 'Select all available',
            key=key + '_all', disabled=disabled, on_click=_replace,
            args=(key, ordered(options), kind))
        campaign.button('Use original 50', key=key + '_baseline',
            disabled=disabled, on_click=_replace, args=(key, ordered(TARGETS, options), kind))
        clear.button('Clear selection', key=key + '_clear',
            disabled=disabled, on_click=_replace, args=(key, [], kind))


def periodic_selector(key, initial):
    """Inline selector retained for primary batch and planning compositions."""
    st.session_state.setdefault(key, ordered(initial))
    st.write('**Select elements from the periodic table**')
    with st.container(key=key + '_controls'):
        _controls(key, ELEMENTS)
    _table(key, ELEMENTS)
    return list(st.session_state[key])


def _inline(key, options, kind, disabled=False):
    if kind != 'single':
        _controls(key, options, kind, disabled)
    _table(key, options, kind, disabled)


def element_multiselect(label, options, *, key, **kwargs):
    if key in st.session_state:
        kwargs.pop('default', None)
        retained=[v for v in st.session_state[key] if v in options]
        if retained!=st.session_state[key]:
            st.session_state[key]=retained
    selected = st.multiselect(label, options, key=key, **kwargs)
    _inline(key, options, 'multiple', kwargs.get('disabled', False))
    return selected


def element_selectbox(label, options, *, key, **kwargs):
    if key in st.session_state:
        kwargs['index'] = 0
        if st.session_state[key] not in options:
            st.session_state[key]=options[0] if options else None
    selected = st.selectbox(label, options, key=key, **kwargs)
    _inline(key, options, 'single', kwargs.get('disabled', False))
    return selected


def element_text_area(label, value, *, key):
    text = st.text_area(label, value=None if key in st.session_state else value, key=key)
    _inline(key, ELEMENTS, 'text')
    return text
