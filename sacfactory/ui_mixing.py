"""Recipe-level mixing descriptors with full periodic selection and source audit."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from .domain import ELEMENTS
from .mixing import LABELS, pair_coverage, reference, reference_identity
from .ui_elements import element_multiselect


def render_mixing(samples, recipes, origin, profile):
    st.write('**Mixing thermodynamics / nominal metal feed**')
    st.caption('Mixing descriptors use precursor molar fractions. Missing binary data remain missing; these proxies do not establish atomic dispersion.')
    if profile in {'chemistry_mixing', 'chemistry_pulseicp'}:
        st.success('Mixing thermodynamics is enabled in the selected prediction profile.')
    else:
        st.info('These descriptors are calculated for inspection. Select Thermal properties + mixing H, S and G proxies under Next batch > Prediction descriptor profile to use them in model fitting.')
    observed = list(dict.fromkeys(e for s in samples for e in s['elements']))
    elements = element_multiselect('Mixing descriptor elements', ELEMENTS, default=observed, key=f'mixing_elements_{origin}')
    ids = {s['sample_id'] for s in samples if not elements or set(elements) & set(s['elements'])}
    frame = recipes.loc[recipes.sample_id.isin(ids)].copy()
    columns = ['sample_id', 'batch_id', 'composition', 'status', *LABELS,
               'mix_complete', 'mix_pair_count', 'mix_available_pair_count', 'mix_missing_pairs', 'mix_input_status']
    st.dataframe(frame.reindex(columns=columns), hide_index=True, width='stretch')
    st.download_button('Download mixing descriptor recipes CSV', frame.reindex(columns=columns).to_csv(index=False).encode(),
        f'{origin.lower()}_mixing_descriptors.csv', 'text/csv', key=f'mixing_csv_{origin}')
    if not frame.empty:
        metric = st.selectbox('Mixing descriptor to plot', list(LABELS), format_func=LABELS.get, key=f'mixing_metric_{origin}')
        points = frame.dropna(subset=[metric])
        if points.empty or points[metric].nunique() < 2:
            st.info('This descriptor is unavailable or constant in the current scope. Pure elements have zero mixing H and S by definition; this does not mean synthesis succeeded.')
        if not points.empty:
            dots = alt.Chart(points).mark_circle(size=65, opacity=.7).encode(
                x=alt.X('element_count:Q', title='Elements per full recipe', scale=alt.Scale(zero=False)),
                y=alt.Y(metric + ':Q', title=LABELS[metric], scale=alt.Scale(zero=False)), color='status:N',
                tooltip=['sample_id:N', 'composition:N', 'status:N', metric + ':Q', 'mix_input_status:N'])
            st.altair_chart(dots.properties(height=300), width='stretch')
    with st.expander('Binary mixing parameter coverage and source'):
        pairs = pair_coverage(elements)
        st.caption(f'{int(pairs.available.sum())} / {len(pairs)} selected unlike pairs have documented values. The original 50-metal scope has 1,225 pairs; the full pinned source contains 2,628 records across 75 symbols, not a complete 75-element matrix. Missing pairs remain unknown; an explicit documented zero remains zero.')
        st.dataframe(pairs, hide_index=True, width='stretch')
        st.download_button('Download selected binary mixing pairs', pairs.to_csv(index=False).encode(),
            'binary_mixing_pairs.csv', 'text/csv', key=f'mixing_pairs_csv_{origin}')
        data = reference()
        st.download_button('Download mixing reference provenance', json.dumps(reference_identity(), indent=2).encode(),
            'mixing_reference_provenance.json', 'application/json', key=f'mixing_reference_{origin}')
