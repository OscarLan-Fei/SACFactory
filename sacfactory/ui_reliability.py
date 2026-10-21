"""Recipe evidence, repetition budgets and attributed descriptor coverage."""
import json
import pandas as pd
import streamlit as st

from .chemical_descriptors import coverage, reference
from .recipe_reliability import recipe_evidence
from .repetition_allocation import future_repeat_budget


def render_recipe_reliability(samples, methods, origin, key):
    st.write('**Recipe reliability and independent evidence**')
    rows = recipe_evidence(samples, methods)
    half_width = st.slider('Working repeat precision target', .05, .45, .15, .05, key=f'recipe_precision_{key}')
    cap = st.number_input('Maximum additional repeats per recipe', min_value=1, max_value=100, value=20, key=f'recipe_cap_{key}')
    table = []
    for row in rows:
        m, scale, capped = future_repeat_budget(row['a'], row['b'], row['independent_batches'], half_width, cap)
        table.append({**{k: v for k, v in row.items() if k not in {'recipe', 'source_sample_ids', 'a', 'b', 'note'}},
            'elements': ';'.join(row['recipe']['elements']), 'precursor_mM': json.dumps(row['recipe']['concentrations_mM'], sort_keys=True),
            'pulse_count': row['recipe']['pulse_count'], 'cutoff_C': row['recipe'].get('pulse_limit_C'),
            'planned_additional_repeats': m, 'expected_interval_scale': scale, 'repeat_cap_reached': capped,
            'source_sample_ids': ';'.join(row['source_sample_ids'])})
    if table:
        frame = pd.DataFrame(table)
        st.dataframe(frame, hide_index=True, width='stretch')
        st.download_button('Download recipe reliability and repeat budgets', frame.to_csv(index=False).encode(),
            f'{origin.lower()}_recipe_reliability.csv', 'text/csv', key=f'recipe_export_{key}')
    else:
        st.info('Resolved outcomes, actual batch IDs and complete controlled recipes are needed for recipe reliability.')
    st.caption('Observed repeatability and model uncertainty are reported separately.')
    with st.expander('Chemical descriptor sources and coverage'):
        elements = sorted({e for s in samples for e in s['elements']})
        st.dataframe(pd.DataFrame(coverage(elements)), hide_index=True, width='stretch')
        data = reference()
