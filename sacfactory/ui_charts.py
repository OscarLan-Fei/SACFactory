"""Category-aware analysis charts with explicit axes and scrollable matrices."""
import altair as alt
import pandas as pd
import streamlit as st


def category_axis(horizontal=False):
    return alt.Axis(labelOverlap=False, labelLimit=220, labelAngle=-65 if horizontal else 0)


def row_height(count, minimum=140, step=32):
    return max(minimum, int(count) * step)


def correlation_grid(pairs, variables):
    values = {}
    for row in pairs.to_dict('records'):
        values[row['parameter_1'], row['parameter_2']] = row
        values[row['parameter_2'], row['parameter_1']] = row
    rows = []
    for x in variables:
        for y in variables:
            row = values.get((x, y), {})
            rho = row.get('spearman_rho')
            valid = rho is not None and pd.notna(rho)
            rows.append({'parameter_1': x, 'parameter_2': y,
                         'paired_n': row.get('paired_n'), 'spearman_rho': rho if valid else None,
                         'availability': 'Available' if valid else 'Diagonal not evaluated' if x == y else 'Constant or insufficient paired observations'})
    return pd.DataFrame(rows)


def matrix_chart(frame, x, y, value, x_order, y_order, title, scheme='tealblues', domain=None):
    # Explicit inline data avoids Altair's 5,000-row transformer limit for 118 x 118.
    data = alt.Data(values=frame.astype(object).where(pd.notna(frame), None).to_dict('records'))
    if domain is None:
        observed = pd.to_numeric(frame[value], errors='coerce').dropna()
        domain = [0, max(1., float(observed.max()))] if len(observed) else [0, 1]
    scale = alt.Scale(scheme=scheme, domain=domain)
    return alt.Chart(data).mark_rect(stroke='#ffffff', strokeWidth=.3).encode(
        x=alt.X(f'{x}:N', title=None, sort=list(x_order), scale=alt.Scale(domain=list(x_order)), axis=category_axis(True)),
        y=alt.Y(f'{y}:N', title=None, sort=list(y_order), scale=alt.Scale(domain=list(y_order)), axis=category_axis()),
        color=alt.condition(f'isValid(datum.{value})', alt.Color(f'{value}:Q', title=title, scale=scale), alt.value('#dce3ec')),
        tooltip=[alt.Tooltip(c, type='quantitative' if c in {value, 'paired_n'} else 'nominal') for c in frame.columns],
    ).properties(width=alt.Step(34), height=alt.Step(32))


def render_matrix(chart, rows, key):
    with st.container(height=min(680, row_height(rows) + 180), border=False, key='matrix_scroll_' + key):
        st.altair_chart(chart, width='content')
    st.caption('Scroll large matrices to inspect all selected labels. Gray cells are unavailable or intentionally unevaluated, not zero-valued measurements.')


def interaction_chart(frame, x_title, y_title):
    """Reserve space per bin instead of fitting axes inside a small total height."""
    x_order = frame.sort_values('x_order')['x_range'].drop_duplicates().tolist()
    y_order = frame.sort_values('y_order')['y_range'].drop_duplicates().tolist()
    chart = matrix_chart(frame, 'x_range', 'y_range', 'pass_rate', x_order, y_order,
                         'Candidate pass rate', domain=[0, 1])
    chart = chart.encode(
        x=alt.X('x_range:N', title=x_title, sort=x_order, scale=alt.Scale(domain=x_order),
                axis=alt.Axis(labelOverlap=False, labelAngle=0, labelLimit=180)),
        y=alt.Y('y_range:N', title=y_title, sort=y_order, scale=alt.Scale(domain=y_order),
                axis=category_axis()),
        tooltip=['x_range:N', 'y_range:N', 'n:Q', 'passed:Q', 'failed:Q', 'independent_batches:Q',
                 alt.Tooltip('pass_rate:Q', format='.1%'), alt.Tooltip('lower_95:Q', format='.1%'), alt.Tooltip('upper_95:Q', format='.1%')],
    ).properties(width=alt.Step(180), height=alt.Step(52))
    if 'cell_label' not in frame:
        return chart
    text = chart.mark_text(fontSize=12).encode(text='cell_label:N',
        color=alt.condition('datum.pass_rate > 0.65', alt.value('white'), alt.value('#183749')))
    return chart + text


def chart_page(frame, key, size=24):
    if len(frame) <= size:
        return frame
    pages = (len(frame) + size - 1) // size
    page = st.selectbox('Chart page', range(pages), format_func=lambda i: f'{i + 1} / {pages}', key=key)
    st.caption(f'Chart rows {page * size + 1}–{min((page + 1) * size, len(frame))} of {len(frame)}. Tables and exports retain all rows.')
    return frame.iloc[page * size:(page + 1) * size]
