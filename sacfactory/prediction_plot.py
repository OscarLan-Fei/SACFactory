"""Predicted probability versus observed frequency, with auditable source tables.

Figure contract: evaluate agreement of eligible forecasts with returned screening
outcomes; single quantitative validation panel, 105 x 100 mm, editable vector
text, at least 7 pt labels, batch-bootstrap intervals where supported. No claim
of atomic dispersion or guaranteed calibration. Regression is descriptive only.
"""
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
import json

import numpy as np
import pandas as pd

from .analysis import wilson_interval


TABLE_COLUMNS = ['group', 'point', 'n', 'independent_batches', 'mean_prediction', 'observed_pass_rate',
                 'lower_95', 'upper_95', 'interval_method']


def agreement_table(records, bins=5, by_stage=True, recipe_level=False, seed=42, draws=1000):
    data = records.loc[records['eligible']].dropna(subset=['label', 'p_screen_pass']).copy()
    data = data.loc[data['p_screen_pass'].between(0, 1) & data['label'].isin([0, 1])]
    data['group'] = data['element_count'].map(lambda n: f'{int(n)} element' if n == 1 else f'{int(n)} elements') if by_stage else 'All selected records'
    data['point'] = data['recipe_identity'] if recipe_level else pd.cut(data['p_screen_pass'], np.linspace(0, 1, int(bins)+1), include_lowest=True).astype(str)
    rows = []
    for (group, point), part in data.groupby(['group', 'point'], sort=True):
        batch = part.groupby('batch_id')['label'].agg(['sum', 'count'])
        if recipe_level and len(batch) < 2:
            continue
        if len(batch) >= 3:
            pick = np.random.default_rng(seed).integers(0, len(batch), (draws, len(batch)))
            rates = batch['sum'].to_numpy()[pick].sum(axis=1) / batch['count'].to_numpy()[pick].sum(axis=1)
            low, high = np.quantile(rates, [.025, .975])
            method = '95% actual-batch cluster bootstrap percentile'
        else:
            low, high = wilson_interval(int(part['label'].sum()), len(part))
            method = '95% sample Wilson, descriptive; batch dependence not corrected'
        rows.append({'group': group, 'point': point, 'n': len(part), 'independent_batches': len(batch),
            'mean_prediction': float(part['p_screen_pass'].mean()), 'observed_pass_rate': float(part['label'].mean()),
            'lower_95': float(low), 'upper_95': float(high), 'interval_method': method})
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def descriptive_line(table):
    """Sample-count weighted OLS on aggregated points; not a classifier score."""
    if len(table) < 3 or table['mean_prediction'].nunique() < 3:
        return None
    x, y, weights = [table[c].to_numpy(dtype=float) for c in ('mean_prediction', 'observed_pass_rate', 'n')]
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
    error = np.sum(weights * (y - intercept - slope*x)**2)
    total = np.sum(weights * (y - np.average(y, weights=weights))**2)
    return {'intercept': float(intercept), 'slope': float(slope), 'aggregated_r_squared': float(1-error/total) if total > 1e-12 else None,
            'points': len(x), 'note': 'Descriptive weighted fit of aggregated points; sensitive to bins and not prediction accuracy.'}


def figure_bundle(table, records, origin, show_fit=False):
    """Single-panel Python export; source observations and figure settings included."""
    from matplotlib.figure import Figure
    import matplotlib as mpl
    if table.empty:
        raise ValueError('No eligible agreement points to export.')
    fit = descriptive_line(table) if show_fit else None
    with mpl.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .8}):
        fig = Figure(figsize=(4.13385827, 3.93700787))
        ax = fig.subplots()
        fig.subplots_adjust(left=.18, bottom=.18, right=.96, top=.78)
        ax.plot([0, 1], [0, 1], '--', color='#87939d', linewidth=.8, label='Ideal agreement')
        colors = ['#097f7a', '#5470a6', '#ac7947', '#925b89', '#687a55', '#525d6b', '#bb6960']
        groups = list(table.groupby('group', sort=False))
        groups.sort(key=lambda item: int(item[0].split()[0]) if item[0].split()[0].isdigit() else 0)
        for i, (name, part) in enumerate(groups):
            # Draw actual interval endpoints directly: floating-point boundary
            # rounding or percentile intervals need not center on the estimate.
            ax.vlines(part['mean_prediction'],part['lower_95'],part['upper_95'],
                color=colors[i % len(colors)],linewidth=.8)
            for endpoint in ('lower_95','upper_95'):
                ax.hlines(part[endpoint],part['mean_prediction']-.006,part['mean_prediction']+.006,
                    color=colors[i % len(colors)],linewidth=.8)
            ax.scatter(part['mean_prediction'], part['observed_pass_rate'], s=16+4*np.sqrt(part['n']),
                       color=colors[i % len(colors)], marker=['o', 's', '^', 'D', 'v', 'P', 'X'][i % 7], label=name, zorder=3)
        if fit is not None:
            ax.plot([0, 1], [fit['intercept'], fit['intercept']+fit['slope']], color='#303841', linewidth=.8,
                    label='Descriptive weighted fit')
        ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel='Predicted screening success probability',
               ylabel='Observed screening success rate')
        ax.set_aspect('equal', adjustable='box')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.05), ncol=2, frameon=False, fontsize=7)
        display_origin = 'SIMULATED DEMO' if origin == 'DEMO' else origin
        fig.text(.18, .035, f'{display_origin} | n = {int(table.n.sum())} | Error bars: 95% intervals', fontsize=7)
        formats = {}
        buffer = BytesIO()
        fig.savefig(buffer, format='svg', metadata={'Creator': 'SACFactory'})
        formats['svg'] = buffer.getvalue()
        buffer = BytesIO()
        fig.savefig(buffer, format='pdf', metadata={'Creator': 'SACFactory'})
        formats['pdf'] = buffer.getvalue()
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=600)
        formats['png'] = buffer.getvalue()
    notes = {'origin': origin, 'figure_contract': 'Predicted probability versus observed frequency; screening, not atomic dispersion.',
        'records_before_eligibility_filter': len(records), 'eligible_records': int(records['eligible'].sum()),
        'plotted_observations': int(table.n.sum()), 'interval_methods': table.interval_method.unique().tolist(),
        'recipe_level_exclusion': 'Recipe plots require observations in at least two actual batches.',
        'fit': fit, 'grouping_note': 'Forecasts may come from successive model versions; x is their mean, not a retrospective refit.',
        'alignment': 'Not applicable: one plot area.', 'dimensions_mm': [105, 100]}
    archive = BytesIO()
    with ZipFile(archive, 'w', ZIP_DEFLATED) as zipfile:
        for name, content in {'prediction_agreement.svg': formats['svg'], 'prediction_agreement.pdf': formats['pdf'],
                              'prediction_agreement.png': formats['png']}.items():
            zipfile.writestr(name, content)
        zipfile.writestr('agreement_points.csv', table.to_csv(index=False))
        zipfile.writestr('prediction_audit.csv', records.to_csv(index=False))
        zipfile.writestr('figure_notes.json', json.dumps(notes, indent=2, allow_nan=False))
    return archive.getvalue()
