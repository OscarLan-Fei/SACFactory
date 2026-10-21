"""Recorded-condition associations and chemical outcome comparisons, without relabeling."""
import pandas as pd

from .analysis import PARAMETERS, experiment_frame
from .descriptor_analysis import LABELS, descriptor_report
from .domain import ELEMENTS, screen
from .manual import actual_batch_recorded
from .mixing import LABELS as MIXING_LABELS


FAMILIES = ['Concentrations', 'Process conditions', 'Composition', 'Chemical means',
            'Chemical dispersion', 'Precursor fractions', 'Descriptor coverage', 'Target membership', 'Mixing thermodynamics']


def enriched_frame(samples, methods, origin, elements=()):
    records = [s for s in samples if s.get('origin') == origin]
    base = experiment_frame(records, {s['sample_id']: screen(s, methods) for s in records})
    report = descriptor_report(records, methods, origin)
    chemistry = report['recipes']
    if not chemistry.empty:
        extra = [c for c in chemistry if c not in base or c == 'sample_id']
        base = base.merge(chemistry[extra], on='sample_id', how='left', validate='one_to_one')
    additions = {}
    for e in elements:
        if e not in ELEMENTS:
            raise ValueError('Select a periodic-table element.')
        if e + '_mM' not in base:
            additions[e + '_mM'] = float('nan')
        additions[e + '_present'] = [int(e in sample['elements']) for sample in records]
    return pd.concat([base, pd.DataFrame(additions, index=base.index)], axis=1).copy()


def variable_groups(frame, elements):
    return {
        'Concentrations': [e + '_mM' for e in elements if e + '_mM' in frame],
        'Process conditions': list(PARAMETERS),
        'Composition': ['element_count', 'total_concentration_mM'],
        'Mixing thermodynamics': [key for key in MIXING_LABELS if key in frame],
        'Chemical means': [key + '_mean' for key in LABELS],
        'Chemical dispersion': [key + '_sd' for key in LABELS],
        'Precursor fractions': [c for c in frame if c.startswith('salt_') and c.endswith('_fraction')],
        'Descriptor coverage': [key + '_coverage' for key in LABELS],
        'Target membership': [e + '_present' for e in elements],
    }


def variable_label(column):
    if column in MIXING_LABELS:
        return MIXING_LABELS[column]
    if column in PARAMETERS:
        return PARAMETERS[column]
    if column == 'label':
        return 'Screening outcome (PASS=1, FAIL=0)'
    if column == 'element_count':
        return 'Target element count'
    if column == 'total_concentration_mM':
        return 'Total precursor concentration / mM'
    for suffix, prefix in [('_mean', 'Weighted mean: '), ('_sd', 'Weighted dispersion: '), ('_coverage', 'Known concentration fraction: ')]:
        key = column.removesuffix(suffix)
        if column.endswith(suffix) and key in LABELS:
            return prefix + LABELS[key]
    if column.endswith('_mM'):
        return column[:-3] + ' concentration / mM'
    if column.endswith('_present'):
        return column[:-8] + ' target membership (0/1)'
    return column.replace('_', ' ')


def outcome_groups(frame, metric, branches=False):
    """Keep unresolved/conflicting outcomes outside PASS/FAIL distributions."""
    resolved = frame.loc[frame['label'].notna() & ~frame['conflict']].copy()
    resolved['outcome_group'] = resolved['status'] if branches else resolved['label'].map({1: 'Screen PASS', 0: 'Screen FAIL'})
    rows = []
    for group, part in resolved.groupby('outcome_group', sort=False):
        valid = part.dropna(subset=[metric])
        values = valid[metric]
        rows.append({'outcome_group': group, 'resolved_records': len(part), 'metric_observed': len(valid),
                     'metric_missing': len(part) - len(valid),
                     'named_batches_with_metric': len({b for b in valid.batch_id if actual_batch_recorded({'batch_id': b})}),
                     'median': float(values.median()) if len(valid) else None,
                     'q25': float(values.quantile(.25)) if len(valid) else None,
                     'q75': float(values.quantile(.75)) if len(valid) else None,
                     'minimum': float(values.min()) if len(valid) else None,
                     'maximum': float(values.max()) if len(valid) else None})
    return resolved, pd.DataFrame(rows, columns=['outcome_group', 'resolved_records', 'metric_observed', 'metric_missing',
        'named_batches_with_metric', 'median', 'q25', 'q75', 'minimum', 'maximum'])
