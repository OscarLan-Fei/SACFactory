"""Non-destructive complete-composition ICP ingestion and explicit eligibility audits."""
from dataclasses import dataclass
import copy
import numpy as np
import pandas as pd

from .schema import Recipe
from .features import features


@dataclass
class PreparedData:
    records: list
    audit: pd.DataFrame
    x: np.ndarray
    delta: np.ndarray
    w: np.ndarray


def prepare(frame, settings, origin, thermodynamic_features=False):
    from .composition import Composition, inverse_composition, component_features
    if origin not in {'SIMULATED', 'EXPERIMENTAL'}:
        raise ValueError('Choose a declared data origin.')
    groups = {}
    for i, row in enumerate(frame.to_dict('records')):
        groups.setdefault((str(row.get('origin', '')), str(row.get('sample_id', ''))), []).append((i, row))
    records, audits, x, delta, w = [], [], [], [], []
    for (_, ident), group in groups.items():
        reason = ''
        try:
            rows = [raw for _, raw in group]
            if rows[0].get('origin') != origin:
                raise ValueError('Different or undeclared data origin; never mixed into training.')
            if not ident.strip() or ident.lower() == 'nan':
                raise ValueError('Sample IDs must be nonblank.')
            members = [str(raw.get('element', '')).strip() for raw in rows]
            if len(set(members)) != len(members):
                raise ValueError('Each sample-element pair must be unique; duplicates never overwrite.')
            for key in ('batch_id', 'protocol_id'):
                text = str(rows[0].get(key, '')).strip()
                if not text or text.upper() in {'UNKNOWN', 'UNASSIGNED', 'NAN', '<NA>'}:
                    raise ValueError(f'An actual recorded {key} is required.')
            for key in ('batch_id', 'protocol_id', 'temperature_C', 'pulse_count', 'pan_mass_g', 'volume_mL', 'carbonization_loss_fraction', 'result_reported_at'):
                if len({str(raw.get(key)) for raw in rows}) != 1:
                    raise ValueError('Mixture rows need identical shared conditions and metadata: '+key)
            components = {str(raw['element']).strip(): Recipe.from_row(raw) for raw in rows}
            first = next(iter(components.values()))
            shared = lambda r:(r.temperature_C,r.pulse_count,r.pan_mass_g,r.volume_mL,r.carbonization_loss_fraction)
            if any(shared(r)!=shared(first) for r in components.values()):
                raise ValueError('Mixture components need identical converted process/material conditions.')
            composition = Composition({e: r.concentration_mM for e, r in components.items()},
                first.temperature_C, first.pulse_count, first.pan_mass_g, first.volume_mL, first.carbonization_loss_fraction).validate()
            observed = {}
            for raw in rows:
                if str(raw.get('icp_flag', '')).strip().upper() != 'MEASURED':
                    raise ValueError('Complete quantified MEASURED ICP is needed; ND/BLQ/missing stay audited, not imputed.')
                text = str(raw.get('icp_w_wt_percent', '')).strip()
                if text.startswith('<') or text.upper() in {'ND', 'BLQ', '', 'NAN', '<NA>'}:
                    raise ValueError('Unquantified or censored ICP is not imputed.')
                observed[str(raw['element']).strip()] = float(text)
            residual = inverse_composition(composition, observed, settings)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            reason = str(exc)
        for i, raw in group:
            audits.append({'row': i, 'sample_id': raw.get('sample_id'), 'batch_id': raw.get('batch_id'),
                'origin': raw.get('origin'), 'element': raw.get('element'),
                'icp_w_wt_percent_recorded': raw.get('icp_w_wt_percent'), 'eligible': not reason, 'exclusion_reason': reason})
            if not reason:
                e = str(raw['element']).strip()
                records.append({'raw': copy.deepcopy(raw), 'recipe': components[e], 'composition': composition})
                x.append(component_features(composition, e, settings, thermodynamic_features)); delta.append(residual[e]); w.append(observed[e])
    from .composition import INTERACTION_NAMES
    width = len(features(Recipe(), settings))+len(INTERACTION_NAMES)
    if thermodynamic_features:
        from .thermodynamics import FEATURE_NAMES as THERMODYNAMIC_NAMES
        width += len(THERMODYNAMIC_NAMES)
    return PreparedData(records, pd.DataFrame(audits).sort_values('row') if audits else pd.DataFrame(columns=['eligible','exclusion_reason']),
        np.asarray(x, dtype=np.float32).reshape(-1, width), np.asarray(delta, dtype=np.float32), np.asarray(w, dtype=float))


def batch_split(data, seed=20261014, protocol='Batch holdout'):
    batches = sorted({str(r['raw']['batch_id']) for r in data.records})
    if len(batches) < 6:
        raise ValueError('Need at least six actual batches for train/calibration/test separation.')
    if protocol == 'Chronological replay':
        times = {}
        for b in batches:
            dates = pd.to_datetime([r['raw'].get('result_reported_at') for r in data.records if str(r['raw']['batch_id']) == b], utc=True, errors='coerce')
            if dates.isna().any():
                raise ValueError('Chronological splitting needs explicit result timestamps for every eligible record.')
            times[b] = (dates.min(), dates.max())
        batches.sort(key=lambda b: times[b][0])
    elif protocol == 'Batch holdout':
        np.random.default_rng(seed).shuffle(batches)
    else:
        raise ValueError('Choose batch holdout or chronological replay.')
    a, b = max(2, int(len(batches)*.6)), max(1, int(len(batches)*.2))
    groups = [set(batches[:a]), set(batches[a:a+b]), set(batches[a+b:])]
    if protocol == 'Chronological replay':
        for left, right in zip(groups, groups[1:]):
            if max(times[g][1] for g in left) >= min(times[g][0] for g in right):
                raise ValueError('Batch reporting times overlap split boundaries; chronological leakage cannot be ruled out.')
    indices = [np.array([i for i, r in enumerate(data.records) if str(r['raw']['batch_id']) in g], dtype=int) for g in groups]
    if len({data.records[i]['raw']['sample_id'] for i in indices[0]}) < 16 or not len(indices[1]) or not len(indices[2]):
        raise ValueError('Need 16 quantified training specimens and nonempty calibration/test groups.')
    return indices, [{'role': role, 'batch_ids': sorted(g),
                      'sample_ids': sorted({data.records[i]['raw']['sample_id'] for i in ix})}
                     for role, g, ix in zip(('train', 'calibration', 'test'), groups, indices)]
