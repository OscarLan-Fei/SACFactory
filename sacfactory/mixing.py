"""Audited alloy-mixing proxies on nominal metal molar fractions.

These are composition descriptors, not supported-atom aggregation energies,
measured catalyst states, or an energy-to-screening-probability rule.
"""
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from .domain import ELEMENTS

R = 8.31446261815324
SCHEMA = 'sacfactory-feed-mixing-v1'
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / 'resources'
FEATURE_NAMES = ['mix_h_known_scaled', 'mix_entropy_over_R', 'mix_g_complete_scaled',
    'mix_pair_weight_coverage', 'mix_complete_available', 'mix_unlike_pairs_applicable',
    'mix_known_negative_pair_weight', 'mix_known_positive_pair_weight']
LABELS = {'h_mix_kJ_mol_atoms': 'Complete mixing H / kJ mol-1 atoms',
          'h_mix_known_contribution_kJ_mol_atoms': 'Known-pair H contribution / kJ mol-1 atoms',
          's_mix_J_mol_K': 'Ideal mixing S / J mol-1 K-1',
          'g_mix_proxy_kJ_mol_atoms': 'Complete H - T S proxy / kJ mol-1 atoms',
          'mix_pair_weight_coverage': 'Known unlike-pair weight fraction'}


@lru_cache(maxsize=1)
def reference():
    raw = (RESOURCE_ROOT / 'binary_mixing_enthalpies.json').read_bytes()
    result = json.loads(raw)
    result['snapshot_sha256'] = hashlib.sha256(raw).hexdigest()
    return result


@lru_cache(maxsize=1)
def mixing_table():
    """Validate the pinned full table and agreement with every reviewed scope row."""
    metadata = reference()
    raw = (RESOURCE_ROOT / 'MiedemaLiquidDeltaHf.tsv').read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata['upstream_sha256']:
        raise ValueError('The pinned binary-mixing TSV failed its reference SHA-256 check.')
    table = {}
    for line in raw.decode('utf-8').splitlines()[1:]:
        a, b, text = line.split()
        pair = tuple(sorted((a, b)))
        value = float(text)
        if a not in ELEMENTS or b not in ELEMENTS or a == b or pair in table or not math.isfinite(value):
            raise ValueError('Binary reference contains an invalid, duplicate or nonfinite pair.')
        table[pair] = value
    seen = set()
    for row in metadata['entries']:
        pair = tuple(sorted((row['element_a'], row['element_b'])))
        value = float(row['h_mix_equiatomic_kJ_mol_atoms'])
        if pair in seen or pair not in table or table[pair] != value:
            raise ValueError('Reviewed pair scope conflicts with the pinned raw table.')
        seen.add(pair)
    return table


def reference_identity():
    data = reference()
    return {'schema': SCHEMA, 'scope_snapshot_sha256': data['snapshot_sha256'],
            'raw_table_sha256': data['upstream_sha256'], 'upstream_commit': data['upstream_commit'],
            'primary_doi': data['primary_doi'], 'units': data['units'],
            'reference_state': data['reference_state'],
            'source_project': 'PulseICP; independently integrated without importing its application or checkpoints'}


def descriptors(concentrations_mM, temperature_C=None, table=None):
    """Use precursor metal mM, never ICP W, with explicit partial-pair coverage."""
    if not concentrations_mM or set(concentrations_mM) - set(ELEMENTS):
        raise ValueError('Provide a nonempty mapping of recognized target elements.')
    values = {}
    for element, raw in concentrations_mM.items():
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError('Positive numeric precursor metal concentrations are required.')
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite precursor metal concentrations are required.')
        values[element] = value
    total = math.fsum(values.values())
    if not math.isfinite(total):
        raise ValueError('Total precursor metal concentration must be finite.')
    fractions = {e: values[e] / total for e in sorted(values)}
    if any(x <= 0 for x in fractions.values()):
        raise ValueError('Molar fractions underflow; rescale the concentration inputs.')
    temperature = None if temperature_C is None else float(temperature_C) + 273.15
    if temperature is not None and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Controlled cutoff must be finite and above absolute zero.')
    table = mixing_table() if table is None else table
    pairs, missing = [], []
    known_h = known_weight = all_weight = negative = positive = 0.
    for a, b in combinations(fractions, 2):
        weight = fractions[a] * fractions[b]
        all_weight += weight
        raw = table.get((a, b))
        known = raw is not None and math.isfinite(float(raw))
        value = float(raw) if known else None
        contribution = 4. * weight * value if known else None
        if known:
            known_h += contribution
            known_weight += weight
            negative += weight * (value < 0)
            positive += weight * (value > 0)
        else:
            missing.append([a, b])
        pairs.append({'element_a': a, 'element_b': b,
            'molar_fraction_a': fractions[a], 'molar_fraction_b': fractions[b],
            'h_mix_equiatomic_kJ_mol_atoms': value,
            'weighted_contribution_kJ_mol_atoms': contribution, 'available': known})
    entropy = -R * math.fsum(x * math.log(x) for x in fractions.values())
    complete = not missing
    return {'thermodynamic_schema': SCHEMA, 'molar_fractions': fractions,
        'h_mix_kJ_mol_atoms': known_h if complete else None,
        'h_mix_known_contribution_kJ_mol_atoms': known_h,
        's_mix_J_mol_K': entropy,
        'g_mix_proxy_kJ_mol_atoms': known_h - temperature * entropy / 1000. if complete and temperature is not None else None,
        'mix_pair_weight_coverage': known_weight / all_weight if all_weight else None,
        'mix_complete': complete, 'mix_unlike_pairs_applicable': bool(pairs),
        'mix_known_negative_pair_weight': negative / all_weight if all_weight else 0.,
        'mix_known_positive_pair_weight': positive / all_weight if all_weight else 0.,
        'mix_pair_count': len(pairs), 'mix_available_pair_count': len(pairs) - len(missing),
        'missing_pairs': missing, 'pairs': pairs,
        'interpretation': 'Nominal metal-feed alloy-mixing proxy; not a supported-atom aggregation energy, final composition or calibrated success probability.'}


@lru_cache(maxsize=4096)
def _context(concentrations, cutoff):
    return descriptors(dict(concentrations), cutoff)


def sample_descriptors(sample):
    if set(sample['concentrations_mM']) != set(sample['elements']):
        raise ValueError('Concentration keys must match the full target composition.')
    if any(isinstance(v, (bool, np.bool_)) for v in sample['concentrations_mM'].values()):
        raise ValueError('Boolean values are not precursor concentrations.')
    inputs = tuple(sorted((e, float(v)) for e, v in sample['concentrations_mM'].items()))
    cutoff = sample.get('pulse_limit_C')
    return _context(inputs, None if cutoff is None else float(cutoff))


def feature_vector(sample):
    if sample.get('pulse_limit_C') is None:
        raise ValueError('Mixing prediction features require a controlled cutoff.')
    d = sample_descriptors(sample)
    return [d['h_mix_known_contribution_kJ_mol_atoms'] / 50., d['s_mix_J_mol_K'] / R,
        0. if d['g_mix_proxy_kJ_mol_atoms'] is None else d['g_mix_proxy_kJ_mol_atoms'] / 50.,
        d['mix_pair_weight_coverage'] or 0., float(d['mix_complete']), float(d['mix_unlike_pairs_applicable']),
        d['mix_known_negative_pair_weight'], d['mix_known_positive_pair_weight']]


def component_features(sample, element):
    d = sample_descriptors(sample)
    if element not in d['molar_fractions']:
        raise ValueError('The ICP target must belong to the recipe.')
    local_h = known = total = 0.
    for pair in d['pairs']:
        if element not in (pair['element_a'], pair['element_b']):
            continue
        other = pair['element_b'] if pair['element_a'] == element else pair['element_a']
        weight = d['molar_fractions'][other]
        total += weight
        if pair['available']:
            known += weight
            local_h += 4. * weight * pair['h_mix_equiatomic_kJ_mol_atoms']
    return [local_h / 50., known / total if total else 0.]


def summary(sample):
    try:
        d = sample_descriptors(sample)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return {**{k: None for k in LABELS}, 'mix_input_status': str(exc),
                'mix_complete': False, 'mix_unlike_pairs_applicable': len(sample.get('elements', [])) > 1}
    return {k: v for k, v in d.items() if k not in {'pairs', 'molar_fractions', 'interpretation', 'missing_pairs'}} | {
        'mix_missing_pairs': '; '.join('-'.join(pair) for pair in d['missing_pairs']),
        'mix_input_status': 'No unlike pairs: single element' if not d['mix_unlike_pairs_applicable'] else
            'Missing binary parameters' if not d['mix_complete'] else
            'H and S available; controlled cutoff missing' if sample.get('pulse_limit_C') is None else 'Complete reference coverage'}


def pair_coverage(elements):
    table = mixing_table()
    return pd.DataFrame([{'element_a': a, 'element_b': b,
        'h_mix_equiatomic_kJ_mol_atoms': table.get(tuple(sorted((a, b)))),
        'available': tuple(sorted((a, b))) in table} for a, b in combinations(elements, 2)],
        columns=['element_a', 'element_b', 'h_mix_equiatomic_kJ_mol_atoms', 'available'])


def forecast_metadata(sample, model_info=None, probability=None):
    info = model_info or {}
    return {**summary(sample), 'mixing_descriptors_used_by_model':
            info.get('descriptor_profile') in {'chemistry_mixing', 'chemistry_pulseicp'} and probability is not None,
            'mixing_reference_sha256': reference()['snapshot_sha256']}
