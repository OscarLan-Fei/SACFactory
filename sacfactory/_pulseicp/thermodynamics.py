"""Sourced alloy-mixing descriptors, not supported-atom aggregation energies."""
from functools import lru_cache
from itertools import combinations
import math
import hashlib
import numpy as np

from .references import ROOT, snapshot, elements

SCHEMA = 'binary-mixing-v1'
R = 8.31446261815324
FEATURE_NAMES = ['mix_h_known_scaled', 'mix_entropy_over_R', 'mix_g_complete_scaled',
    'mix_pair_weight_coverage', 'mix_complete_available', 'mix_unlike_pairs_applicable',
    'mix_known_negative_pair_weight', 'mix_known_positive_pair_weight',
    'component_mix_h_known_scaled', 'component_mix_pair_weight_coverage']


@lru_cache(maxsize=1)
def mixing_table():
    data = snapshot('binary_mixing_enthalpies.json')
    table = {}
    # Preserve the pinned content check across Git's Windows newline conversion.
    raw = (ROOT/'resources/MiedemaLiquidDeltaHf.tsv').read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(raw).hexdigest() != data['upstream_sha256']:
        raise ValueError('Extended mixing table failed the SHA256 bound by the reference snapshot.')
    for line in raw.decode('utf-8').splitlines()[1:]:
        a,b,text = line.split()
        key = tuple(sorted((a,b)))
        value = float(text)
        if a == b or key in table or not math.isfinite(value):
            raise ValueError('Upstream mixing table contains duplicate, self or nonfinite pairs.')
        table[key] = value
    checked = set()
    for row in data['entries']:
        a, b = row['element_a'], row['element_b']
        key = tuple(sorted((a, b)))
        value = float(row['h_mix_equiatomic_kJ_mol_atoms'])
        if key in checked or key not in table or table[key] != value:
            raise ValueError('Scope table conflicts with the pinned upstream reference.')
        checked.add(key)
    return table


def reference_identity():
    # The scope snapshot contains the pinned upstream hash. Validating that raw
    # file extends coverage without changing any original pair or checkpoint ID.
    return {'schema': SCHEMA,
        'binary_table_sha256': snapshot('binary_mixing_enthalpies.json')['snapshot_sha256']}


def descriptors(concentrations_mM, temperature_C, table=None):
    """Compute actual-molar-fraction descriptors with explicit partial coverage.

    An injected table is useful for independently audited tables and missing-pair
    sensitivity tests. Unknown pairs never receive a fabricated zero enthalpy.
    """
    if not concentrations_mM or not math.isfinite(float(temperature_C)) or float(temperature_C) <= -273.15:
        raise ValueError('Provide a nonempty composition and temperature above absolute zero.')
    if any(e not in elements() or isinstance(v, bool) or not math.isfinite(float(v)) or float(v) <= 0
           for e, v in concentrations_mM.items()):
        raise ValueError('Use recognized elements and finite positive metal concentrations.')
    total = math.fsum(float(v) for v in concentrations_mM.values())
    if not math.isfinite(total):
        raise ValueError('Total concentration must be finite.')
    fractions = {e: float(concentrations_mM[e])/total for e in sorted(concentrations_mM)}
    table = mixing_table() if table is None else table
    pairs, missing = [], []
    known_h = known_weight = negative = positive = all_weight = 0.
    for a, b in combinations(fractions, 2):
        weight = fractions[a]*fractions[b]
        all_weight += weight
        value = table.get((a, b))
        known = value is not None and math.isfinite(float(value))
        if known:
            value = float(value)
            contribution = 4*weight*value
            known_h += contribution
            known_weight += weight
            negative += weight*(value < 0)
            positive += weight*(value > 0)
        else:
            contribution = None
            missing.append([a, b])
        pairs.append({'element_a': a, 'element_b': b, 'molar_fraction_a': fractions[a],
            'molar_fraction_b': fractions[b], 'h_mix_equiatomic_kJ_mol_atoms': value if known else None,
            'weighted_contribution_kJ_mol_atoms': contribution, 'available': known})
    entropy = -R*math.fsum(x*math.log(x) for x in fractions.values())
    complete = not missing
    return {'thermodynamic_schema': SCHEMA, 'molar_fractions': fractions,
        'h_mix_kJ_mol_atoms': known_h if complete else None,
        'h_mix_known_contribution_kJ_mol_atoms': known_h,
        's_mix_J_mol_K': entropy,
        'g_mix_proxy_kJ_mol_atoms': known_h-(float(temperature_C)+273.15)*entropy/1000 if complete else None,
        'pair_weight_coverage': known_weight/all_weight if all_weight else None,
        'complete': complete, 'unlike_pairs_applicable': bool(pairs),
        'known_negative_pair_weight_fraction': negative/all_weight if all_weight else 0.,
        'known_positive_pair_weight_fraction': positive/all_weight if all_weight else 0.,
        'pair_count': len(pairs), 'available_pair_count': len(pairs)-len(missing),
        'missing_pairs': missing, 'pairs': pairs,
        'interpretation': 'Alloy-mixing proxy; not a supported-atom aggregation energy or a calibrated XRD probability.'}


@lru_cache(maxsize=512)
def _context(concentrations, temperature_C):
    d = descriptors(dict(concentrations), temperature_C)
    local = {e: [0., 0., 0.] for e in d['molar_fractions']}
    for pair in d['pairs']:
        a,b = pair['element_a'],pair['element_b']
        for e,other in ((a,b),(b,a)):
            weight=d['molar_fractions'][other]
            local[e][1] += weight
            if pair['available']:
                local[e][2] += weight
                local[e][0] += 4*weight*pair['h_mix_equiatomic_kJ_mol_atoms']
    d['_component_context'] = local
    return d


def feature_vector(composition, element):
    d = _context(tuple(sorted(composition.concentrations_mM.items())), composition.temperature_C)
    local_h,local_weight,local_known = d['_component_context'][element]
    return np.asarray([d['h_mix_known_contribution_kJ_mol_atoms']/50., d['s_mix_J_mol_K']/R,
        (d['g_mix_proxy_kJ_mol_atoms'] or 0.)/50., d['pair_weight_coverage'] or 0.,
        float(d['complete']), float(d['unlike_pairs_applicable']),
        d['known_negative_pair_weight_fraction'], d['known_positive_pair_weight_fraction'],
        local_h/50., local_known/local_weight if local_weight else 0.], dtype=float)


def forecast_columns(composition, learned=False):
    d = _context(tuple(sorted(composition.concentrations_mM.items())), composition.temperature_C)
    return {k: d[k] for k in ('h_mix_kJ_mol_atoms', 's_mix_J_mol_K', 'g_mix_proxy_kJ_mol_atoms',
        'pair_weight_coverage', 'complete', 'thermodynamic_schema')} | {
        'thermodynamics_used_by_model': bool(learned),
        'thermodynamic_source_scope': 'ORIGINAL_REVIEWED_50' if all(e in snapshot('binary_mixing_enthalpies.json')['scope_elements'] for e in composition.elements) else 'EXTENDED_PINNED_UPSTREAM',
        'thermodynamic_table_sha256': reference_identity()['binary_table_sha256']}
