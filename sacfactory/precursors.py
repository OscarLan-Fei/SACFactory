"""Declared precursor chemistry and strictly state-matched reference descriptors."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

TYPES = ['UNKNOWN', 'CHLORIDE', 'NITRATE', 'ACETATE', 'ACETYLACETONATE', 'SULFATE', 'OXIDE', 'CARBONATE', 'OTHER']
SPINS = ['UNKNOWN', 'HS', 'LS', 'NOT_SPECIFIED_IN_REFERENCE']
COORDINATIONS = ['UNKNOWN', 'I', 'II', 'III', 'IIIPY', 'IV', 'IVSQ', 'IVPY', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIV']
FIELDS = ['compound', 'type', 'oxidation_state', 'coordination', 'spin', 'source', 'state_source']
COLUMNS = ['precursor_' + key for key in FIELDS]
PROFILES = {'periodic': 'Periodic position and identity', 'chemistry': 'Elemental properties',
            'chemistry_ion': 'Elemental properties + declared precursor oxidation states',
            'chemistry_precursor': 'Elemental properties + declared precursors',
            'chemistry_state': 'Elemental properties + precursors + explicit ionic states',
            'chemistry_thermal': 'Chemical states + sourced thermal properties',
            'chemistry_loss': 'Thermal properties + unfitted loss-prior features',
            'chemistry_mixing': 'Thermal properties + mixing H, S and G proxies',
            'chemistry_pulseicp': 'SAC-Gen physical controls, retention and mixing + chemical states'}


def base_profile(profile):
    return 'chemistry_state' if profile in {'chemistry_thermal', 'chemistry_loss', 'chemistry_mixing', 'chemistry_pulseicp'} else profile


def text(value):
    return '' if value is None or str(value).strip().lower() in {'nan', 'none', '<na>'} else str(value).strip()


def documented(value):
    value = text(value).upper()
    return bool(value and value not in {'UNKNOWN', 'UNASSIGNED', 'NOT_PROVIDED'} and not value.startswith(('FILL_IN_', 'REPLACE_')))


def normalize_precursors(mapping, elements):
    if not isinstance(mapping, dict) or set(mapping) - set(elements):
        raise ValueError('Precursor entries must belong to the selected target elements.')
    result = {}
    for element, raw in mapping.items():
        if not isinstance(raw, dict):
            raise ValueError(f'{element}: precursor information must be a field mapping.')
        entry = {key: text(raw.get(key)) for key in FIELDS}
        for key, choices in [('type', TYPES), ('coordination', COORDINATIONS), ('spin', SPINS)]:
            entry[key] = entry[key] or 'UNKNOWN'
            if entry[key] not in choices:
                raise ValueError(f'{element}: choose a supported precursor {key}.')
        value = entry['oxidation_state']
        entry['oxidation_state'] = None
        if value:
            try:
                number = float(value)
            except ValueError:
                raise ValueError(f'{element}: precursor oxidation state must be an integer or blank.') from None
            if not math.isfinite(number) or not number.is_integer() or not -8 <= number <= 8:
                raise ValueError(f'{element}: precursor oxidation state must be an integer from -8 to 8.')
            entry['oxidation_state'] = int(number)
        result[element] = entry
    return result


def from_row(row):
    return {key: row.get('precursor_' + key) for key in FIELDS}


def to_columns(sample, element):
    return {'precursor_' + key: sample.get('precursors', {}).get(element, {}).get(key, None if key == 'oxidation_state' else '') for key in FIELDS}


def chemistry_identity(sample):
    mapping = normalize_precursors(sample.get('precursors', {}), sample['elements'])
    # Documentation source changes do not alter a recipe. Unknown entries equal absence.
    result = []
    for e in sorted(mapping):
        values = tuple(mapping[e][k] for k in FIELDS[:5])
        if values != ('', 'UNKNOWN', None, 'UNKNOWN', 'UNKNOWN'):
            result.append((e, values))
    return tuple(result)


def confirm_catalog_precursors(sample):
    """Explicit actual-recipe confirmation; never an automatic catalog migration."""
    import copy
    updated = copy.deepcopy(sample)
    for entry in updated.get('precursors', {}).values():
        if text(entry.get('source')).startswith('PLANNED_CATALOG_ASSUMPTION:'):
            entry['source'] = 'Actual reagent identity confirmed by user; ' + entry['source']
    return updated


@lru_cache(maxsize=1)
def state_reference():
    content = (Path(__file__).resolve().parents[1] / 'resources/state_properties.json').read_bytes()
    data = json.loads(content)
    data['snapshot_sha256'] = hashlib.sha256(content).hexdigest()
    return data


def ionic_reference(element, entry):
    """Require an explicit state source and exact charge/geometry/spin match."""
    if not documented(entry.get('state_source')) or entry.get('oxidation_state') is None or entry.get('coordination') in (None, '', 'UNKNOWN') or entry.get('spin') in (None, '', 'UNKNOWN'):
        return None, 'STATE_UNRESOLVED'
    spin = '' if entry['spin'] == 'NOT_SPECIFIED_IN_REFERENCE' else entry['spin']
    matches = [r for r in state_reference()['ionic_radii'] if r['symbol'] == element
               and r['charge'] == entry['oxidation_state'] and r['coordination'] == entry['coordination']
               and (r['spin'] or '') == spin and r['ionic_radius'] is not None]
    if len(matches) != 1:
        return None, 'REFERENCE_UNAVAILABLE_OR_AMBIGUOUS'
    return matches[0], 'EXACT_REFERENCE_MATCH'


def precursor_coverage(sample, for_training=False):
    entries = normalize_precursors(sample.get('precursors', {}), sample['elements'])
    result = []
    for e in sample['elements']:
        entry = entries.get(e, normalize_precursors({e: {}}, [e])[e])
        if for_training and entry['source'].startswith('PLANNED_CATALOG_ASSUMPTION:'):
            entry = {**entry, 'source': ''}
        radius, status = ionic_reference(e, entry)
        result.append({'element': e, **entry, 'type_used_by_model': entry['type'] if documented(entry['source']) else 'UNKNOWN',
                       'ionic_radius_pm': radius['ionic_radius'] if radius else None, 'ionic_status': status,
                       'ionic_reference_row_id': radius['id'] if radius else None,
                       'ionic_origin_flag': radius['origin'] if radius else None,
                       'ionic_most_reliable': bool(radius['most_reliable']) if radius else None})
    return result


def precursor_features(sample, include_state=False, include_types=True, for_training=False):
    """Fixed, molarity-weighted chemistry moments; zeros encode masked unknowns."""
    rows = precursor_coverage(sample, for_training)
    weights = np.array([sample['concentrations_mM'][r['element']] for r in rows], dtype=float)
    weights /= weights.sum()
    families = [r['type_used_by_model'] for r in rows]
    features = [float(sum(w for w, family in zip(weights, families) if family == kind)) if include_types else 0. for kind in TYPES]
    # Unknown oxidation states are never inferred from the elemental database.
    def moments(key, known, scale):
        valid = np.array(known, dtype=bool)
        fraction = float(weights[valid].sum())
        if not fraction:
            return [0., 0., 0.]
        values = np.array([rows[i][key] / scale for i in np.flatnonzero(valid)])
        w = weights[valid] / fraction
        mean = float(w @ values)
        return [mean, float(np.sqrt(max(w @ (values*values) - mean*mean, 0))), fraction]
    features += moments('oxidation_state', [documented(r['source']) and r['oxidation_state'] is not None for r in rows], 8.)
    # Element-conditioned type moments distinguish swapping Fe/Co precursor families.
    from .chemical_descriptors import coverage
    properties = {row['element']: row for row in coverage(sample['elements'])}
    for kind in TYPES[1:]:
        for key, scale in [('en_pauling', 4.), ('covalent_radius_cordero', 250.)]:
            valid = [i for i, r in enumerate(rows) if families[i] == kind and properties[r['element']][key] is not None]
            fraction = float(weights[valid].sum())
            mean = sum(weights[i]*properties[rows[i]['element']][key]/scale for i in valid)/fraction if fraction else 0.
            features += [float(mean), fraction] if include_types else [0., 0.]
    if include_state:
        features += moments('ionic_radius_pm', [r['ionic_radius_pm'] is not None for r in rows], 200.)
    return features
