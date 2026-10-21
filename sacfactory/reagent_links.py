"""Resolve editable reagent defaults without asserting purchased product identity."""
import copy

from .precursor_catalog import reagent_defaults, catalog_rows
from .precursors import normalize_precursors, text


def link_precursors(sample, saved=None):
    """Fill absent declarations, retaining explicit different salts and states."""
    linked = copy.deepcopy(sample)
    library = reagent_defaults(saved)
    mapping = {e: copy.deepcopy(v) for e, v in sample.get('precursors', {}).items() if e in sample['elements']}
    for element in sample['elements']:
        default = library.get(element)
        if default is None:
            continue
        current = mapping.get(element, {})
        meaningful = any(text(current.get(k)) not in {'', 'UNKNOWN'} for k in
                          ['compound', 'type', 'oxidation_state', 'coordination', 'spin', 'state_source'])
        if not meaningful:
            mapping[element] = copy.deepcopy(default)
        elif text(current.get('compound')) == text(default.get('compound')):
            mapping[element] = {**default, **{k: v for k, v in current.items()
                if v is not None and text(v) not in {'', 'UNKNOWN'}}}
    linked['precursors'] = normalize_precursors(mapping, sample['elements'])
    return linked


def catalogue_links(sample):
    result = []
    for row in catalog_rows(sample['elements']):
        actual = sample.get('precursors', {}).get(row['element'], {}).get('compound')
        matched = bool(actual and actual == row.get('compound'))
        result.append({'element': row['element'], 'recorded_compound': actual or '',
            'catalogue_compound': row.get('compound'), 'catalogue_number': row.get('catalog_number') if matched else None,
            'CAS': row.get('cas') if matched else None, 'Sigma_URL': row.get('source_url') if matched else None,
            'link_status': 'Formula matches reference; bottle/hydrate not certified' if matched else
                'Explicit reagent differs or reference unavailable'})
    return result
