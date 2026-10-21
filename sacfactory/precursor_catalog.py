"""Offline catalogue candidates, separate from actual experimental chemistry."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=1)
def catalog():
    content = (Path(__file__).resolve().parents[1] / 'resources/sigma_precursors.json').read_bytes()
    data = json.loads(content)
    data['snapshot_sha256'] = hashlib.sha256(content).hexdigest()
    return data


def preference_key(entry):
    family = {'CHLORIDE': 0, 'NITRATE': 1}.get(entry.get('type'), 2)
    charge = entry.get('formal_oxidation_state')
    return family, float('inf') if charge is None else charge, entry.get('compound', '')


def catalog_rows(elements):
    candidates = {}
    for entry in catalog()['entries']:
        candidates.setdefault(entry['element'], []).append(entry)
    rows = []
    for element in elements:
        entries = sorted(candidates.get(element, []), key=preference_key)
        if entries:
            rows.append(dict(entries[0]))
        else:
            rows.append({'element': element, 'compound': None, 'type': 'UNKNOWN',
                         'formal_oxidation_state': None, 'cas': None, 'catalog_number': None, 'source_url': None,
                         'status': 'UNRESOLVED_NOT_IN_REVIEWED_TARGET50', 'notes': 'Enter an actual reagent or verify a source for this element.'})
    return rows


def planning_draft(elements):
    """Explicit planning assumptions only; no solution-state or actual-use claim."""
    return {row['element']: {'compound': row['compound'], 'type': row['type'],
            'oxidation_state': row['formal_oxidation_state'], 'coordination': 'UNKNOWN', 'spin': 'UNKNOWN', 'state_source': '',
            'source': f"PLANNED_CATALOG_ASSUMPTION: {row['source_url']} | {row['catalog_number']} | checked {row['checked_date']}. Actual reagent/hydrate not confirmed."}
            for row in catalog_rows(elements) if row.get('source_url') and row.get('type') == 'CHLORIDE'}


def reagent_defaults(saved=None):
    """Register reviewed chlorides without replacing user-edited reagent choices.

    The bounded catalogue supplies future defaults only. Missing catalogue
    identities stay unresolved; a nitrate is never an automatic fallback.
    """
    from .domain import TARGETS
    from .precursors import normalize_precursors
    defaults = planning_draft(TARGETS)
    defaults.update(saved or {})
    return normalize_precursors(defaults, list(defaults))


def resolve_planning_precursors(saved, context, elements):
    """Resolve future defaults identically in the UI and reservation validator."""
    import copy
    from .precursors import normalize_precursors
    library = reagent_defaults(saved)
    merged = copy.deepcopy(library)
    for element, entry in context.get('precursors', {}).items():
        if element in library and entry.get('source', '').startswith('PLANNED_CATALOG_ASSUMPTION:'):
            continue
        if entry.get('compound'):
            merged[element] = copy.deepcopy(entry)
    return normalize_precursors({e: value for e, value in merged.items() if e in elements}, elements)


def register_defaults(store):
    """Idempotently populate namespace-local material metadata, never samples."""
    record = store.get('meta', 'reagent_library') or {}
    saved = record.get('precursors', {})
    merged = reagent_defaults(saved)
    if merged != saved:
        store.put_many('meta', [{**record, 'id': 'reagent_library',
            'origin': store.origin, 'precursors': merged}], 'id', overwrite=True)
    return merged
