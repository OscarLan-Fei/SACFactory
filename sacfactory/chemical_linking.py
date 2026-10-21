"""Explicit user-requested linkage to declared reagents; no startup backfill."""
from copy import deepcopy
import re

from .precursors import documented, normalize_precursors, state_reference


def link_declared_chlorides(samples, library, origin, square_planar=False):
    updated, report = [], []
    for original in samples:
        if original.get('origin') != origin:
            continue
        sample = deepcopy(original)
        mapping = sample.setdefault('precursors', {})
        for element in sample['elements']:
            actual = mapping.get(element, {})
            declared = library.get(element, {})
            if not actual.get('compound'):
                if not documented(declared.get('source')) or str(declared['source']).startswith('PLANNED_CATALOG_ASSUMPTION:'):
                    continue
                actual = deepcopy(declared)
            if str(actual.get('source', '')).startswith('PLANNED_CATALOG_ASSUMPTION:'):
                continue
            formula = str(actual.get('compound', '')).split('·')[0].replace(' ', '')
            match = re.fullmatch(re.escape(element) + r'Cl([1-8]?)', formula)
            if not match:
                continue
            charge = int(match.group(1) or 1)
            if actual.get('oxidation_state') not in (None, charge):
                report.append({'sample_id': sample['sample_id'], 'element': element, 'status': 'Existing charge conflicts with chloride formula; unchanged'})
                continue
            actual.update(type='CHLORIDE', oxidation_state=charge)
            note = 'User-requested linkage of recorded chloride identity; formal charge from neutral formula.'
            if note not in str(actual.get('source', '')):
                actual['source'] = str(actual.get('source', '')) + '; ' + note
            if square_planar and actual.get('coordination', 'UNKNOWN') in ('UNKNOWN', '', None):
                actual['coordination'] = 'IVSQ'
                actual['state_source'] = 'MODEL_ASSUMPTION: User requested four-coordinate square-planar reference on 2026-10-13; not measured coordination or final catalyst oxidation state.'
                matches = [r for r in state_reference()['ionic_radii'] if r['symbol'] == element and r['charge'] == charge and r['coordination'] == 'IVSQ']
                # Unspecified spin is valid only when the unique reference row itself has no spin label.
                if len(matches) == 1 and not matches[0]['spin'] and actual.get('spin', 'UNKNOWN') == 'UNKNOWN':
                    actual['spin'] = 'NOT_SPECIFIED_IN_REFERENCE'
            if (square_planar and actual.get('coordination') == 'IVSQ' and actual.get('spin', 'UNKNOWN') == 'UNKNOWN'
                    and declared.get('compound') == actual.get('compound')
                    and str(declared.get('state_source', '')).startswith('MODEL_ASSUMPTION:')
                    and declared.get('coordination') == 'IVSQ' and declared.get('spin') in ('HS', 'LS')):
                actual['spin'], actual['state_source'] = declared['spin'], declared['state_source']
            mapping[element] = normalize_precursors({element: actual}, [element])[element]
            report.append({'sample_id': sample['sample_id'], 'element': element, 'compound': formula,
                           'formal_oxidation_state': charge, 'coordination': mapping[element]['coordination'],
                           'spin': mapping[element]['spin'], 'status': 'Linked user declaration; square-planar state is a model assumption' if square_planar else 'Linked user declaration'})
        if sample != original:
            updated.append(sample)
    return updated, report
