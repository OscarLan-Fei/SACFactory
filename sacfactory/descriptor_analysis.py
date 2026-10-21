"""Read-only, source-aware chemical coverage and physical recipe summaries."""
import math

import numpy as np
import pandas as pd

from .chemical_descriptors import PROPERTIES, coverage, reference
from .precursors import PROFILES, TYPES, documented, precursor_coverage, state_reference, base_profile
from .domain import screen


ELEMENT_PROPERTIES = PROPERTIES + ['atomic_radius_rahm', 'melting_point_K', 'boiling_point_K', 'heat_of_formation_kJ_mol']
LABELS = {'atomic_weight': 'Atomic weight / Da', 'en_pauling': 'Pauling electronegativity',
          'covalent_radius_cordero': 'Cordero covalent radius / pm', 'electron_affinity': 'Electron affinity / eV',
          'first_ionization_energy': 'First ionization energy / eV', 'atomic_radius_rahm': 'Rahm atomic radius / pm',
          'formal_oxidation_state': 'Declared formal oxidation state', 'ionic_radius_pm': 'Matched ionic radius / pm',
          'melting_point_K': 'Elemental melting point / K', 'boiling_point_K': 'Elemental boiling point / K',
          'heat_of_formation_kJ_mol': 'Gas-atom formation energy proxy / kJ mol-1'}


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def weighted_summary(values, weights):
    """Known-only physical moments, with coverage; absent inputs stay unavailable."""
    if weights is None:
        return None, None, None
    known = [i for i, value in enumerate(values) if number(value) is not None]
    fraction = float(sum(weights[i] for i in known))
    if not fraction:
        return None, None, 0.
    w = np.array([weights[i] / fraction for i in known])
    v = np.array([float(values[i]) for i in known])
    mean = float(w @ v)
    return mean, float(np.sqrt(max(w @ ((v - mean) ** 2), 0.))), fraction


def association_availability(frame, variables):
    rows = []
    for key in variables:
        values = pd.to_numeric(frame[key], errors='coerce').dropna()
        count, distinct = len(values), values.nunique()
        status = 'Available' if count >= 3 and distinct >= 2 else 'No recorded values' if not count else 'Constant values' if distinct < 2 else 'Fewer than three observations'
        requirement = ('Recorded precursor formula and documented formal oxidation state; future defaults are separate'
                       if key.startswith('formal_oxidation_state') else
                       'Source-backed charge, coordination and spin matching a reference row; square-planar assumptions must be declared'
                       if key.startswith('ionic_radius_pm') else 'Paired nonmissing observations and variation')
        rows.append({'variable': key, 'observed': count, 'distinct_values': distinct, 'status': status, 'requirement': requirement})
    return pd.DataFrame(rows)


def profile_manifest(profile):
    if profile not in PROFILES:
        raise ValueError('Select a supported descriptor profile.')
    selected_profile = profile
    profile = base_profile(profile)
    rows = [{'descriptor': 'Periodic member/pair representation', 'unit': 'encoding', 'enabled': True,
             'requirement': 'Element identity', 'source': 'composition.py; fixed periodic-position encoding'}]
    ref = reference()
    for key in ELEMENT_PROPERTIES:
        if key in {'melting_point_K', 'boiling_point_K', 'heat_of_formation_kJ_mol'}:
            continue
        meta = ref['properties'].get(key, {'unit': 'pm', 'reference': '; '.join(state_reference()['references']['atomic_radius_rahm'])})
        rows.append({'descriptor': LABELS[key], 'unit': meta['unit'], 'enabled': profile != 'periodic',
                     'requirement': 'Available sourced elemental value; missing mask retained', 'source': meta['reference']})
    for key, enabled, requirement in [
        ('formal_oxidation_state', profile in {'chemistry_ion', 'chemistry_precursor', 'chemistry_state'}, 'Declared precursor state and documented source'),
        ('Salt family and element-conditioned salt moments', profile in {'chemistry_precursor', 'chemistry_state'}, 'Declared family and documented precursor source'),
        ('ionic_radius_pm', profile == 'chemistry_state', 'Exact oxidation/coordination/spin match and documented state evidence')]:
        rows.append({'descriptor': LABELS.get(key, key), 'unit': 'pm' if key == 'ionic_radius_pm' else 'encoding / dimensionless',
                     'enabled': enabled, 'requirement': requirement,
                     'source': '; '.join(state_reference()['references']['ionic_radius']) if key == 'ionic_radius_pm' else 'Actual reagent declarations; no catalogue backfill'})
    from .physics_features import thermal_manifest_rows
    rows += thermal_manifest_rows(selected_profile)
    rows.append({'descriptor':'PulseICP physical-control, retained-mass and mixing component moments with availability mask',
        'unit':'Fixed 50-feature encoding plus availability', 'enabled':selected_profile == 'chemistry_pulseicp',
        'requirement':'Recorded PAN/DMF, controlled cutoff and supported elemental thermal references; frozen physical assumptions',
        'source':'Vendored PulseICP SOURCE_MANIFEST.json and shared configs/physics.json; no measured outcomes used'})
    from .mixing import LABELS as MIXING_LABELS, reference_identity
    for key, label in MIXING_LABELS.items():
        rows.append({'descriptor': label, 'unit': 'fraction' if 'coverage' in key else 'J/(mol K)' if key == 's_mix_J_mol_K' else 'kJ/mol atoms',
                     'enabled': selected_profile in {'chemistry_mixing', 'chemistry_pulseicp'},
                     'requirement': 'Nominal metal molar fractions; complete binary coverage for full H and G; cutoff required for G',
                     'source': reference_identity()['primary_doi'] + '; pinned matminer Miedema table; ideal configurational entropy'})
    return pd.DataFrame(rows)


def descriptor_report(samples, methods, origin, profile='chemistry_ion'):
    """Honor namespace/selection; do not merge future defaults into old samples."""
    if origin not in {'REAL', 'DEMO'}:
        raise ValueError('Select REAL or DEMO.')
    manifest = profile_manifest(profile)
    records = [s for s in samples if s.get('origin') == origin]
    elements = list(dict.fromkeys(e for s in records for e in s['elements']))
    elemental = pd.DataFrame(coverage(elements))
    properties = {r['element']: r for r in elemental.to_dict('records')}
    member_rows, summaries = [], []
    for sample in records:
        members = []
        for row in precursor_coverage(sample, for_training=True):
            raw = sample.get('precursors', {}).get(row['element'], {})
            members.append({'sample_id': sample['sample_id'], 'batch_id': sample.get('batch_id', 'UNKNOWN'),
                **row, **properties[row['element']], 'concentration_mM': number(sample['concentrations_mM'].get(row['element'])),
                'formal_oxidation_state': row['oxidation_state'] if documented(row['source']) else None,
                'declared_source': raw.get('source', ''),
                'reference_state_assumed': str(raw.get('state_source', '')).startswith('MODEL_ASSUMPTION:'),
                'declaration_status': 'Catalog assumption' if raw.get('source', '').startswith('PLANNED_CATALOG_ASSUMPTION:') else 'Source-backed declaration' if documented(row['source']) else 'Unrecorded or unsourced'})
        member_rows.extend(members)
        concentrations = [r['concentration_mM'] for r in members]
        complete = bool(concentrations) and all(v is not None and v > 0 for v in concentrations)
        total = sum(concentrations) if complete else None
        if total is not None and not math.isfinite(total):
            complete, total = False, None
        weights = np.array(concentrations) / total if complete else None
        outcome = screen(sample, methods)
        summary = {'sample_id': sample['sample_id'], 'batch_id': sample.get('batch_id', 'UNKNOWN'),
            'composition': ';'.join(sample['elements']), 'element_count': len(members), 'status': outcome['status'],
            'label': None if outcome['review_conflicts'] else outcome['label'],
            'conflict': bool(outcome['review_conflicts']), 'complete_concentrations': complete, 'total_concentration_mM': total}
        for key in LABELS:
            mean, sd, fraction = weighted_summary([r[key] for r in members], weights)
            summary.update({key + '_mean': mean, key + '_sd': sd, key + '_coverage': fraction})
        for family in TYPES:
            summary['salt_' + family.lower() + '_fraction'] = float(sum(weights[i] for i, r in enumerate(members) if r['type_used_by_model'] == family)) if complete else None
        from .mixing import summary as mixing_summary
        summary.update(mixing_summary(sample))
        summaries.append(summary)
    members = pd.DataFrame(member_rows)
    cells = []
    for element in elements:
        part = members.loc[members.element == element]
        for key in list(LABELS) + ['salt_family']:
            known = int((part.type_used_by_model != 'UNKNOWN').sum()) if key == 'salt_family' else int(part[key].notna().sum())
            cells.append({'element': element, 'descriptor': LABELS.get(key, 'Source-backed salt family'),
                          'known_memberships': known, 'memberships': len(part), 'available_fraction': known / len(part)})
    provenance = {'origin': origin, 'profile': profile, 'sample_ids': [s['sample_id'] for s in records],
        'descriptor_snapshot_sha256': reference()['snapshot_sha256'], 'state_descriptor_snapshot_sha256': state_reference()['snapshot_sha256'],
        'summary_definition': 'Molar-concentration-weighted physical means and standard deviations over known values; coverage reported separately. These are explanatory summaries, not the normalized kernel feature vectors.',
        'missing_rule': 'Missing properties are unavailable. Missing/nonpositive concentrations leave all weighted moments unavailable. No historical backfill from planning defaults.',
        'interpretation': 'Descriptive associations are not feature importance, causal effects, independent validation or proof of atomic dispersion.'}
    from .physics_features import physics_manifest
    provenance['physics'] = physics_manifest(profile)
    from .mixing import reference_identity
    provenance['analysis_mixing_reference'] = reference_identity()
    return {'elemental': elemental, 'members': members, 'recipes': pd.DataFrame(summaries),
            'coverage': pd.DataFrame(cells), 'manifest': manifest, 'provenance': provenance}
