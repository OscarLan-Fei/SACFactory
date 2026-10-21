"""Audited, atomic table editing for experimental records."""
import copy

import pandas as pd
import streamlit as st

from .domain import SAMPLE_CONDITIONS
from .manual import element_rows, apply_element_rows, save_manual_revisions
from .planning import fingerprint
from .reagent_links import link_precursors, catalogue_links
from .precursors import to_columns
from .ui_chemistry import chemistry_columns
from .ui_manual import clean_cells, saved
from .preparation import from_mM, to_mM
import math

CONDITION_FIELDS = ['batch_id', 'pulse_count', 'pulse_limit_C', 'temperature_C', 'voltage_V',
    'current_A', 'pan_mass_g', 'solvent_volume_mL', 'precursor_family', 'support_batch', 'protocol_id', 'notes']


def condition_rows(samples):
    return [{'sample_id': s['sample_id'],
        'concentration_summary_mM': '; '.join(f'{e}: {c:g}' for e, c in s['concentrations_mM'].items() if c is not None),
        'amount_summary_mmol': '; '.join(f'{e}: {from_mM(c, "mmol", s["solvent_volume_mL"]):g}' for e, c in s['concentrations_mM'].items() if c is not None) if s.get('solvent_volume_mL') else '',
        **{k: s.get(k) for k in CONDITION_FIELDS},
        'sample_condition': s.get('sample_condition') or 'NOT_RECORDED',
        'icp_state': s.get('icp_review', {}).get('state', 'AUTO'),
        'xrd_state': s.get('xrd', {}).get('state', 'PENDING'),
        'stem_state': s.get('stem', {}).get('state', 'NOT_TESTED')} for s in samples]


def record_element_rows(samples):
    """Expose both units without changing the canonical stored metal concentration."""
    rows = []
    for sample in samples:
        volume = sample.get('solvent_volume_mL')
        for row in element_rows(sample):
            amount = from_mM(row['precursor_mM'], 'mmol', volume) if volume else None
            rows.append({'sample_id': sample['sample_id'], 'element':row['element'],
                'precursor_mM':row['precursor_mM'], 'precursor_mmol':amount, **row})
    return rows


def reconcile_record_quantity(row, original, volume):
    """A single edited quantity controls conversion; simultaneous edits must agree."""
    result = dict(row)
    if 'precursor_mmol' not in row:
        return result
    concentration_changed = row.get('precursor_mM') != original.get('precursor_mM')
    amount_changed = row.get('precursor_mmol') != original.get('precursor_mmol')
    if amount_changed:
        converted = to_mM(row.get('precursor_mmol'), 'mmol', volume)
        if converted is None:
            raise ValueError('Enter a precursor amount; leave both quantities unchanged to preserve the record.')
        if concentration_changed and not math.isclose(float(row['precursor_mM']), converted, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError('Precursor mM and mmol disagree with DMF volume. Edit one quantity or supply consistent values.')
        result['precursor_mM'] = converted
    result.pop('precursor_mmol', None)
    return result


def apply_record_tables(samples, conditions, details, original_details):
    """Revise only edited sample rows; preserve unchanged outcomes and extras."""
    originals = {s['sample_id']: s for s in samples}
    original_conditions = {r['sample_id']: r for r in condition_rows(samples)}
    original_cells = {(r['sample_id'], r['element']): r for r in original_details}
    supplied = {r['sample_id']: [] for r in details}
    seen = set()
    for row in details:
        key = row['sample_id'], row['element']
        if key not in original_cells or key in seen:
            raise ValueError('Element rows must retain their sample and element identities.')
        seen.add(key)
        supplied[row['sample_id']].append(row)
    if seen != set(original_cells):
        raise ValueError('Keep every selected element row; change target elements in the individual editor.')
    changed, ids = [], set()
    for row in conditions:
        ident = row['sample_id']
        if ident not in originals or ident in ids:
            raise ValueError('Condition rows must retain unique existing sample identities.')
        ids.add(ident)
        edited_cells = supplied.get(ident)
        cell_changed = edited_cells is not None and any(r != original_cells[(ident, r['element'])] for r in edited_cells)
        cell_changed = cell_changed or (edited_cells is not None and any(
            any(r.get(k) != value for k, value in to_columns(originals[ident], r['element']).items()) for r in edited_cells))
        if row == original_conditions[ident] and not cell_changed:
            continue
        old = originals[ident]
        new = copy.deepcopy(old)
        new.update({k: row[k] for k in CONDITION_FIELDS})
        new['sample_condition'] = row['sample_condition']
        if cell_changed:
            reconciled = [reconcile_record_quantity(r, original_cells[(ident, r['element'])],
                new.get('solvent_volume_mL')) for r in edited_cells]
            new = apply_element_rows(new, reconciled)
        for field, branch in [('icp_state', 'icp_review'), ('xrd_state', 'xrd'), ('stem_state', 'stem')]:
            if row[field] != original_conditions[ident][field]:
                new[branch] = {**new.get(branch, {}), 'state': row[field]}
                if branch != 'stem':
                    new[branch].update(source='User judgment in experiment record table')
                if branch == 'xrd':
                    new[branch]['mode'] = 'MANUAL'
        changed.append(new)
    if ids != set(originals):
        raise ValueError('Keep every displayed condition row.')
    return changed


def render_record_tables(store, samples, methods):
    if not samples:
        return
    origin = store.origin
    st.write('**Edit experiment records directly**')
    st.caption('Edit precursor mmol or mM; the other quantity updates using DMF volume. ICP is W / wt%. Save to apply changes.')
    ids = st.multiselect('Element-detail samples to edit', [s['sample_id'] for s in samples],
        default=[samples[0]['sample_id']], key=f'record_table_details_{origin}')
    library = (store.get('meta', 'reagent_library') or {}).get('precursors', {})
    detailed = [link_precursors(s, library) for s in samples if s['sample_id'] in ids]
    details = record_element_rows(detailed)
    baseline = clean_cells(details)
    sig = fingerprint([samples, ids, library])
    with st.form(f'record_tables_{origin}_{sig}'):
        conditions = st.data_editor(pd.DataFrame(condition_rows(samples)), hide_index=True, width='stretch',
            disabled=['sample_id', 'concentration_summary_mM', 'amount_summary_mmol'], key=f'record_conditions_{origin}_{sig}',
            column_config={'sample_condition': st.column_config.SelectboxColumn('Sample condition', options=list(SAMPLE_CONDITIONS)),
                'concentration_summary_mM': st.column_config.TextColumn('Metal concentration / mM', help='Edit the selected element rows below.'),
                'amount_summary_mmol': st.column_config.TextColumn('Metal amount / mmol', help='Derived using each sample DMF volume. Edit element rows below.'),
                'icp_state': st.column_config.SelectboxColumn('ICP judgment', options=['AUTO', 'PENDING', 'PASS', 'FAIL']),
                'xrd_state': st.column_config.SelectboxColumn('XRD judgment', options=['PENDING', 'PASS', 'FAIL']),
                'stem_state': st.column_config.SelectboxColumn('Optional STEM', options=['NOT_TESTED', 'CONFIRMED', 'NOT_CONFIRMED', 'INCONCLUSIVE'])})
        edited_details = details
        if details:
            st.write('**Precursor concentrations, ICP and linked reagents**')
            edited_details = st.data_editor(pd.DataFrame(details), hide_index=True, width='stretch',
                disabled=['sample_id', 'element', 'unit'], key=f'record_elements_{origin}_{sig}',
                column_config={'precursor_mM': st.column_config.NumberColumn('Precursor salt / mM', min_value=1e-12),
                    'precursor_mmol': st.column_config.NumberColumn('Metal amount / mmol', min_value=1e-12, format='%.6f'),
                    'value': st.column_config.TextColumn('ICP W / wt%'),
                    'flag': st.column_config.SelectboxColumn('ICP flag', options=['', 'MISSING', 'MEASURED', 'ND', 'BLQ']),
                    **chemistry_columns()})
        submitted = st.form_submit_button('Save record table changes', type='primary')
    if submitted:
        try:
            cells = edited_details.to_dict('records') if hasattr(edited_details, 'to_dict') else edited_details
            changes = apply_record_tables(samples, clean_cells(conditions.to_dict('records')), clean_cells(cells), baseline)
            if not changes:
                st.info('No record values changed.')
            else:
                result = save_manual_revisions(store, changes, methods, 'records',
                    {s['sample_id']: fingerprint(s) for s in samples})
                saved(store, result, 'records')
        except (ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))
    if detailed:
        with st.expander('Linked chloride catalogue references'):
            st.dataframe(pd.DataFrame([{'sample_id': s['sample_id'], **row} for s in detailed for row in catalogue_links(s)]),
                hide_index=True, width='stretch', column_config={'Sigma_URL': st.column_config.LinkColumn('Sigma catalogue')})
