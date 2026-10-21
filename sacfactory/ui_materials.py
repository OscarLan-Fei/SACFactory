"""Material records separate from operating-range and batch recommendation controls."""
from __future__ import annotations

import copy
import math

import pandas as pd
import streamlit as st

from .domain import cohort_key
from .planning import fingerprint
from .precursors import normalize_precursors
from .precursor_catalog import resolve_planning_precursors
from .ui_chemistry import render_precursor_editor, render_chemical_reference, render_sigma_catalog


LEDGER_COLUMNS = ['item_name', 'category', 'supplier', 'product_code', 'lot_number',
                  'quantity', 'unit', 'storage_location', 'notes']


def planning_precursors(store, context, elements):
    """Use saved reagent defaults for new targets while retaining explicit recipes."""
    library = (store.get('meta', 'reagent_library') or {}).get('precursors', {})
    return resolve_planning_precursors(library, context, elements)


def normalize_ledger(rows):
    """Retain partial inventory entries without assigning chemical/model values."""
    result = []
    for row in rows:
        entry = {}
        for key in LEDGER_COLUMNS:
            value = row.get(key)
            if value is None or pd.isna(value):
                value = None if key == 'quantity' else ''
            if key == 'quantity' and value is not None:
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Inventory quantities must be finite and nonnegative, or blank.')
            entry[key] = value if key == 'quantity' else str(value).strip()
        if any(value not in ('', None) for value in entry.values()):
            result.append(entry)
    return result


def render_materials(store, context, samples):
    origin = store.origin
    st.subheader('Materials & consumables')
    st.caption('Keep reagent identities, fixed process metadata and inventory here. Next batch reads saved planning metadata automatically; historical experiment records are edited on Samples & data.')
    setup, inventory, references = st.tabs(['Precursors & fixed system', 'Inventory', 'Chemical references'])
    with setup:
        st.write('**Active target:** ' + ', '.join(context['elements']))
        st.caption('Change target elements and concentrations on Next batch. Material identity and formal precursor states can affect chemical transfer; inventory quantities and purchasing notes do not enter the predictor.')
        with st.expander('Load an existing experimental system'):
            known = {}
            for sample in samples:
                known.setdefault(cohort_key(sample), sample)
            values = list(known.values())
            selected = st.selectbox('Load an existing cohort', range(len(values)),
                format_func=lambda i: f"{len(values[i]['elements'])} elements | {values[i]['protocol_id']} | {values[i]['sample_id']}",
                key=f'cohort_picker_{origin}', disabled=not values)
            if st.button('Use this cohort', disabled=not values):
                ctx = copy.deepcopy(values[selected])
                st.session_state[f'plan_elements_{origin}'] = list(ctx['elements'])
                store.put_many('meta', [{'id': 'plan_context', 'origin': origin, 'context': ctx}], 'id', overwrite=True)
                st.rerun()
        library = (store.get('meta', 'reagent_library') or {}).get('precursors', {})
        with st.expander('Link declared chlorides to recorded experiments'):
            from .chemical_linking import link_declared_chlorides
            square = st.checkbox('Use square-planar IVSQ as an explicit reference assumption', key=f'link_square_{origin}')
            changes, link_report = link_declared_chlorides(samples, library, origin, square)
            st.caption('Record actual reagents. Catalogue defaults are editable; confirm chemical states before using state-dependent descriptors.')
            st.dataframe(pd.DataFrame(link_report), hide_index=True, width='stretch')
            if st.button('Apply declared chloride links', disabled=not changes, key=f'link_chlorides_{origin}'):
                store.put_many('sample', changes, 'sample_id', overwrite=True)
                st.rerun()
        with st.expander('Saved reagent defaults'):
            if library:
                with st.form(f'reagent_library_{origin}'):
                    edited_library = render_precursor_editor(list(library), library,
                        f'reagent_defaults_{origin}_{fingerprint(library)}')
                    if st.form_submit_button('Save reagent defaults'):
                        try:
                            saved = normalize_precursors(edited_library, list(library))
                            ctx = {**context, 'precursors': normalize_precursors(
                                {**context.get('precursors', {}), **{e: v for e, v in saved.items() if e in context['elements']}}, context['elements'])}
                            store.put_many('meta', [{'id': 'reagent_library', 'origin': origin, 'precursors': saved},
                                {'id': 'plan_context', 'origin': origin, 'context': ctx}], 'id', overwrite=True)
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
            else:
                st.caption('Save a material setup below to register defaults for its elements.')
        draft_key = f"catalog_draft_{origin}_{fingerprint(context['elements'])}"
        with st.expander('Sigma precursor catalogue and planning draft'):
            render_sigma_catalog(context['elements'], f'planning_catalog_{origin}')
            st.caption('Loading stages an editable assumption only. Review compound, hydrate and source before saving; it does not confirm historical reagent usage.')
            if st.button('Load Sigma candidates into planning draft', key=f'catalog_load_{origin}'):
                from .precursor_catalog import planning_draft
                st.session_state[draft_key] = {**planning_draft(context['elements']),
                    **{e: v for e, v in library.items() if e in context['elements']}}
                st.rerun()
        starting = st.session_state.get(draft_key, planning_precursors(store, context, context['elements']))
        with st.form(f'material_setup_{origin}'):
            precursor = st.text_input('Fixed precursor system / version', value=context['precursor_family'])
            support = st.text_input('Fixed support system / batch', value=context['support_batch'])
            protocol = st.text_input('Fixed protocol version', value=context['protocol_id'])
            chemistry = render_precursor_editor(context['elements'], starting,
                f"plan_chemistry_{origin}_{fingerprint([context['elements'], starting])}")
            if st.form_submit_button('Save material setup', type='primary'):
                try:
                    if not all(value.strip() for value in (precursor, support, protocol)):
                        raise ValueError('Provide system/version fields; use UNKNOWN where not yet recorded.')
                    ctx = {**context, 'precursor_family': precursor.strip(), 'support_batch': support.strip(),
                           'protocol_id': protocol.strip(), 'precursors': normalize_precursors(chemistry, context['elements'])}
                    saved_library = {**library, **ctx['precursors']}
                    store.put_many('meta', [{'id': 'plan_context', 'origin': origin, 'context': ctx},
                        {'id': 'reagent_library', 'origin': origin, 'precursors': saved_library}], 'id', overwrite=True)
                    st.session_state.pop(draft_key, None)
                    st.session_state[f'materials_saved_{origin}'] = True
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        if st.session_state.pop(f'materials_saved_{origin}', False):
            st.success('Material setup saved. Next batch will use these planning inputs.')
        if st.button('Open Next batch', key=f'materials_to_plan_{origin}'):
            st.session_state['page_jump'] = 'Next batch'
            st.rerun()
    with inventory:
        st.caption('Optional inventory log. Partial entries are allowed. Add or remove rows, then save. This table is separate from experiment results and model features.')
        record = store.get('meta', 'materials_consumables') or {}
        frame = pd.DataFrame(record.get('rows', []), columns=LEDGER_COLUMNS)
        with st.form(f'materials_inventory_{origin}'):
            edited = st.data_editor(frame, num_rows='dynamic', hide_index=True, width='stretch',
                column_config={'category': st.column_config.SelectboxColumn('Category', options=['Precursor', 'Support', 'Solvent', 'Consumable', 'Other']),
                    'quantity': st.column_config.NumberColumn('Quantity', min_value=0.)},
                key=f'materials_ledger_{origin}_{fingerprint(record)}')
            if st.form_submit_button('Save inventory'):
                try:
                    rows = normalize_ledger(edited.to_dict('records'))
                    store.put_many('meta', [{'id': 'materials_consumables', 'origin': origin, 'rows': rows}], 'id', overwrite=True)
                    st.session_state[f'inventory_saved_{origin}'] = True
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        if st.session_state.pop(f'inventory_saved_{origin}', False):
            st.success('Inventory saved.')
        st.download_button('Download inventory CSV', frame.to_csv(index=False).encode('utf-8-sig'),
            'materials_consumables.csv', 'text/csv', key=f'materials_csv_{origin}')
    with references:
        render_chemical_reference(context, f'materials_reference_{origin}')
