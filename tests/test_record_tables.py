"""Linked chloride defaults, entry batches and freely editable audited records."""
import copy
import tempfile
import unittest

from sacfactory.storage import Store
from sacfactory.domain import validate_sample, screen
from sacfactory.manual import element_rows, save_manual_revisions, has_observations
from sacfactory.planning import fingerprint
from sacfactory.reagent_links import link_precursors, catalogue_links
from sacfactory.ui_record_tables import condition_rows, apply_record_tables
from test_science import sample


class RecordTableTests(unittest.TestCase):
    def test_integer_cutoff_grid_respects_bounds_and_does_not_round_old_records(self):
        from sacfactory.candidate_grid import cutoff_candidates, on_temperature_grid
        step = {'temperature_step_C': 10}
        values = cutoff_candidates(803., 1197., step, 6, [955., 2000.])
        self.assertGreaterEqual(len(values), 6)
        self.assertTrue(all(isinstance(v, int) and v % 10 == 0 and 803 <= v <= 1197 for v in values))
        self.assertFalse(on_temperature_grid(955., step))
        self.assertEqual(cutoff_candidates(1000., 1000., step), [1000])
        with self.assertRaises(ValueError):
            cutoff_candidates(951., 959., step)
        with self.assertRaises(ValueError):
            cutoff_candidates(800., 1200., {'temperature_step_C': 2.5})

    def test_shared_process_defaults_preserve_explicit_systems_and_explain_fixed_ranges(self):
        from sacfactory.experiment_defaults import link_shared_process, fixed_search_space
        raw = {**sample(), 'precursor_family': 'UNKNOWN', 'support_batch': '', 'protocol_id': 'USER_PROTOCOL'}
        linked = link_shared_process(raw)
        self.assertEqual(linked['protocol_id'], 'USER_PROTOCOL')
        self.assertEqual(linked['support_batch'], 'PAN_DERIVED_CARBON')
        self.assertEqual(raw['precursor_family'], 'UNKNOWN')
        fixed = {'concentration_bounds_mM': {'Fe': {'min_mM': 5., 'max_mM': 5.}},
            'pulses': [1], 'pulse_limit_min_C': 1000., 'pulse_limit_max_C': 1000.}
        self.assertIn('Only one distinct recipe', fixed_search_space(fixed, ['Fe']))
        fixed['pulse_limit_max_C'] = 1200.
        self.assertIsNone(fixed_search_space(fixed, ['Fe']))

    def test_fitting_gate_counts_are_reported_without_imposing_a_batch_size(self):
        from sacfactory.adaptive import fit_adaptive
        from test_transfer import source, context
        records = [source(i, passed=i >= 2) for i in range(12)]
        state = fit_adaptive(records, {}, context('Fe'), 'REAL', True)
        self.assertIsNone(state.model)
        self.assertEqual(state.info['fitting_gate']['eligible_samples'], 12)
        self.assertEqual(state.info['fitting_gate']['fail_recipes'], 2)
        self.assertIn('12/12 eligible', state.info['reason'])

    def test_known_outcomes_with_unknown_batches_can_fit_without_independent_batch_claims(self):
        from sacfactory.adaptive import fit_adaptive
        from test_transfer import source, context
        records = [{**source(i, passed=i >= 4), 'batch_id': 'UNKNOWN'} for i in range(16)]
        state = fit_adaptive(records, {}, context('Fe'), 'REAL', True,
            {'learning_iterations': 2, 'batch_effect': False, 'skip_validation': True})
        self.assertIsNotNone(state.model)
        self.assertEqual(state.info['n_labeled'], 16)
        self.assertEqual(state.info['n_batches'], 0)
        self.assertEqual(state.info['unknown_batch_samples'], 16)
        self.assertFalse(state.model.gp.batch_enabled)

    def test_entry_batch_counter_advances_only_after_save_and_preserves_custom_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp, 'REAL')
            self.assertEqual(store.next_batch_id(), 'BATCH_0001')
            self.assertEqual(store.next_batch_id(), 'BATCH_0001')
            first = validate_sample({**sample(), 'sample_id': 'SAC_0001', 'batch_id': 'BATCH_0001'})
            second = validate_sample({**sample(), 'sample_id': 'SAC_0002', 'batch_id': 'BATCH_0001'})
            store.import_bundle([first, second], [])
            self.assertEqual(store.next_batch_id(), 'BATCH_0002')
            store.delete_samples(['SAC_0001', 'SAC_0002'])
            self.assertEqual(Store(tmp, 'REAL').next_batch_id(), 'BATCH_0001')
            self.assertEqual(Store(tmp, 'DEMO').next_batch_id(), 'BATCH_0001')

    def test_deleted_trailing_batch_is_reused_despite_legacy_counter_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp,'REAL')
            records=[validate_sample({**sample(),'sample_id':f'SAC_{i:04d}',
                'batch_id':f'BATCH_{i:04d}'}) for i in range(1,9)]
            store.import_bundle(records,[])
            self.assertEqual(store.next_batch_id(),'BATCH_0009')
            store.delete_samples(['SAC_0008'])
            self.assertEqual(store.get('meta','batch-numbering')['high_water'],7)
            # A stale pre-upgrade counter must not reserve a deleted batch forever.
            store.put_many('meta',[{'id':'batch-numbering','origin':'REAL','high_water':99}],'id',overwrite=True)
            self.assertEqual(Store(tmp,'REAL').next_batch_id(),'BATCH_0008')
            replacement=validate_sample({**sample(),'sample_id':'SAC_0008','batch_id':'BATCH_0008'})
            store.import_bundle([replacement],[])
            self.assertEqual(store.next_batch_id(),'BATCH_0009')

    def test_partial_or_middle_batch_deletion_preserves_other_batch_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp,'REAL')
            rows=[validate_sample({**sample(),'sample_id':f'SAC_{i:04d}','batch_id':batch})
                for i,batch in enumerate(['BATCH_0001','BATCH_0002','BATCH_0003','BATCH_0003','CUSTOM'],1)]
            store.import_bundle(rows,[])
            store.delete_samples(['SAC_0003'],realign=False)
            self.assertEqual(store.next_batch_id(),'BATCH_0004')
            store.delete_samples(['SAC_0002'],realign=False)
            self.assertEqual(store.next_batch_id(),'BATCH_0004')
            self.assertEqual(store.get('sample','SAC_0004')['batch_id'],'BATCH_0003')
            store.delete_samples(['SAC_0004'],realign=False)
            self.assertEqual(Store(tmp,'REAL').next_batch_id(),'BATCH_0002')
            self.assertEqual(store.get('sample','SAC_0005')['batch_id'],'CUSTOM')

    def test_linked_chlorides_preserve_explicit_other_salts_and_states(self):
        record = {**sample(), 'elements': ['Fe', 'Cu', 'Co', 'Og'],
            'concentrations_mM': {e: 50. for e in ['Fe', 'Cu', 'Co', 'Og']},
            'precursors': {'Co': {'compound': 'Co(NO3)2', 'type': 'NITRATE', 'oxidation_state': 2,
                'source': 'User actual reagent', 'coordination': 'UNKNOWN', 'spin': 'UNKNOWN', 'state_source': ''}}}
        linked = link_precursors(record)
        self.assertEqual(linked['precursors']['Fe']['compound'], 'FeCl2')
        self.assertEqual(linked['precursors']['Cu']['compound'], 'CuCl2')
        self.assertEqual(linked['precursors']['Co']['compound'], 'Co(NO3)2')
        self.assertNotIn('Og', linked['precursors'])
        self.assertEqual(linked['precursors']['Fe']['coordination'], 'UNKNOWN')
        self.assertTrue(linked['precursors']['Fe']['source'].startswith('PLANNED_CATALOG_ASSUMPTION:'))
        rows = {r['element']: r for r in catalogue_links(linked)}
        self.assertIn('sigmaaldrich.com', rows['Fe']['Sigma_URL'])
        self.assertIsNone(rows['Co']['Sigma_URL'])
        self.assertEqual(link_precursors(linked), linked)
        self.assertNotIn('Fe', record['precursors'])

    def test_record_tables_save_conditions_assays_and_links_without_changing_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp, 'REAL')
            original = validate_sample({**sample(), 'sample_id': 'EDIT_1', 'record_uid': 'physical-EDIT-1', 'elements': ['Fe'], 'concentrations_mM': {'Fe': 50.}, 'icp': []})
            store.put_many('sample', [original], 'sample_id')
            original = store.get('sample', 'EDIT_1')
            conditions = condition_rows([original])
            conditions[0].update(batch_id='BATCH_0007', pulse_count=3, pulse_limit_C=1200.,
                pan_mass_g=2., solvent_volume_mL=20., voltage_V=32., current_A=80.,
                icp_state='PASS', xrd_state='PASS', notes='Corrected manual entry')
            baseline = [{'sample_id': 'EDIT_1', **r} for r in element_rows(link_precursors(original))]
            cells = copy.deepcopy(baseline)
            cells[0].update(precursor_mM=25., value='2.5')
            changed = apply_record_tables([original], conditions, cells, baseline)
            save_manual_revisions(store, changed, {}, 'records', {'EDIT_1': fingerprint(original)})
            revised = store.get('sample', 'EDIT_1')
            self.assertEqual(revised['record_uid'], original['record_uid'])
            self.assertEqual(revised['concentrations_mM']['Fe'], 25.)
            self.assertEqual(revised['precursor_amounts_mmol']['Fe'], .5)
            self.assertEqual(revised['icp'][0]['value'], '2.5')
            self.assertEqual(revised['precursors']['Fe']['compound'], 'FeCl2')
            self.assertEqual(screen(revised, {})['label'], 1)
            self.assertTrue(any(r['before_json'] for r in store.export()['audit'] if r['kind'] == 'sample'))

    def test_invalid_record_table_save_is_atomic_and_ids_are_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp, 'REAL')
            original = validate_sample({**sample(), 'sample_id': 'EDIT_1'})
            store.put_many('sample', [original], 'sample_id')
            before = store.all('sample')
            rows = condition_rows(before)
            rows[0]['pulse_count'] = -1
            changed = apply_record_tables(before, rows, [], [])
            with self.assertRaises(ValueError):
                save_manual_revisions(store, changed, {}, 'records', {'EDIT_1': fingerprint(before[0])})
            self.assertEqual(store.all('sample'), before)
            rows[0]['sample_id'] = 'RENAMED'
            with self.assertRaises(ValueError):
                apply_record_tables(before, rows, [], [])

    def test_electrical_settings_do_not_turn_unmeasured_plans_into_observations(self):
        pending = {**sample(), 'icp': [], 'icp_review': {'state': 'PENDING'}, 'xrd': {'state': 'PENDING'},
            'temperature_C': None, 'voltage_V': 32., 'current_A': 80., 'electrical_input_kind': 'SETTING'}
        self.assertFalse(has_observations(pending))
        pending.pop('electrical_input_kind')
        self.assertTrue(has_observations(pending))
