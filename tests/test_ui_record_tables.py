"""Entry batch automation and editable experiment tables in both namespaces."""
import unittest
from unittest.mock import patch

import test_ui as ui_fixture


class RecordTableUITests(unittest.TestCase):
    setUp = ui_fixture.UITests.setUp
    tearDown = ui_fixture.UITests.tearDown

    def test_external_batch_grouping_refreshes_automatic_id_but_preserves_manual_override(self):
        from sacfactory.storage import Store
        self.app.radio(key='page').set_value('Samples & data').run()
        store = Store(self.tmp.name, 'REAL')
        record = store.get('sample', 'REAL_ICP_001')
        record['batch_id'] = 'BATCH_0003'
        store.put_many('sample', [record], 'sample_id', overwrite=True)
        self.app.run()
        self.assertEqual(self.app.text_input(key='entry_batch_id_REAL').value, 'BATCH_0004')
        self.app.text_input(key='entry_batch_id_REAL').set_value('CUSTOM_GROUP').run()
        record['batch_id'] = 'BATCH_0008'
        store.put_many('sample', [record], 'sample_id', overwrite=True)
        self.app.run()
        self.assertEqual(self.app.text_input(key='entry_batch_id_REAL').value, 'CUSTOM_GROUP')

    def test_save_range_and_generate_uses_new_cutoff_and_ten_pulse_setting(self):
        from sacfactory.storage import Store
        from test_transfer import context
        store = Store(self.tmp.name, 'REAL')
        ctx = context('Fe')
        ctx['concentrations_mM'] = {'Fe': 1.}
        space = {'concentration_bounds_mM': {'Fe': {'min_mM': 5., 'max_mM': 5.}},
            'pulses': [1], 'pulse_limit_min_C': 1000., 'pulse_limit_max_C': 1000.,
            'max_total_mM': 5., 'verified': True, 'baseline_repeat': False,
            'range_source': 'Synthetic test bounds', 'model_engine': 'adaptive', 'learning_iterations': 2}
        store.put_many('meta', [{'id': 'plan_context', 'origin': 'REAL', 'context': ctx},
            {'id': 'space', 'origin': 'REAL', 'space': space}], 'id', overwrite=True)
        self.app.radio(key='page').set_value('Next batch').run()
        next(w for w in self.app.number_input if w.label == 'Cutoff lower bound / °C').set_value(800.)
        next(w for w in self.app.number_input if w.label == 'Cutoff upper bound / °C').set_value(1200.)
        next(w for w in self.app.text_input if w.label == 'Available pulse counts, comma-separated').set_value('10')
        next(w for w in self.app.button if w.label == 'Save range and generate batch').click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        rows = self.app.session_state['preview_REAL']
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row['recipe']['pulse_count'] == 10 for row in rows))
        self.assertTrue(all(800. <= row['recipe']['pulse_limit_C'] <= 1200. for row in rows))
        self.assertEqual(len({row['recipe']['pulse_limit_C'] for row in rows}), 6)
        self.assertTrue(all(row['recipe']['pulse_limit_C'] % 10 == 0 for row in rows))
        self.assertTrue(all(row['recipe']['concentrations_mM']['Fe'] == 5. for row in rows))

    def test_each_saved_entry_group_has_a_new_automatic_batch(self):
        from sacfactory.storage import Store
        self.app.radio(key='page').set_value('Samples & data').run()
        self.app.number_input(key='batch_n_REAL').set_value(2).run()
        for batch in ['BATCH_0001', 'BATCH_0002']:
            self.assertEqual(self.app.text_input(key='entry_batch_id_REAL').value, batch)
            next(w for w in self.app.button if w.label == 'Preview filled batch').click().run()
            next(w for w in self.app.button if w.label == 'Save filled batch').click().run()
            self.assertEqual(len(self.app.exception), 0)
            records = [s for s in Store(self.tmp.name, 'REAL').all('sample') if s['batch_id'] == batch]
            self.assertEqual(len(records), 2)
        self.assertEqual(self.app.text_input(key='entry_batch_id_REAL').value, 'BATCH_0003')

    def test_record_table_edits_persist_and_show_linked_chloride_reference(self):
        from sacfactory.storage import Store
        self.app.radio(key='page').set_value('Samples & data').run()
        def edit(frame, *args, **kwargs):
            frame = frame.copy()
            key = str(kwargs.get('key', ''))
            if key.startswith('record_conditions_'):
                frame.loc[0, 'pulse_count'] = 3
                frame.loc[0, 'pulse_limit_C'] = 1200.
                frame.loc[0, 'notes'] = 'Edited in record table'
                frame.loc[0, 'icp_state'] = 'PASS'
                frame.loc[0, 'xrd_state'] = 'PASS'
            if key.startswith('record_elements_'):
                frame.loc[frame.element == 'Fe', 'precursor_mM'] = 50.
                frame.loc[frame.element == 'Fe', 'value'] = '2.5'
            return frame
        with patch('streamlit.data_editor', side_effect=edit):
            next(w for w in self.app.button if w.label == 'Save record table changes').click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        sample = Store(self.tmp.name, 'REAL').get('sample', 'REAL_ICP_001')
        self.assertEqual(sample['pulse_count'], 3)
        self.assertEqual(sample['concentrations_mM']['Fe'], 50.)
        self.assertEqual(next(r['value'] for r in sample['icp'] if r['element'] == 'Fe'), '2.5')
        self.assertEqual(sample['precursors']['Fe']['compound'], 'FeCl2')
        self.assertTrue(any('Sigma_URL' in w.value.columns for w in self.app.dataframe))
        self.app.radio(key='mode').set_value('Simulated demo · DEMO').run()
        self.assertTrue(any(w.label == 'Save record table changes' for w in self.app.button))
        self.assertEqual(len(self.app.exception), 0)
