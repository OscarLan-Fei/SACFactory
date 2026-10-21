"""Amounts/default materials in batch entry, registration and manual correction."""
import copy
import unittest
from unittest.mock import patch

import test_ui as ui_fixture


class PreparationUITests(unittest.TestCase):
    setUp = ui_fixture.UITests.setUp
    tearDown = ui_fixture.UITests.tearDown

    def test_top_salt_mmol_entry_synchronizes_detail_and_saves_in_mM(self):
        from sacfactory.storage import Store
        self.app.radio(key='page').set_value('Samples & data').run()
        self.app.radio(key='entry_preparation_REAL_unit').set_value('mM').run()
        self.app.number_input(key='batch_n_REAL').set_value(1).run()
        changed = [False]
        def entered(frame, *args, **kwargs):
            frame = frame.copy()
            if str(kwargs.get('key', '')).startswith('batch_grid_') and not changed[0]:
                frame['concentration_mM'] = '1 mmol'
                changed[0] = True
            return frame
        with patch('streamlit.data_editor', side_effect=entered):
            next(w for w in self.app.button if w.label == 'Preview filled batch').click().run()
        self.assertEqual(len(self.app.exception), 0)
        cached = self.app.session_state['entry_cells_REAL']
        self.assertTrue(all(r['precursor_mM'] == 50. for r in cached['detail'] if r['sample_id'] == 'SAC_0001'))
        self.assertEqual(cached['batch'][0]['concentration_mM'], '50')
        next(w for w in self.app.button if w.label == 'Save filled batch').click().run()
        record = Store(self.tmp.name, 'REAL').get('sample', 'SAC_0001')
        self.assertTrue(all(v == 50. for v in record['concentrations_mM'].values()))
        self.assertTrue(all(v == 1. for v in record['precursor_amounts_mmol'].values()))

    def test_detail_edit_updates_top_recipe_and_volume_updates_equivalent_amount(self):
        self.app.radio(key='page').set_value('Samples & data').run()
        self.app.radio(key='entry_preparation_REAL_unit').set_value('mM').run()
        self.app.number_input(key='batch_n_REAL').set_value(1).run()
        def entered(frame, *args, **kwargs):
            frame = frame.copy()
            if str(kwargs.get('key', '')).startswith('element_grid_'):
                frame['precursor_mM'] = 50.
            return frame
        with patch('streamlit.data_editor', side_effect=entered):
            self.app.run()
        self.assertEqual(self.app.session_state['entry_cells_REAL']['batch'][0]['concentration_mM'], '50')
        self.app.number_input(key='entry_preparation_REAL_volume').set_value(40.).run()
        self.assertEqual(len(self.app.exception), 0)
        cache = self.app.session_state['entry_cells_REAL']
        self.assertEqual(cache['batch'][0]['solvent_volume_mL'], 40.)
        detail = next(w.value for w in self.app.dataframe if 'converted_precursor' in w.value.columns)
        self.assertTrue(all(detail.converted_precursor == 2.))
        self.app.radio(key='entry_preparation_REAL_unit').set_value('mmol').run()
        self.assertEqual(self.app.session_state['entry_cells_REAL']['batch'][0]['concentration_mM'], '2')
        self.assertTrue(all(r['precursor_mM'] == 2. for r in self.app.session_state['entry_cells_REAL']['detail'] if r['sample_id'] == 'SAC_0001'))

    def test_batch_entry_mmol_defaults_and_save_record_canonical_concentration(self):
        from sacfactory.storage import Store
        self.app.radio(key='page').set_value('Samples & data').run()
        self.app.number_input(key='batch_n_REAL').set_value(1).run()
        self.app.radio(key='entry_preparation_REAL_unit').set_value('mmol').run()
        self.assertEqual(self.app.number_input(key='entry_preparation_REAL_pan').value, 2.)
        self.assertEqual(self.app.number_input(key='entry_preparation_REAL_volume').value, 20.)
        rows = self.app.session_state['entry_cells_REAL']['batch']
        self.assertEqual((rows[0]['voltage_V'], rows[0]['current_A']), (32., 80.))
        def entered(frame, *args, **kwargs):
            frame = frame.copy()
            if str(kwargs.get('key', '')).startswith('element_grid_'):
                frame['precursor_mM'] = 1.
            return frame
        with patch('streamlit.data_editor', side_effect=entered):
            next(w for w in self.app.button if w.label == 'Preview filled batch').click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        # Preserve the patched edit in subsequent editor reruns.
        with patch('streamlit.data_editor', side_effect=entered):
            next(w for w in self.app.button if w.label == 'Save filled batch').click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        record = Store(self.tmp.name, 'REAL').get('sample', 'SAC_0001')
        self.assertTrue(all(c == 50. for c in record['concentrations_mM'].values()))
        self.assertEqual((record['pan_mass_g'], record['solvent_volume_mL']), (2., 20.))
        self.assertEqual((record['voltage_V'], record['current_A']), (32., 80.))

    def test_registration_mmol_uses_edited_DMF_and_records_pan(self):
        from sacfactory.storage import Store
        self.app.radio(key='page').set_value('Samples & data').run()
        self.app.radio(key='register_preparation_REAL_unit').set_value('mmol').run()
        self.app.number_input(key='register_preparation_REAL_pan').set_value(4.)
        self.app.number_input(key='register_preparation_REAL_volume').set_value(40.).run()
        def entered(frame, *args, **kwargs):
            frame = frame.copy()
            if str(kwargs.get('key', '')).startswith('register_concentrations_'):
                frame['precursor_mM'] = 1.
            return frame
        with patch('streamlit.data_editor', side_effect=entered):
            next(w for w in self.app.button if w.label == 'Register pending sample').click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        record = Store(self.tmp.name, 'REAL').get('sample', 'SAC_0001')
        self.assertTrue(all(c == 25. for c in record['concentrations_mM'].values()))
        self.assertEqual((record['pan_mass_g'], record['solvent_volume_mL']), (4., 40.))

    def test_two_element_salt_inputs_sync_both_ways_without_changing_icp(self):
        from sacfactory.batching import pairs
        self.app.radio(key='page').set_value('Samples & data').run()
        self.app.radio(key='entry_preparation_REAL_unit').set_value('mM').run()
        self.app.number_input(key='batch_n_REAL').set_value(1).run()
        next(w for w in self.app.button if w.label == 'Clear selection').click().run()
        next(w for w in self.app.button if w.label == 'Fe').click().run()
        next(w for w in self.app.button if w.label == 'Co').click().run()
        changed = [False]
        def top(frame, *args, **kwargs):
            frame = frame.copy()
            if str(kwargs.get('key', '')).startswith('batch_grid_') and not changed[0]:
                frame['concentration_mM'] = 'Fe:1 mmol;Co:25 mM'
                changed[0] = True
            return frame
        with patch('streamlit.data_editor', side_effect=top):
            self.app.run()
        cached = self.app.session_state['entry_cells_REAL']
        rows = {r['element']: r for r in cached['detail'] if r['sample_id'] == 'SAC_0001' and r['element'] in {'Fe', 'Co'}}
        self.assertEqual({e: r['precursor_mM'] for e, r in rows.items()}, {'Fe': 50., 'Co': 25.})
        changed[0] = False
        def bottom(frame, *args, **kwargs):
            frame = frame.copy()
            if str(kwargs.get('key', '')).startswith('element_grid_') and not changed[0]:
                frame.loc[frame.element == 'Co', 'precursor_mM'] = 12.5
                changed[0] = True
            return frame
        with patch('streamlit.data_editor', side_effect=bottom):
            self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        summary = self.app.session_state['entry_cells_REAL']['batch'][0]['concentration_mM']
        self.assertEqual(pairs(summary, 'test'), {'Fe': '50', 'Co': '12.5'})
        rows = [r for r in self.app.session_state['entry_cells_REAL']['detail'] if r['sample_id'] == 'SAC_0001']
        self.assertTrue(all(not r.get('value') for r in rows))
