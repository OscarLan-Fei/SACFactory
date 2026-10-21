"""REAL/DEMO physical comparison controls and editable Pending end-to-end."""
import copy
import unittest
from unittest.mock import patch

import test_ui as ui_fixture


class PhysicsUITests(unittest.TestCase):
    setUp = ui_fixture.UITests.setUp
    tearDown = ui_fixture.UITests.tearDown

    def test_material_inputs_are_explicit_in_both_spaces_and_demo_comparison_runs(self):
        from sacfactory.storage import Store
        for mode, origin in [('Real experiments · REAL', 'REAL'), ('Simulated demo · DEMO', 'DEMO')]:
            self.app.radio(key='mode').set_value(mode).run()
            self.app.radio(key='page').set_value('Data analysis').run()
            self.app.radio(key=f'loading_data_mode_{origin}').set_value('Single-element model').run()
            self.app.checkbox(key=f'loading_physics_{origin}_enable').check().run()
            self.assertEqual(len(self.app.exception), 0)
            self.assertIsNone(self.app.number_input(key=f'loading_physics_{origin}_pan').value)
            self.assertFalse(self.app.checkbox(key=f'loading_physics_{origin}_confirmed').value)
            self.assertTrue(self.app.button(key=f'loading_run_{origin}').disabled)
        store = Store(self.tmp.name, 'DEMO')
        before = copy.deepcopy(store.all('sample'))
        self.app.number_input(key='loading_physics_DEMO_pan').set_value(2.)
        self.app.number_input(key='loading_physics_DEMO_volume').set_value(20.)
        self.app.number_input(key='loading_physics_DEMO_carbon').set_value(6.)
        self.app.text_input(key='loading_physics_DEMO_source').set_value('Synthetic 2 g PAN / 20 mL DMF demonstration')
        self.app.checkbox(key='loading_physics_DEMO_confirmed').check()
        self.app.selectbox(key='loading_model_DEMO').set_value('Ridge baseline')
        self.app.selectbox(key='loading_profile_DEMO').set_value('chemistry_thermal').run()
        self.app.button(key='loading_run_DEMO').click().run()
        self.assertEqual(len(self.app.exception), 0)
        predictions, _, info = self.app.session_state['loading_result_DEMO'][1]
        self.assertFalse(predictions.empty)
        self.assertEqual(len(info['paired_comparison']), 3)
        self.assertEqual(store.all('sample'), before)

    def test_pending_tables_save_conditions_then_refresh_probability_and_keep_original(self):
        from test_chemistry import campaign
        from test_transfer import space
        from sacfactory.storage import Store
        from sacfactory.planning import fit_model, recommend, fingerprint
        from sacfactory.domain import RULE_VERSION
        store = Store(self.tmp.name, 'REAL')
        records = campaign()[:36]
        store.put_many('sample', records, 'sample_id')
        ctx = records[0]
        operating = space()
        operating['concentration_bounds_mM'] = {'Fe': {'min_mM': .01, 'max_mM': .15}}
        operating.update(model_engine='adaptive', descriptor_profile='chemistry_thermal', learning_iterations=2)
        store.put_many('meta', [{'id': 'plan_context', 'origin': 'REAL', 'context': ctx},
                               {'id': 'space', 'origin': 'REAL', 'space': operating}], 'id', overwrite=True)
        samples = store.all('sample')
        methods = {m['method_id']: m for m in store.all('method')}
        state = fit_model(samples, methods, ctx, 'REAL', True, 'adaptive',
                          {'descriptor_profile': 'chemistry_thermal', 'learning_iterations': 2, 'skip_validation': True})
        rows, _ = recommend(state, samples, ctx, operating, 'REAL', 1)
        rows[0]['snapshot'] = fingerprint({'samples': samples, 'methods': methods, 'context': ctx,
            'space': operating, 'rule_version': RULE_VERSION})
        ident = store.enqueue_pending(rows, ctx, operating)[0]
        original = copy.deepcopy(store.get('sample', ident)['proposal'])
        self.app.radio(key='page').set_value('Samples & data').run()
        next(w for w in self.app.selectbox if w.label == 'Pending sample to edit').set_value(ident).run()
        def edited(frame, *args, **kwargs):
            frame = frame.copy()
            key = kwargs.get('key', '')
            if key.startswith('pending_conditions_'):
                frame.loc[frame.sample_id == ident, ['pulse_count', 'pulse_limit_C']] = [5, 1200]
            if key.startswith('pending_concentrations_'):
                frame.loc[frame.sample_id == ident, 'concentration_mM'] = .07
            return frame
        with patch('streamlit.data_editor', side_effect=edited):
            next(w for w in self.app.button if w.label == 'Save pending table changes').click().run()
        self.assertEqual(len(self.app.exception), 0)
        sample = store.get('sample', ident)
        self.assertEqual((sample['pulse_count'], sample['pulse_limit_C'], sample['concentrations_mM']['Fe']), (5, 1200., .07))
        self.assertTrue(sample['proposal']['prediction_stale'])
        next(w for w in self.app.button if w.label == 'Predict saved pending parameters').click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        sample = store.get('sample', ident)
        self.assertTrue(0 <= sample['manual_prediction']['p_screen_pass'] <= 1)
        self.assertEqual(sample['manual_prediction']['recipe']['pulse_count'], 5)
        self.assertEqual(sample['proposal']['p_screen_pass'], original['p_screen_pass'])
        queue = store.get('recommendation', sample['recommendation_id'])
        self.assertFalse(queue['prediction_stale'])
        self.assertEqual(queue['current_prediction_basis'], 'HUMAN_ADJUSTED_RECIPE')
        self.assertEqual(sample['icp'], [])
