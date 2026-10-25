"""Mixing descriptor analysis and its use in actual model forecasts."""
import copy
import unittest

import test_ui as ui_fixture


class MixingUITests(unittest.TestCase):
    setUp = ui_fixture.UITests.setUp
    tearDown = ui_fixture.UITests.tearDown

    def test_real_demo_analysis_has_periodic_mixing_filter_and_source_exports(self):
        from sacfactory.storage import Store
        for label, origin in [('Real experiments · REAL', 'REAL'), ('Simulated demo · DEMO', 'DEMO')]:
            self.app.radio(key='mode').set_value(label).run()
            store = Store(self.tmp.name, origin)
            before = copy.deepcopy(store.all('sample'))
            self.app.radio(key='page').set_value('Data analysis').run()
            self.assertEqual(len(self.app.exception), 0)
            self.assertIn('Mixing thermodynamics', [w.label for w in self.app.tabs])
            widget = self.app.multiselect(key=f'mixing_elements_{origin}')
            self.assertEqual(len(widget.options), 118)
            self.assertIsNotNone(self.app.button(key=f'mixing_elements_{origin}_3_Fe'))
            widget.set_value(['Fe', 'Co', 'Og']).run()
            self.assertEqual(len(self.app.exception), 0)
            self.assertTrue(any(w.label == 'Download mixing descriptor recipes CSV' for w in self.app.get('download_button')))
            self.assertTrue(any(w.label == 'Download selected binary mixing pairs' for w in self.app.get('download_button')))
            self.assertEqual(store.all('sample'), before)

    def test_next_batch_mixing_profile_outputs_enter_pending_and_refreshed_forecast(self):
        from test_chemistry import campaign
        from sacfactory.storage import Store
        store = Store(self.tmp.name, 'REAL')
        records = campaign()
        store.put_many('sample', records, 'sample_id')
        context = records[-1]
        operating = {'model_engine': 'adaptive', 'descriptor_profile': 'chemistry_mixing', 'learning_iterations': 2,
            'concentration_bounds_mM': {e: {'min_mM': .01, 'max_mM': .15} for e in context['elements']},
            'pulses': [1, 2], 'pulse_limit_min_C': 900, 'pulse_limit_max_C': 1100, 'max_total_mM': .3,
            'verified': True, 'baseline_repeat': False, 'repeat_strategy': 'none', 'candidate_budget': 256,
            'range_source': 'Synthetic mixing UI fixture', 'element_transfer': True}
        store.put_many('meta', [{'id': 'plan_context', 'origin': 'REAL', 'context': context},
            {'id': 'space', 'origin': 'REAL', 'space': operating}], 'id', overwrite=True)
        self.app.radio(key='page').set_value('Next batch').run()
        self.assertEqual(len(self.app.exception), 0)
        next(w for w in self.app.number_input if w.label == 'Maximum experiments in the next batch').set_value(2).run()
        next(w for w in self.app.button if w.label == 'Generate batch preview').click().run()
        self.assertEqual(len(self.app.exception), 0)
        from sacfactory.planning import recommendation_table
        table = recommendation_table(self.app.session_state['preview_REAL'])
        self.assertTrue(all(row['mixing_descriptors_used_by_model'] for row in table))
        self.assertTrue(any('mixing_descriptors_used_by_model' in w.value.columns for w in self.app.dataframe))
        next(w for w in self.app.button if w.label == 'Add batch to pending').click().run()
        self.assertEqual(len(self.app.exception), 0)
        pending = next(s for s in store.all('sample') if s.get('recommendation_id'))
        self.app.radio(key='page').set_value('Samples & data').run()
        next(w for w in self.app.selectbox if w.label == 'Pending sample to edit').set_value(pending['sample_id']).run()
        next(w for w in self.app.button if w.label == 'Predict saved pending parameters').click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        forecast = store.get('sample', pending['sample_id'])['manual_prediction']
        self.assertTrue(forecast['mixing']['mixing_descriptors_used_by_model'])
        self.assertIsNotNone(forecast['mixing']['s_mix_J_mol_K'])
