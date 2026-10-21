"""REAL/DEMO quantity controls and migrated ICP workflows in isolated stores."""
import copy
import unittest

import test_ui as fixture


class PulseICPUITests(unittest.TestCase):
    setUp = fixture.UITests.setUp
    tearDown = fixture.UITests.tearDown

    def test_new_entries_default_to_mmol_and_records_expose_both_units(self):
        for label, origin in [('Real experiments · REAL','REAL'),('Simulated demo · DEMO','DEMO')]:
            self.app.radio(key='mode').set_value(label).run()
            self.app.radio(key='page').set_value('Samples & data').run()
            self.assertEqual(self.app.radio(key=f'entry_preparation_{origin}_unit').value,'mmol')
            tables=[w.value for w in self.app.dataframe if {'precursor_mM','precursor_mmol'}.issubset(w.value.columns)]
            self.assertTrue(tables)
            self.assertEqual(len(self.app.exception),0)

    def test_migrated_icp_models_validate_and_render_without_altering_records(self):
        from sacfactory.storage import Store
        from test_pulseicp_bridge import specimens
        store=Store(self.tmp.name,'REAL');store.clear_namespace()
        store.put_many('sample',specimens(),'sample_id')
        before=copy.deepcopy(store.all('sample'))
        self.app.radio(key='page').set_value('Data analysis').run()
        self.app.radio(key='loading_engine_REAL').set_value('SAC-Gen loading models').run()
        self.assertEqual(len(self.app.exception),0)
        self.app.selectbox(key='picp_protocol_REAL').set_value('Batch holdout').run()
        self.app.number_input(key='picp_members_REAL').set_value(1)
        self.app.number_input(key='picp_gepoch_REAL').set_value(2)
        self.app.number_input(key='picp_depoch_REAL').set_value(2)
        self.app.button(key='picp_run_REAL').click().run(timeout=90)
        self.assertEqual(len(self.app.exception),0)
        self.assertEqual(len(self.app.error),0)
        result=self.app.session_state['picp_result_REAL'][1]
        self.assertFalse(result[1].empty)
        self.assertEqual(len(result[2]),3)
        self.assertEqual(store.all('sample'),before)

    def test_rotating_one_or_two_batch_groups_pool_all_samples_under_one_algorithm(self):
        from sacfactory.storage import Store
        from test_pulseicp_bridge import specimens
        from sacfactory import pulseicp_bridge as bridge
        store=Store(self.tmp.name,'REAL');store.clear_namespace()
        rows=specimens();rows[0]['icp']=[]
        store.put_many('sample',rows,'sample_id')
        before=copy.deepcopy(store.all('sample'))
        self.app.session_state['picp_protocol_REAL']='Batch holdout'
        self.app.radio(key='page').set_value('Data analysis').run()
        self.app.radio(key='loading_engine_REAL').set_value('SAC-Gen loading models').run()
        self.assertEqual(self.app.selectbox(key='picp_protocol_REAL').value,bridge.ROTATING_CV)
        self.assertEqual(self.app.number_input(key='picp_cv_batches_REAL').value,2)
        self.app.number_input(key='picp_members_REAL').set_value(1)
        self.app.number_input(key='picp_gepoch_REAL').set_value(2)
        self.app.number_input(key='picp_depoch_REAL').set_value(2)
        self.app.button(key='picp_run_REAL').click().run(timeout=90)
        self.assertEqual(len(self.app.exception),0)
        self.assertEqual(len(self.app.error),0)
        result=self.app.session_state['picp_result_REAL'][1]
        self.assertEqual(len(result[1]),59)
        self.assertEqual(len(result[2]),1)
        self.assertEqual(len(result[3]['partitions']),3)
        self.assertEqual(next(w.value for w in self.app.metric if w.label=='Out-of-fold predicted specimens'),'59')
        del self.app.session_state['picp_result_REAL']
        self.app.run()
        self.assertEqual(len(self.app.session_state['picp_result_REAL'][1][1]),59)
        self.app.number_input(key='picp_cv_batches_REAL').set_value(1).run()
        self.app.button(key='picp_run_REAL').click().run(timeout=90)
        self.assertEqual(len(self.app.exception),0)
        self.assertEqual(len(self.app.session_state['picp_result_REAL'][1][3]['partitions']),6)
        self.assertEqual(store.all('sample'),before)


if __name__=='__main__':
    unittest.main()
