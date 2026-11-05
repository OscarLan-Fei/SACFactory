"""Zero-record planning, model switching and human-adjusted public forecasts."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from sacfactory.compute_runtime import execution
from sacfactory.planning import recommend, ModelState, fingerprint
from sacfactory.planning_models import options, selection, load, initial_context, initial_space
from sacfactory.storage import Store

APP = Path(__file__).resolve().parents[1]/'app.py'


class PretrainedPlanningTests(unittest.TestCase):
    def test_both_models_plan_without_records_and_keep_space_and_identity(self):
        with tempfile.TemporaryDirectory() as root, execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            self.assertEqual(selection(root,'REAL'),'Pretrained SAC-Gen-GB')
            self.assertEqual(options(root,'DEMO'),{})
            context, space = initial_context(), initial_space()
            space.update(pulse_limit_min_C=1000.,pulse_limit_max_C=1020.,pulses=[1],sac_gen_candidate_budget=16)
            state = ModelState(None, {'engine':'sac_gen'}, [])
            for name in options(root,'REAL'):
                model = load(root,'REAL',name,[])
                rows, _ = recommend(state,[],context,space,'REAL',6,sac_gen=model)
                self.assertEqual(len(rows),6)
                self.assertEqual(len({fingerprint(row['recipe']) for row in rows}),6)
                for row in rows:
                    self.assertEqual(row['model_id'],model.model_id)
                    self.assertGreater(row['predicted_total_wt_percent'],0)
                    self.assertTrue(0 <= row['p_xrd_pass'] <= 1)
                    self.assertIsNone(row['p_screen_pass'])
                    self.assertTrue(row['extrapolation'])
                    self.assertEqual(row['evaluation_capture']['training_membership_status'],'NOT_RELEASED')
                    self.assertTrue(1000 <= row['recipe']['pulse_limit_C'] <= 1020)
                duplicate_free, _ = recommend(state,[],context,space,'REAL',6,rows,sac_gen=model)
                self.assertFalse({fingerprint(r['recipe']) for r in rows} & {fingerprint(r['recipe']) for r in duplicate_free})
            self.assertEqual(Store(root,'REAL').all('sample'),[])

    def test_fresh_install_ui_generates_switches_and_enqueues_without_fitting(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,{'SACFACTORY_DATA_DIR':root}), \
             execution({'device':'cpu','threads':1,'cpu_jobs':1}), \
             patch('sacfactory.planning.fit_model', side_effect=AssertionError('No local fit is needed')):
            app = AppTest.from_file(str(APP),default_timeout=90)
            app.session_state['page']='Next batch'
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(app.checkbox(key='next_sac_gen_REAL').value)
            self.assertEqual(app.selectbox(key='next_sac_variant_REAL').value,'Pretrained SAC-Gen-GB')
            next(b for b in app.button if b.label=='Generate batch preview').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            rows=app.session_state['preview_REAL']
            self.assertEqual(len(rows),6)
            first=rows[0]['model_id']
            app.run()
            self.assertEqual(len(app.session_state['preview_REAL']),6)
            app.selectbox(key='next_sac_variant_REAL').set_value('Pretrained SAC-Gen-Diffusion').run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state.get('preview_REAL',[]))
            next(b for b in app.button if b.label=='Generate batch preview').click().run()
            self.assertFalse(app.error)
            self.assertNotEqual(app.session_state['preview_REAL'][0]['model_id'],first)
            app.button(key='pending_plan').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            store=Store(root,'REAL')
            pending=store.all('sample')
            self.assertEqual(len(pending),6)
            self.assertTrue(all(s['xrd']['state']=='PENDING' for s in pending))
            from sacfactory.pending_prediction import refresh_pending_prediction
            sample=pending[0]
            forecast=refresh_pending_prediction(store,sample['sample_id'],fingerprint(sample))
            self.assertEqual(forecast['sac_gen_prediction']['model_name'],'Pretrained SAC-Gen-Diffusion')
            self.assertGreater(forecast['sac_gen_prediction']['predicted_total_wt_percent'],0)


if __name__ == '__main__':
    unittest.main()
