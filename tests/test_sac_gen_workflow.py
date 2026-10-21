"""Deployable SAC-Gen drives candidate acquisition and human-adjusted forecasts."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from test_pulseicp_bridge import specimens
from test_sac_gen import budget
from sacfactory.storage import Store
from sacfactory.sac_gen import train_deployment,latest
from sacfactory.compute_runtime import execution
from sacfactory.planning import recommend,recommendation_table,fingerprint
from sacfactory.pending_prediction import refresh_pending_prediction


class SACGenWorkflowTests(unittest.TestCase):
    def test_next_batch_and_pending_use_named_xrd_probability_not_joint_screening(self):
        with tempfile.TemporaryDirectory() as folder,execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            store=Store(folder,'REAL');store.import_bundle(specimens(40),[])
            source=store.all('sample')
            pointer=train_deployment(source,'REAL',Path(folder),'Gaussian residual',budget())
            model=latest(folder,'REAL',source)
            self.assertEqual(pointer['model_id'],model.model_id)
            context=copy.deepcopy(source[0]);context['pulse_limit_C']=1000.
            space={'verified':True,'concentration_bounds_mM':{'Fe':{'min_mM':70.,'max_mM':90.}},
                'pulses':[1],'max_total_mM':100.,'pulse_limit_min_C':1000.,'pulse_limit_max_C':1000.,
                'sac_gen_strategy':'High-loading XRD failure mapping'}
            state=SimpleNamespace(model=None,training=[],info={})
            rows,note=recommend(state,source,context,space,'REAL',3,sac_gen=model)
            self.assertEqual(len(rows),3)
            self.assertTrue(all(r['p_screen_pass'] is None and 0<=r['p_xrd_pass']<=1 for r in rows))
            self.assertTrue(all(r['policy']=='SAC_GEN_XRD_ACTIVE_LEARNING' for r in rows))
            self.assertIn('p_xrd_fail',recommendation_table(rows)[0])
            pending=copy.deepcopy(source[0]);pending.update(sample_id='SAC_1000',batch_id='BATCH_NEW',
                result_reported_at=None,icp=[],icp_review={'state':'PENDING'},xrd={'state':'PENDING'},sample_condition='NOT_RECORDED')
            store.import_bundle([pending],[])
            saved=store.get('sample','SAC_1000')
            result=refresh_pending_prediction(store,'SAC_1000',fingerprint(saved))
            self.assertEqual(result['model_name'],'SAC-Gen')
            self.assertIsNone(result['p_screen_pass'])
            self.assertTrue(0<=result['p_xrd_pass']<=1)
            self.assertTrue(result['icp_prediction']['values'])
            self.assertEqual(source,[s for s in store.all('sample') if s['sample_id']!='SAC_1000'])


if __name__=='__main__':unittest.main()
