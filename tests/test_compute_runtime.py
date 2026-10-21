"""CPU/CUDA execution, reference equivalence, chunk isolation and fit reuse."""
import copy
import tempfile
import unittest
import numpy as np
import torch

from sacfactory import pulseicp_bridge as bridge
from sacfactory.compute_runtime import execution, resolve, batched_predict
from sacfactory._pulseicp.ensemble import ResidualEnsemble as Frozen
from sacfactory.accelerated_ensemble import ResidualEnsemble
from test_pulseicp_bridge import specimens


def budget(device='cpu'):
    return {**bridge.configuration()['model'],'device':device,'threads':1,
            'members':1,'epochs_baseline':1,'epochs_diffusion':1,'draws_per_member':4,
            'diffusion_steps':10,'token_width':16,'attention_heads':2,'attention_layers':1}


class ComputeTests(unittest.TestCase):
    def test_cpu_adapter_matches_frozen_network_and_draws(self):
        rows=specimens(30);settings=budget();data=bridge.prepare(rows,'REAL',settings)
        indices=np.arange(len(data.records));compositions=[bridge.composition(s) for s in rows[:3]]
        for family in bridge.FAMILIES:
            old=Frozen(family,settings,bridge.configuration()['physics']).fit(data,indices)
            new=ResidualEnsemble(family,settings,bridge.configuration()['physics']).fit(data,indices)
            np.testing.assert_allclose(old.predict_compositions(compositions)[1],new.predict_compositions(compositions)[1],rtol=1e-6)
        with tempfile.TemporaryDirectory() as folder:
            new.save(folder)
            with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
                loaded=ResidualEnsemble.load(folder)
            self.assertEqual(new.model_id,loaded.model_id)
            np.testing.assert_allclose(new.predict_compositions(compositions)[1],loaded.predict_compositions(compositions)[1])

    def test_chunk_isolation_and_request_local_exact_fit_cache(self):
        calls=[];errors=[]
        def predict(model,rows,origin):
            calls.append(len(rows))
            if any(s['id']==3 for s in rows):raise ValueError('Invalid fixture')
            return rows
        values=batched_predict(None,[{'id':i} for i in range(10)],'REAL',predict,
                               lambda s,e:errors.append(s['id']),chunk_size=5)
        self.assertEqual([s['id'] for s in values],[0,1,2,4,5,6,7,8,9])
        self.assertEqual(errors,[3]);self.assertIn(5,calls)
        rows=specimens(30);settings=budget()
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            first,_=bridge.fit(rows,'REAL',settings=settings)
            same,_=bridge.fit(copy.deepcopy(rows),'REAL',settings=settings)
            self.assertIs(first,same)
            changed=copy.deepcopy(rows);changed[0]['icp'][0]['value']='0.5'
            different,_=bridge.fit(changed,'REAL',settings=settings)
            self.assertIsNot(first,different)
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            other,_=bridge.fit(rows,'REAL',settings=settings)
            self.assertIsNot(first,other)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_four_families_train_and_sample_on_cuda(self):
        rows=specimens(30)
        for family in [*bridge.FAMILIES,'Transformer residual','Diffusion Transformer']:
            model,_=bridge.fit(rows,'REAL',family,budget('cuda'))
            self.assertEqual(next(model.members[0].parameters()).device.type,'cuda')
            frame=bridge.predict(model,rows[:4],'REAL')
            self.assertEqual(len(frame),4)
            self.assertTrue(np.isfinite(frame.predicted_mean_w_wt_percent).all())
            self.assertTrue(frame.predicted_mean_w_wt_percent.between(0,100).all())

    def test_invalid_device_and_cpu_override(self):
        with self.assertRaises(ValueError):resolve({'device':'unknown'})
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            self.assertEqual(resolve({'device':'cuda'})['device'],'cpu')


if __name__=='__main__':unittest.main()
