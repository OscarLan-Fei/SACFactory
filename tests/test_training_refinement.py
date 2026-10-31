"""Training-only stopping, checkpoint compatibility and conditional head calibration."""
import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
from test_pulseicp_bridge import specimens
from test_sac_gen import budget
from sacfactory import pulseicp_bridge as bridge
from sacfactory.compute_runtime import execution
from sacfactory.sac_gen import SACGen
from sacfactory.xrd_refinement import fit_head


class RefinementTests(unittest.TestCase):
    def test_validated_loading_refits_full_pool_and_roundtrips(self):
        rows=specimens(40)
        settings={**budget(),'training_variant':'validated-token-v1','candidate_learning_rates':[.001],
            'tuning_min_epochs':1,'tuning_max_epochs':2,'tuning_interval':1,'tuning_patience':1,'tuning_draws':2}
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}),tempfile.TemporaryDirectory() as folder:
            model,audit=bridge.fit(rows,'REAL','Diffusion Transformer',settings)
            detail=model.tuning
            self.assertFalse(set(detail['selection_train_ids'])&set(detail['selection_validation_ids']))
            self.assertEqual(detail['selection_train_ids'],detail['scaler_training_ids'])
            self.assertEqual(set(model.training_ids),set(s['sample_id'] for s in rows))
            path=Path(folder)/'model';model.save(path)
            from sacfactory.tuned_token_residual import TunedTokenResidualEnsemble
            loaded=TunedTokenResidualEnsemble.load(path)
            self.assertEqual(loaded.model_id,model.model_id)
            np.testing.assert_allclose(bridge.predict(model,rows[:2],'REAL').predicted_mean_w_wt_percent,
                bridge.predict(loaded,rows[:2],'REAL').predicted_mean_w_wt_percent)

    def test_refined_sac_gen_roundtrip_and_result_mask(self):
        rows=specimens(60)
        settings={**budget(),'training_variant':'validated-token-v1','refine_xrd':True,'crossfit_grouping':'recipe',
            'candidate_learning_rates':[.001],'tuning_min_epochs':1,'tuning_max_epochs':1,'tuning_interval':1,'tuning_draws':2}
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}),tempfile.TemporaryDirectory() as folder:
            model=SACGen('Diffusion Transformer',settings).fit(rows[:45],'REAL')
            path=Path(folder)/'model';model.save(path);loaded=SACGen.load(path,rows)
            self.assertEqual(model.model_id,loaded.model_id)
            original=loaded.predict(rows[45:48]);changed=copy.deepcopy(rows[45:48])
            for row in changed:row.update(icp=[],xrd={'state':'PENDING'})
            masked=loaded.predict(changed)
            np.testing.assert_allclose([r['p_xrd_pass'] for r in original],[r['p_xrd_pass'] for r in masked])
            self.assertTrue(all(0<=r['xrd_pass_threshold']<=1 for r in original))
            self.assertTrue(all(r['predicted_xrd_state'] in {'PASS','FAIL'} for r in original))

    def test_head_folds_preserve_recipe_groups_and_monotonic_calibration(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(60,4));y=np.tile([0,1],30)
        groups=np.array([str(i//2) for i in range(60)])
        head,info=fit_head(x,y,groups,42)
        for fold in info['head_selection']['nested_head_folds']:
            self.assertFalse(set(groups[fold['train_indices']])&set(groups[fold['held_indices']]))
        self.assertTrue(.05<=head['pass_threshold']<=.95)
        if 'calibration' in head:self.assertGreater(head['calibration']['coefficient'],0)


if __name__=='__main__':unittest.main()
