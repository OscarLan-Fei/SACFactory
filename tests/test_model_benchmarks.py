"""Matched model folds and acquisition replay without test-outcome access."""
import copy
import unittest
import numpy as np
from test_pulseicp_bridge import specimens
from test_sac_gen import budget
from sacfactory.model_benchmarks import compare,MODELS,fitted_baseline
from sacfactory.acquisition_benchmarks import replay,POLICIES
from sacfactory.weighted_validation import collect
from sacfactory.compute_runtime import execution


class BenchmarkTests(unittest.TestCase):
    def test_all_models_share_outer_masks_and_common_rows(self):
        source=specimens(60);before=copy.deepcopy(source)
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            points,table,info=compare(source,{},'REAL',['Fe'],batches_per_fold=2,
                sac_family='Gaussian residual',sac_settings=budget())
        self.assertEqual(set(table.model),set(MODELS))
        self.assertTrue(table.ICP_n.eq(60).all(),table.to_string())
        self.assertTrue(table.XRD_n.eq(60).all())
        self.assertTrue(np.isfinite(table.ICP_MAE).all())
        for fold in info['partitions']:
            self.assertFalse(set(fold['train_batches'])&set(fold['test_batches']))
            self.assertEqual(set(fold['models']),set(MODELS))
        self.assertEqual(source,before)

    def test_baseline_predictions_do_not_read_test_outcomes(self):
        rows=collect(specimens(60),{},'REAL','chemistry_pulseicp')[0]
        train,test=rows[:40],rows[40:];changed=copy.deepcopy(test)
        for r in changed:r['assays']={'Fe':.001};r['outcomes']['xrd_pass']=1-r['outcomes']['xrd_pass']
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            for model in MODELS[:-1]:
                a,_=fitted_baseline(train,test,model,'chemistry_pulseicp')
                b,_=fitted_baseline(train,changed,model,'chemistry_pulseicp')
                np.testing.assert_allclose([v['predicted_w_wt_percent'] for v in a],[v['predicted_w_wt_percent'] for v in b])
                np.testing.assert_allclose([v['xrd_pass_predicted'] for v in a],[v['xrd_pass_predicted'] for v in b])

    def test_replay_keeps_fixed_test_batches_and_initial_observations(self):
        source=specimens(60);before=copy.deepcopy(source)
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            curves,queries,predictions,info=replay(source,{},'REAL',initial=24,budget=28,interval=2,
                sac_family='Gaussian residual',sac_settings=budget())
        self.assertEqual(set(curves.policy),set(POLICIES))
        self.assertEqual(len(info['initial_ids']),24)
        self.assertFalse(set(info['candidate_batches'])&set(info['test_batches']))
        self.assertFalse(set(queries.sample_id)&set(info['test_ids']))
        self.assertEqual(set(curves.observations),{24,26,28})
        changed=copy.deepcopy(source)
        for s in changed:
            if s['sample_id'] in info['test_ids']:
                s['xrd']['state']='FAIL' if s['xrd']['state']=='PASS' else 'PASS'
                s['icp'][0]['value']='.001'
        policies=['Random','GP-BO expected improvement']
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            _,q2,_,_=replay(changed,{},'REAL',policies=policies,initial=24,budget=28,interval=2)
        self.assertEqual(queries.loc[queries.policy.isin(policies)].to_dict('records'),q2.to_dict('records'))
        self.assertEqual(source,before)


if __name__=='__main__':unittest.main()
