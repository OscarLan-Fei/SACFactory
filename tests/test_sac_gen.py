"""SAC-Gen nested labels, independent failures, deployment and mutation guards."""
import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
from test_pulseicp_bridge import specimens
from sacfactory.sac_gen import SACGen
from sacfactory import pulseicp_bridge as bridge
from sacfactory.compute_runtime import execution
from sacfactory.weighted_validation import weighted_validation,ROTATING_CV
from sacfactory.prediction_agreement import component_score


def budget():
    return {**bridge.configuration()['model'],'members':1,'epochs_baseline':1,'epochs_diffusion':1,
        'draws_per_member':4,'diffusion_steps':10,'token_width':16,'attention_heads':2,'attention_layers':1}


class SACGenTests(unittest.TestCase):
    def test_nested_training_masks_and_prediction_is_independent_of_results(self):
        rows=specimens(60);train,test=rows[:40],rows[40:];settings=budget()
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            model=SACGen(settings=settings).fit(train,'REAL')
            first=model.predict(test)
            changed=copy.deepcopy(test)
            for s in changed:
                s['icp']=[];s['xrd']={'state':'FAIL'};s['icp_review']={'state':'FAIL'}
            second=model.predict(changed)
            np.testing.assert_array_equal([r['p_xrd_pass'] for r in first],[r['p_xrd_pass'] for r in second])
            np.testing.assert_array_equal([r['predicted_total_wt_percent'] for r in first],[r['predicted_total_wt_percent'] for r in second])
        self.assertEqual(set(model.xrd_info['training_ids']),set(s['sample_id'] for s in train))
        for fold in model.xrd_info['inner_folds']:
            self.assertFalse(set(fold['train_batches'])&set(fold['held_batches']))
            self.assertFalse(set(fold.get('loading_training_ids',[]))&set(fold['held_ids']))
        self.assertGreater(component_score('xrd_pass',(.1-0)**2),98.)

    def test_all_families_export_reload_and_changed_sources_are_rejected(self):
        rows=specimens(40)
        for family in ['Gaussian residual','Conditional diffusion','Transformer residual','Diffusion Transformer']:
            with tempfile.TemporaryDirectory() as folder,execution({'device':'cpu','threads':1,'cpu_jobs':1}):
                model=SACGen(family,budget()).fit(rows,'REAL')
                path=Path(folder)/'model';model.save(path)
                loaded=SACGen.load(path,rows)
                self.assertEqual(model.model_id,loaded.model_id)
                np.testing.assert_allclose([r['p_xrd_pass'] for r in model.predict(rows[:3])],
                    [r['p_xrd_pass'] for r in loaded.predict(rows[:3])])
                changed=copy.deepcopy(rows);changed[0]['icp'][0]['value']='0.1'
                with self.assertRaises(ValueError):SACGen.load(path,changed)

    def test_weighted_workflow_does_not_train_icp_pass_or_penalize_fail_labels(self):
        rows=specimens(60);before=copy.deepcopy(rows)
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            points,scores,_,_,info=weighted_validation(rows,{},'REAL',['Fe'],protocol=ROTATING_CV,
                batches_per_fold=2,profile='chemistry_pulseicp',icp_family='Gaussian residual',
                icp_settings=budget(),sac_gen=True)
        self.assertFalse(info['system_score']['enabled'])
        self.assertEqual(info['model_name'],'SAC-Gen')
        self.assertTrue(points.icp_pass_predicted.isna().all())
        self.assertTrue(points.xrd_pass_predicted.notna().all())
        self.assertEqual(set(points.xrd_pass_observed),{0,1})
        self.assertEqual(len(scores),60)
        self.assertEqual(rows,before)


if __name__=='__main__':unittest.main()
