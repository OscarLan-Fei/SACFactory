"""Learned physical score fusion, nested batch masks and tabular attention."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from sacfactory import pulseicp_bridge as bridge
from sacfactory.system_score import fit_fold,configuration,PULSE_MODELS,PULSE_GAUSSIAN,PULSE_TRANSFORMER
from sacfactory.weighted_validation import collect,weighted_validation,ROTATING_CV
from test_pulseicp_bridge import specimens


def budget():
    return {**bridge.configuration()['model'],'members':1,'epochs_baseline':1,'epochs_diffusion':1,
        'draws_per_member':4,'diffusion_steps':10,'token_width':16,'attention_heads':2,'attention_layers':1}


def records():
    return collect(specimens(),{},'REAL','chemistry_pulseicp')[0]


class PulseICPScoreTests(unittest.TestCase):
    def test_all_four_families_train_and_use_protected_learned_loading(self):
        rows=records();train,test=rows[:40],rows[40:];before=copy.deepcopy(rows)
        weights,target=configuration({'icp':.8,'screening':.2})
        for family in PULSE_MODELS:
            results,info=fit_fold(train,test,weights,target,trees=5,model=family,pulse_settings=budget())
            self.assertTrue(info['available'],str(info))
            self.assertEqual(len(results),20)
            self.assertTrue(all(0<=r['predicted_system_score']<=100 for r in results))
            self.assertTrue(all(0<=r['system_learned_expected_attainment']<=1 for r in results))
            np.testing.assert_allclose([r['system_loading_baseline_contribution'] for r in results],
                80*np.array([r['system_learned_expected_attainment'] for r in results]))
            nested=info['pulseicp_fusion'];held=[]
            self.assertEqual(nested['covered_training_samples'],40)
            for fold in nested['inner_folds']:
                self.assertFalse(set(fold['train_sample_ids'])&set(fold['held_sample_ids']))
                self.assertFalse(set(fold['train_batches'])&set(fold['held_batches']))
                self.assertFalse(set(fold['actual_training_ids'])&set(fold['held_sample_ids']))
                held.extend(fold['held_sample_ids'])
            self.assertEqual(sorted(held),sorted(r['sample']['sample_id'] for r in train))
            self.assertFalse(set(nested['outer_model']['training_ids'])&{r['sample']['sample_id'] for r in test})
            json.dumps(info,allow_nan=False)
        self.assertEqual([r['sample'] for r in rows],[r['sample'] for r in before])

    def test_test_icp_labels_and_burnout_do_not_change_fusion_predictions(self):
        rows=records();train,test=rows[:40],rows[40:]
        weights,target=configuration({'icp':.8,'screening':.2})
        first,_=fit_fold(train,test,weights,target,trees=5,model=PULSE_TRANSFORMER,pulse_settings=budget())
        changed=copy.deepcopy(test)
        for r in changed:
            r['assays']={'Fe':40.};r['outcomes'].update(screening=0,burnout=1)
            r['sample'].update(sample_condition='BURNED_OUT',icp=[])
        second,_=fit_fold(train,changed,weights,target,trees=5,model=PULSE_TRANSFORMER,pulse_settings=budget())
        np.testing.assert_array_equal([r['predicted_system_score'] for r in first],[r['predicted_system_score'] for r in second])
        np.testing.assert_array_equal([r['system_learned_expected_attainment'] for r in first],[r['system_learned_expected_attainment'] for r in second])
        self.assertTrue(all(r['observed_system_score']==0 for r in second))

    def test_capped_utility_uses_joint_member_draws_not_capped_mean(self):
        s=specimens(1)[0];s['elements']=['Fe','Co'];s['concentrations_mM']['Co']=10.
        fake=SimpleNamespace(origin='EXPERIMENTAL',physics=bridge.configuration()['physics'],protocols=[s['protocol_id']])
        fake.predict_compositions=lambda *a,**k:(pd.DataFrame({'composition_index':[0,0],
            'outside_training_support':[False,False]}),np.array([[0.,4.],[0.,4.]]))
        result=bridge.predict_attainment(fake,[s],'REAL',2.5)[0]
        self.assertEqual(result['total_mean_wt_percent'],4.)
        self.assertEqual(result['expected_attainment'],.5)
        self.assertNotEqual(result['expected_attainment'],min(result['total_mean_wt_percent']/2.5,1))

    def test_missing_materials_and_small_inner_pools_are_audited_not_fallback(self):
        rows=records();train,test=rows[:40],copy.deepcopy(rows[40:])
        test[0]['sample']['pan_mass_g']=None
        weights,target=configuration({'icp':.8,'screening':.2})
        result,info=fit_fold(train,test,weights,target,trees=5,model=PULSE_GAUSSIAN,pulse_settings=budget())
        self.assertIsNone(result[0]['predicted_system_score'])
        self.assertIsNotNone(result[0]['observed_system_score'])
        self.assertTrue(any(e['sample_id']==test[0]['sample']['sample_id'] for e in info['pulseicp_fusion']['exclusions']))
        result,info=fit_fold(train[:20],test,weights,target,trees=5,model=PULSE_GAUSSIAN,pulse_settings=budget())
        self.assertFalse(info['available'])
        self.assertTrue(all(r['predicted_system_score'] is None for r in result))
        self.assertIn('3 outer-training batches',info['pulseicp_fusion']['reason'])

    def test_zero_strength_does_not_fit_network_and_matches_direct(self):
        rows=records();weights,target=configuration({'icp':.8,'screening':.2})
        direct,_=fit_fold(rows[:40],rows[40:],weights,target,trees=5)
        with patch('sacfactory.pulseicp_score.loading_baselines') as mocked:
            result,_=fit_fold(rows[:40],rows[40:],weights,target,trees=5,model=PULSE_GAUSSIAN,prior_strength=0.)
        mocked.assert_not_called()
        np.testing.assert_array_equal([r['predicted_system_score'] for r in direct],[r['predicted_system_score'] for r in result])

    def test_rotating_validation_pools_every_outer_prediction_and_preserves_scores(self):
        rows=specimens();before=copy.deepcopy(rows)
        common={'protocol':ROTATING_CV,'batches_per_fold':2,'trees':5,'profile':'chemistry_pulseicp',
            'system_weights':{'icp':.8,'screening':.2}}
        direct=weighted_validation(rows,{},'REAL',['Fe'],**common)
        fused=weighted_validation(rows,{},'REAL',['Fe'],system_model=PULSE_GAUSSIAN,
            system_icp_settings=budget(),**common)
        self.assertEqual(fused[4]['system_score']['metrics']['evaluated_samples'],60)
        self.assertFalse(fused[1].duplicated(['sample_id']).any())
        np.testing.assert_array_equal(direct[1].observed_system_score,fused[1].observed_system_score)
        self.assertEqual(rows,before)
        self.assertEqual(len(fused[4]['folds']),3)

    def test_transformer_attention_does_not_mix_experiments(self):
        from sacfactory.token_residual import TransformerGaussian,TransformerDenoiser
        torch.manual_seed(5);x=torch.randn(3,7)
        gaussian=TransformerGaussian(7,16,2,1).eval()
        with torch.no_grad():
            np.testing.assert_allclose(gaussian(x)[0][0].numpy(),gaussian(x[:1])[0][0].numpy(),atol=1e-6)
            denoiser=TransformerDenoiser(7,16,2,1).eval()
            z=torch.randn(3,1);t=torch.tensor([.2,.4,.6])
            np.testing.assert_allclose(denoiser(z,t,x)[0].numpy(),denoiser(z[:1],t[:1],x[:1])[0].numpy(),atol=1e-6)
