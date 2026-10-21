"""Fixed experimental utilities, failed samples and honest held-out score forecasts."""
import copy
import json
import unittest

import numpy as np
import pandas as pd

from sacfactory.system_score import configuration,observed_score,PRIOR_RESIDUAL
from sacfactory.weighted_validation import weighted_validation,ROTATING_CV
from test_element_campaign import singles


def record():
    return {'sample':{'elements':['Fe'],'sample_condition':'RECOVERED'},
        'assays':{'Fe':1.25},'outcomes':{'screening':1,'icp_pass':1,'xrd_pass':1,
            'stem_confirmed':None,'burnout':0}}


class SystemScoreTests(unittest.TestCase):
    def test_prior_residual_matches_frozen_baseline_formula_and_zero_strength_is_direct(self):
        rows=singles()
        for r in rows:r.update(pan_mass_g=2.,solvent_volume_mL=20.)
        from sacfactory.weighted_validation import collect,split_records
        from sacfactory.system_score import fit_fold,prior_baseline
        from sklearn.ensemble import RandomForestRegressor
        records,_=collect(rows,{},'DEMO','chemistry_pulseicp')
        train,test,_,_=split_records(records,rows,'DEMO',['Fe','Co'],ROTATING_CV,3,2)[0]
        weights,target=configuration({'icp':.8,'screening':.2})
        predicted,info=fit_fold(train,test,weights,target,trees=8,model=PRIOR_RESIDUAL)
        bases=np.array([prior_baseline(r['sample'],weights,target,1.)[0] for r in train])
        values=np.array([observed_score(r,weights,target)['observed_system_score'] for r in train])
        forest=RandomForestRegressor(n_estimators=8,min_samples_leaf=3,max_depth=16,random_state=42,n_jobs=1).fit(
            [r['x'] for r in train],values-bases)
        expected=np.clip(np.array([prior_baseline(r['sample'],weights,target,1.)[0] for r in test])+
            forest.predict([r['x'] for r in test]),0.,100.)
        np.testing.assert_array_equal([r['predicted_system_score'] for r in predicted],expected)
        self.assertTrue(info['prior_used'])
        self.assertEqual(info['prior_exclusions'],[])
        direct,_=fit_fold(train,test,weights,target,trees=8)
        zero,_=fit_fold(train,test,weights,target,trees=8,model=PRIOR_RESIDUAL,prior_strength=0.)
        np.testing.assert_array_equal([r['predicted_system_score'] for r in direct],[r['predicted_system_score'] for r in zero])

    def test_prior_predictions_ignore_test_outcomes_and_do_not_change_experimental_scores(self):
        rows=singles()
        for r in rows:r.update(pan_mass_g=2.,solvent_volume_mL=20.)
        common={'protocol':ROTATING_CV,'batches_per_fold':2,'trees':8,'profile':'chemistry_pulseicp',
            'system_weights':{'icp':.8,'screening':.2}}
        direct=weighted_validation(rows,{},'DEMO',['Fe','Co'],**common)
        prior=weighted_validation(rows,{},'DEMO',['Fe','Co'],system_model=PRIOR_RESIDUAL,**common)
        np.testing.assert_array_equal(direct[1].observed_system_score,prior[1].observed_system_score)
        fold=prior[4]['folds'][0]
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['sample_id'] in fold['test_sample_ids']:
                r['icp'][0]['value']='42'
                r['xrd']['state']='FAIL'
        other=weighted_validation(changed,{},'DEMO',['Fe','Co'],system_model=PRIOR_RESIDUAL,**common)
        a=prior[1].loc[prior[1].fold==fold['fold']]
        b=other[1].loc[other[1].fold==fold['fold']]
        np.testing.assert_array_equal(a.predicted_system_score,b.predicted_system_score)
        np.testing.assert_array_equal(a.system_prior_total_wt_percent,b.system_prior_total_wt_percent)
        json.dumps(prior[4],allow_nan=False)

    def test_missing_prior_preparation_is_audited_without_assumptions_or_score_imputation(self):
        rows=singles()
        for r in rows:r.update(pan_mass_g=2.,solvent_volume_mL=20.)
        rows[0]['pan_mass_g']=None
        _,scores,_,_,info=weighted_validation(rows,{},'DEMO',['Fe','Co'],protocol=ROTATING_CV,
            trees=8,system_model=PRIOR_RESIDUAL)
        row=scores.loc[scores.sample_id==rows[0]['sample_id']].iloc[0]
        self.assertTrue(pd.isna(row.predicted_system_score))
        self.assertTrue(pd.notna(row.observed_system_score))
        self.assertTrue(any(p['sample_id']==rows[0]['sample_id'] for f in info['folds']
            for p in f['system_score_head']['prior_exclusions']))
        self.assertEqual(info['system_score']['metrics']['predicted_samples'],47)

    def test_same_fixed_formula_rewards_target_attainment_and_grades_failure(self):
        weights,target=configuration()
        r=record()
        self.assertAlmostEqual(observed_score(r,weights,target)['observed_system_score'],85.)
        r['outcomes']['screening']=0
        self.assertAlmostEqual(observed_score(r,weights,target)['observed_system_score'],15.)
        r['outcomes']['screening']=1
        for value in (2.5,10.):
            r['assays']['Fe']=value
            self.assertAlmostEqual(observed_score(r,weights,target)['observed_system_score'],100.)

    def test_burnout_is_zero_utility_without_icp_imputation_or_fake_test_labels(self):
        r=record();r['sample']['sample_condition']='BURNED_OUT'
        r['assays']={};r['outcomes'].update(screening=0,icp_pass=None,xrd_pass=None,burnout=1)
        before=copy.deepcopy(r)
        result=observed_score(r,*configuration())
        self.assertEqual(result['observed_system_score'],0.)
        self.assertIsNone(result['system_total_target_wt_percent'])
        self.assertIsNone(result['system_component_icp_pass'])
        self.assertEqual(r,before)

    def test_missing_components_keep_bounds_without_renormalization(self):
        r=record();r['assays']={};r['outcomes']['screening']=0
        result=observed_score(r,*configuration())
        self.assertIsNone(result['observed_system_score'])
        self.assertEqual(result['observed_system_score_lower'],0.)
        self.assertAlmostEqual(result['observed_system_score_upper'],30.)
        self.assertIn('icp',result['system_missing_components'])
        weights,target=configuration({'screening':1,'stem_confirmed':1})
        r['outcomes']['screening']=1
        result=observed_score(r,weights,target)
        self.assertIsNone(result['observed_system_score'])
        self.assertEqual(result['observed_system_score_lower'],50.)
        self.assertEqual(result['observed_system_score_upper'],100.)

    def test_mixture_uses_whole_recipe_total_not_each_member_target(self):
        r=record();r['sample']['elements']=['Fe','Co'];r['assays']={'Fe':1.,'Co':1.5}
        result=observed_score(r,*configuration())
        self.assertEqual(result['observed_system_score'],100.)
        self.assertEqual(result['system_total_target_wt_percent'],2.5)
        r['assays']={'Fe':60.,'Co':60.}
        self.assertIsNone(observed_score(r,*configuration())['observed_system_score'])
        for value in (0.,101.,float('nan')):
            with self.assertRaises(ValueError):configuration(target_wt_percent=value)

    def test_all_scores_use_whole_batch_masks_and_failures_remain_training_targets(self):
        rows=singles(); before=copy.deepcopy(rows)
        rows[3]['sample_condition']='BURNED_OUT'
        rows[3]['icp']=[]
        points,scores,_,_,info=weighted_validation(rows,{},'DEMO',['Fe','Co'],
            protocol=ROTATING_CV,batches_per_fold=2,trees=8)
        self.assertEqual(info['system_score']['metrics']['evaluated_samples'],48)
        self.assertGreater(info['system_score']['metrics']['failed_samples_evaluated'],0)
        self.assertFalse(scores.duplicated(['sample_id']).any())
        burned=scores.loc[scores.sample_id==rows[3]['sample_id']].iloc[0]
        self.assertEqual(burned.observed_system_score,0.)
        self.assertTrue(pd.isna(burned.system_total_target_wt_percent))
        self.assertTrue(pd.notna(burned.predicted_system_score))
        for fold in info['folds']:
            head=fold['system_score_head']
            self.assertFalse(set(head['training_sample_ids']) & set(fold['test_sample_ids']))
            self.assertFalse(set(head['training_batch_ids']) & set(fold['test_batch_ids']))
        changed=copy.deepcopy(rows)
        first_fold=info['folds'][0]
        for row in changed:
            if row['sample_id'] in first_fold['test_sample_ids']:
                row['icp']=[{**before[0]['icp'][0],'element':row['elements'][0],'value':'10'}]
                row['xrd']['state']='FAIL'
                row['sample_condition']='BURNED_OUT'
        later=weighted_validation(changed,{},'DEMO',['Fe','Co'],protocol=ROTATING_CV,batches_per_fold=2,trees=8)
        a=scores.loc[scores.fold==first_fold['fold']]
        b=later[1].loc[later[1].fold==first_fold['fold']]
        np.testing.assert_array_equal(a.predicted_system_score,b.predicted_system_score)
        np.testing.assert_array_equal(a.baseline_system_score,b.baseline_system_score)
        self.assertTrue((b.observed_system_score==0.).all())
        json.dumps(info,allow_nan=False)

    def test_unseen_element_score_can_learn_from_other_elements(self):
        rows=singles();before=copy.deepcopy(rows)
        _,scores,_,_,info=weighted_validation(rows,{},'DEMO',['Co'],trees=8)
        self.assertEqual(len(scores),24)
        self.assertTrue(scores.predicted_system_score.notna().all())
        self.assertEqual(info['folds'][0]['train_elements'],['Fe'])
        self.assertEqual(rows,before)

    def test_missing_selected_stem_prevents_score_training_without_blocking_other_heads(self):
        rows=singles()
        _,scores,_,_,info=weighted_validation(rows,{},'DEMO',['Fe','Co'],protocol=ROTATING_CV,
            trees=8,system_weights={'screening':1,'stem_confirmed':1})
        self.assertTrue(scores.predicted_system_score.isna().all())
        self.assertTrue(scores.observed_system_score.isna().all())
        self.assertEqual(info['system_score']['metrics']['evaluated_samples'],0)
        self.assertEqual(next(m for m in info['outcome_metrics'] if m['endpoint']=='screening')['evaluated_samples'],48)
