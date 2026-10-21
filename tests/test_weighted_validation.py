"""Meaningful outcome-mask, endpoint and multi-element weighting checks."""
import copy
import unittest
import numpy as np
import pandas as pd

from sacfactory.weighted_validation import weighted_loss,display_score,weighted_validation,aggregate
from test_element_campaign import singles


class WeightedValidationTests(unittest.TestCase):
    def test_formula_and_correctly_predicted_failure(self):
        self.assertAlmostEqual(weighted_loss(.12,.10,.8,1),.2826666666666665)
        self.assertEqual(display_score(weighted_loss(0,0,0,0)),100)
        with self.assertRaises(ValueError):
            weighted_loss(0,0,0,0,delta=0)

    def test_target_outcomes_are_masked_from_both_models_and_baselines(self):
        rows=singles(); original=copy.deepcopy(rows)
        first,_,_,_,info=weighted_validation(rows,{},'DEMO',['Co'],trees=12)
        self.assertEqual(len(first),24)
        self.assertEqual(info['folds'][0]['train_elements'],['Fe'])
        changed=copy.deepcopy(rows)
        for s in changed[24:]:
            s['icp'][0]['value']='70'
            s['xrd']['state']='PASS' if s['xrd']['state']=='FAIL' else 'FAIL'
        second,_,_,_,_=weighted_validation(changed,{},'DEMO',['Co'],trees=12)
        np.testing.assert_allclose(first.predicted_w_wt_percent,second.predicted_w_wt_percent)
        np.testing.assert_allclose(first.p_screen_pass,second.p_screen_pass)
        self.assertFalse(np.allclose(first.validation_loss,second.validation_loss))
        self.assertEqual(rows,original)

    def test_global_target_batch_purge_even_in_other_context(self):
        rows=singles()
        other=copy.deepcopy(rows[24]); other['sample_id']='OTHER'
        other['support_batch']='OTHER_CONTEXT'; other['batch_id']=rows[0]['batch_id']
        rows.append(other)
        _,_,_,_,info=weighted_validation(rows,{},'DEMO',['Co'],trees=12)
        fold=next(f for f in info['folds'] if len(f['test_sample_ids'])>1)
        self.assertNotIn(rows[0]['batch_id'],fold['train_batch_ids'])

    def test_censoring_and_missing_screen_do_not_receive_combined_scores(self):
        rows=singles()
        rows[24]['icp'][0].update(value='<0.005',flag='BLQ')
        rows[25]['icp_review']['state']='PENDING'; rows[25]['xrd']['state']='PENDING'
        points,_,_,audit,_=weighted_validation(rows,{},'DEMO',['Co'],trees=12)
        a=points.loc[points.sample_id==rows[24]['sample_id']].iloc[0]
        b=points.loc[points.sample_id==rows[25]['sample_id']].iloc[0]
        self.assertTrue(pd.isna(a.observed_w_wt_percent)); self.assertTrue(pd.isna(a.validation_score))
        self.assertTrue(pd.isna(b.label)); self.assertTrue(pd.isna(b.validation_score))
        self.assertTrue(any('Censored' in str(v) for v in audit.reason))

    def test_post_synthesis_electrical_and_measured_temperature_not_features(self):
        rows=singles()
        first=weighted_validation(rows,{},'DEMO',['Co'],trees=12)[0]
        for s in rows:
            s.update(temperature_C=99999,voltage_V=99999,current_A=99999)
        second=weighted_validation(rows,{},'DEMO',['Co'],trees=12)[0]
        np.testing.assert_array_equal(first.predicted_w_wt_percent,second.predicted_w_wt_percent)
        np.testing.assert_array_equal(first.p_screen_pass,second.p_screen_pass)

    def test_time_replay_never_trains_on_later_results(self):
        rows=singles()
        frame,_,_,_,info=weighted_validation(rows,{},'DEMO',['Co'],protocol='Chronological batch replay',trees=12)
        self.assertFalse(frame.empty)
        for fold in info['folds']:
            self.assertLess(fold['latest_train_result'],fold['first_test_result'])

    def test_multi_element_screen_is_counted_once_per_sample(self):
        points=pd.DataFrame([{'fold':'F','sample_id':'S','batch_id':'B','composition':'Fe+Co','element_count':2,
            'endpoint_set':'ICP + screening','icp_loss':q,'screening_loss':.25,
            'validation_loss':.6*q+.4*.25,'baseline_loss':2} for q in [1,3]])
        result=aggregate(points)
        self.assertEqual(len(result),1)
        self.assertAlmostEqual(result.iloc[0].validation_loss,1.3)
        self.assertAlmostEqual(result.iloc[0].icp_loss,2.)
