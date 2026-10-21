"""Independent continuous results, predictive masks and descriptive regression."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sacfactory.result_utility import CONTINUOUS,loading_utility
from sacfactory.system_score import observed_score,configuration,summary,fit_fold,PULSE_MODELS
from sacfactory import pulseicp_bridge as bridge
from test_system_score import record
from test_pulseicp_score import records,budget


class ContinuousResultTests(unittest.TestCase):
    def test_reference_loading_and_failure_are_continuous_not_flat(self):
        weights,target=configuration({'icp':.8,'screening':.2});r=record()
        r['assays']['Fe']=target
        self.assertEqual(observed_score(r,weights,target,CONTINUOUS)['observed_system_score'],100.)
        r['assays']['Fe']=2*target
        self.assertAlmostEqual(observed_score(r,weights,target,CONTINUOUS)['observed_system_score'],20+80*np.exp(-.5))
        r['outcomes']['screening']=0
        scores=[]
        for w in [2.5,5.,10.,19.]:
            r['assays']['Fe']=w;scores.append(observed_score(r,weights,target,CONTINUOUS)['observed_system_score'])
        self.assertTrue(all(a>b for a,b in zip(scores,scores[1:])))
        self.assertEqual(scores[0],20.)
        self.assertEqual(float(loading_utility(0,target,CONTINUOUS)),0.)
        self.assertAlmostEqual(float(loading_utility(target/2,target,CONTINUOUS)),np.exp(-.5))

    def test_each_family_uses_same_result_definition_and_cannot_see_held_results(self):
        rows=records();train,test=rows[:40],rows[40:];weights,target=configuration({'icp':.8,'screening':.2})
        expected=[observed_score(r,weights,target,CONTINUOUS)['observed_system_score'] for r in test]
        for family in PULSE_MODELS:
            first,info=fit_fold(train,test,weights,target,trees=5,model=family,pulse_settings=budget(),score_definition=CONTINUOUS)
            np.testing.assert_allclose([r['observed_system_score'] for r in first],expected)
            self.assertEqual(info['pulseicp_fusion']['score_definition'],CONTINUOUS)
            changed=copy.deepcopy(test)
            for r in changed:r['assays']={'Fe':19.};r['outcomes']['screening']=0
            second,_=fit_fold(train,changed,weights,target,trees=5,model=family,pulse_settings=budget(),score_definition=CONTINUOUS)
            np.testing.assert_array_equal([r['predicted_system_score'] for r in first],[r['predicted_system_score'] for r in second])

    def test_draw_expectation_uses_selected_nonlinear_transform(self):
        from test_pulseicp_bridge import specimens
        s=specimens(1)[0]
        fake=SimpleNamespace(origin='EXPERIMENTAL',physics=bridge.configuration()['physics'],protocols=[s['protocol_id']])
        fake.predict_compositions=lambda *a,**k:(pd.DataFrame({'composition_index':[0],
            'outside_training_support':[False]}),np.array([[1.25,5.]]))
        value=bridge.predict_attainment(fake,[s],'REAL',2.5,CONTINUOUS)[0]
        self.assertAlmostEqual(value['expected_attainment'],np.exp(-.5))
        self.assertNotAlmostEqual(value['expected_attainment'],float(loading_utility(3.125,2.5,CONTINUOUS)))

    def test_line_fit_does_not_replace_raw_prediction_accuracy(self):
        rows=[{'observed_system_score':2*x,'predicted_system_score':x,'baseline_system_score':40,
               'system_screening_observed':1} for x in [10,20,30]]
        before=copy.deepcopy(rows);metrics=summary(rows)
        self.assertAlmostEqual(metrics['linear_fit_slope'],2.)
        self.assertAlmostEqual(metrics['linear_fit_intercept'],0.)
        self.assertAlmostEqual(metrics['linear_fit_r_squared'],1.)
        self.assertAlmostEqual(metrics['r_squared'],-.75)
        self.assertEqual(rows,before)
