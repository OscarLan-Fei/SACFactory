"""Accuracy rewards correct failures without saturating observed ICP targets."""
import copy
import unittest
import numpy as np
from sacfactory.prediction_agreement import component_score,combine,LEGACY
from sacfactory.weighted_validation import weighted_validation,ROTATING_CV
from test_element_campaign import singles


class PredictionAgreementTests(unittest.TestCase):
    def test_confident_correct_pass_and_failure_receive_equal_credit(self):
        self.assertAlmostEqual(component_score('xrd_pass',(.05-0)**2),99.75)
        self.assertAlmostEqual(component_score('xrd_pass',(.95-1)**2),99.75)
        self.assertAlmostEqual(component_score('xrd_pass',(.95-0)**2),9.75)
        self.assertEqual(component_score('xrd_pass',1.),0.)

    def test_loading_has_no_attainment_cutoff_and_only_exact_error_is_perfect(self):
        self.assertEqual(component_score('icp',0),100)
        self.assertAlmostEqual(component_score('icp',1),50)
        self.assertLess(component_score('icp',4),component_score('icp',1))
        # Equal absolute errors at 2.5 and 19 wt% have equal agreement.
        self.assertEqual(component_score('icp',((2.75-2.5)/.25)**2),
                         component_score('icp',((19.25-19)/.25)**2))
        self.assertIsNone(combine({'icp':0},{'icp':.6,'xrd_pass':.4}))
        self.assertAlmostEqual(combine({'icp':0,'xrd_pass':.0025},{'icp':.6,'xrd_pass':.4}),99.9)
        self.assertAlmostEqual(combine({'icp':1,'xrd_pass':.25},{'icp':.6,'xrd_pass':.4},LEGACY),100/1.7)

    def test_grouped_metrics_are_means_of_complete_sample_scores(self):
        data=singles();before=copy.deepcopy(data)
        points,scores,report,_,info=weighted_validation(data,{},'DEMO',['Fe','Co'],
            protocol=ROTATING_CV,batches_per_fold=2,trees=8,delta=.25,
            endpoint_weights={'icp':.6,'xrd_pass':.4},include_system_score=False)
        np.testing.assert_allclose(scores.validation_score,.6*scores.icp_agreement_score+.4*scores.xrd_pass_agreement_score)
        np.testing.assert_allclose(points.xrd_failure_probability,1-points.xrd_pass_predicted)
        self.assertAlmostEqual(info['sample_weighted_score'],scores.validation_score.mean())
        self.assertAlmostEqual(info['baseline_score'],scores.baseline_agreement_score.mean())
        self.assertFalse(info['system_score']['enabled'])
        for row in report.to_dict('records'):
            ids=points.loc[points.element==row['element'],'sample_id']
            self.assertAlmostEqual(row['score'],scores.loc[scores.sample_id.isin(ids),'validation_score'].mean())
        self.assertEqual(data,before)
