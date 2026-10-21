"""Independent score targets, whole-recipe completeness and regression statistics."""
import copy
import unittest
import numpy as np
import pandas as pd
from sacfactory.outcome_index import score_pairs,loading_coordinate,shared_score_support
from sacfactory.validation_diagnostics import regression_summary,probability_diagnostics,common_rows


def points():
    return pd.DataFrame([{'model':model,'fold':'A','sample_id':str(i),'batch_id':'B','composition':'Fe','element':'Fe',
        'observed_w_wt_percent':w,'predicted_w_wt_percent':w,'xrd_pass_observed':y,'xrd_pass_predicted':float(y)}
        for model in ['SAC-Gen','Baseline'] for i,(w,y) in enumerate([(0.,0),(1.,0),(2.5,1),(8.,0),(100.,1)])])


class OutcomeDiagnosticsTests(unittest.TestCase):
    def test_identical_endpoints_have_identical_scores_including_failures(self):
        frame=points();before=frame.copy(deep=True);scores=score_pairs(frame)
        np.testing.assert_allclose(scores.predicted_outcome_score,scores.observed_outcome_score)
        failed=scores.loc[scores.recorded_xrd=='FAIL']
        self.assertEqual(failed.observed_outcome_score.nunique(),3)
        self.assertEqual(loading_coordinate(0),0)
        self.assertEqual(loading_coordinate(100),1)
        self.assertLess(loading_coordinate(99),loading_coordinate(100))
        summary=regression_summary(scores,'predicted_outcome_score','observed_outcome_score')
        np.testing.assert_allclose(summary.predictive_r2,1)
        np.testing.assert_allclose(summary.fit_slope,1)
        pd.testing.assert_frame_equal(frame,before)

    def test_observed_scores_do_not_depend_on_predictions_and_missing_is_not_zero(self):
        original=points();changed=original.copy();changed['predicted_w_wt_percent']=20.;changed['xrd_pass_predicted']=.5
        np.testing.assert_allclose(score_pairs(original).observed_outcome_score,score_pairs(changed).observed_outcome_score)
        changed.loc[0,'observed_w_wt_percent']=np.nan
        self.assertEqual(score_pairs(changed).iloc[0].status,'INCOMPLETE_ENDPOINTS')
        changed.loc[1,'predicted_w_wt_percent']=-1.
        self.assertEqual(score_pairs(changed).iloc[1].status,'OUTSIDE_PHYSICAL_BOUNDS')

    def test_partial_mixture_requires_explicit_complete_totals(self):
        frame=points().iloc[:1].copy();frame['composition']='Fe+Co'
        self.assertEqual(score_pairs(frame).iloc[0].status,'INCOMPLETE_ENDPOINTS')
        frame['predicted_total_w_wt_percent']=4.;frame['observed_total_w_wt_percent']=4.
        row=score_pairs(frame).iloc[0]
        self.assertEqual(row.status,'PAIRED');self.assertAlmostEqual(row.predicted_outcome_score,row.observed_outcome_score)

    def test_sample_balancing_and_fit_r2_do_not_hide_prediction_bias(self):
        frame=points();frame['predicted_w_wt_percent']=2*frame.observed_w_wt_percent+1
        result=regression_summary(frame,'predicted_w_wt_percent','observed_w_wt_percent')
        np.testing.assert_allclose(result.fit_slope,.5)
        np.testing.assert_allclose(result.fit_intercept,-.5)
        np.testing.assert_allclose(result.fit_r2,1)
        self.assertTrue((result.predictive_r2<1).all())

    def test_probability_bins_keep_endpoints_and_common_support(self):
        frame=points();summary,bins,curves=probability_diagnostics(frame)
        self.assertEqual(set(summary.n),{5});self.assertEqual(bins.n.sum(),10)
        self.assertEqual(set(bins['bin']),{0,9});self.assertEqual(set(summary.brier),{0.})
        frame.loc[0,'predicted_w_wt_percent']=np.nan
        aligned=common_rows(frame)
        self.assertEqual(aligned.predicted_w_wt_percent.notna().sum(),8)
        missing=common_rows(frame,['SAC-Gen','Baseline','Unavailable model'])
        self.assertEqual(missing.predicted_w_wt_percent.notna().sum(),0)
        self.assertEqual(missing.xrd_pass_predicted.notna().sum(),0)

    def test_score_models_use_common_support_without_clipping_invalid_predictions(self):
        frame=points();frame.loc[0,'predicted_w_wt_percent']=-1
        scores=shared_score_support(score_pairs(frame))
        self.assertEqual(scores.comparison_common.sum(),8)
        self.assertTrue(scores.loc[scores.sample_id=='0','comparison_common'].eq(False).all())
        self.assertEqual(scores.loc[scores.model=='SAC-Gen'].iloc[0].status,'OUTSIDE_PHYSICAL_BOUNDS')

    def test_older_baseline_rows_can_share_recipe_identity_without_sharing_predictions(self):
        frame=points();frame.loc[frame.model=='Baseline','composition']=np.nan
        result=shared_score_support(score_pairs(frame))
        self.assertEqual(result.comparison_common.sum(),10)
        self.assertEqual(set(result.composition),{'Fe'})

    def test_plot_preserves_more_than_five_thousand_observations(self):
        from sacfactory.ui_model_diagnostics import parity_plot
        frame=pd.concat([points().iloc[:5]]*1001,ignore_index=True)
        frame['sample_id']=[str(i) for i in range(len(frame))]
        frame['recorded_xrd']='PASS'
        summary=regression_summary(frame,'predicted_w_wt_percent','observed_w_wt_percent')
        spec=parity_plot(frame,summary,'predicted_w_wt_percent','observed_w_wt_percent','ICP').to_dict()
        self.assertEqual(sum(v.get('kind')=='Observation' for v in spec['data']['values']),5005)


if __name__=='__main__':unittest.main()
