"""Saved model forecasts retain targets, timing, revisions and specimen weights."""
import copy
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from sacfactory.forecast_review import follow_up, metrics, record_endpoint_times, compare_batch
from sacfactory.planning_models import initial_context


def specimen(ident='FOLLOWUP_1', measured=2.5, probability=.1, state='FAIL'):
    sample={**initial_context(),'sample_id':ident,'record_uid':ident+'_UID','batch_id':'BATCH_2',
            'icp':[],'xrd':{'state':'PENDING','mode':'MANUAL'}}
    recipe=copy.deepcopy(sample)
    forecast={'model_name':'Fixture model','model_id':'MODEL_A','p_xrd_pass':probability,
              'xrd_pass_threshold':.5,'predicted_total_wt_percent':2.4,
              'icp_predictions':[{'element':'Fe','predicted_mean_w_wt_percent':2.4}]}
    sample.update(icp=[{'element':'Fe','value':str(measured),'unit':'wt%','flag':'MEASURED'}],
                  xrd={'state':state,'mode':'MANUAL'},endpoint_reported_at={'icp':'2026-01-03T00:00:00Z','structure':'2026-01-04T00:00:00Z'})
    sample['proposal']={'recipe':recipe,'created':'2026-01-01T00:00:00Z','model_id':'MODEL_A',
                        'sac_gen_prediction':forecast,'evaluation_capture':{'training_membership_status':'NOT_RELEASED',
                        'training_records':[],'training_batch_ids':[]}}
    return sample


class ForecastReviewTests(unittest.TestCase):
    def test_fail_is_evaluated_independently_from_joint_icp_screening(self):
        sample=specimen()
        pairs,structures,audit=follow_up([sample],'REAL')
        self.assertEqual(len(pairs),1)
        self.assertEqual(structures.label.tolist(),[0])
        result=metrics(pairs,structures).iloc[0]
        self.assertAlmostEqual(result.structural_Brier,.01)
        self.assertEqual(result.FAIL_recall,1)
        self.assertAlmostEqual(result.ICP_MAE,.1)
        self.assertFalse(pairs.overlap_verified.any())
        self.assertTrue(audit.eligible.all())

    def test_partial_targets_and_censored_values_never_become_zero(self):
        sample=specimen()
        sample['icp'][0].update(value='ND',flag='ND')
        pairs,structures,audit=follow_up([sample],'REAL')
        self.assertTrue(pairs.empty)
        self.assertEqual(len(structures),1)
        self.assertIn('ND/BLQ',audit.loc[audit.endpoint=='icp','reason'].iloc[0])
        sample=specimen();sample['xrd']['state']='PENDING'
        pairs,structures,_=follow_up([sample],'REAL')
        self.assertEqual(len(pairs),1)
        self.assertTrue(structures.empty)

    def test_each_endpoint_uses_its_first_result_time(self):
        sample=specimen()
        sample['proposal']['created']='2026-01-03T12:00:00Z'
        pairs,structures,audit=follow_up([sample],'REAL')
        self.assertTrue(pairs.empty)
        self.assertEqual(len(structures),1)
        self.assertIn('timing',audit.loc[audit.endpoint=='icp','reason'].iloc[0])

    def test_recording_partial_results_preserves_time_without_backdating_imports(self):
        sample=specimen();old=copy.deepcopy(sample['proposal']['recipe'])
        sample['xrd']['state']='PENDING';sample.pop('endpoint_reported_at')
        record_endpoint_times(sample,old,'2026-01-05T00:00:00Z')
        self.assertEqual(sample['endpoint_reported_at'],{'icp':'2026-01-05T00:00:00Z'})
        older=copy.deepcopy(sample);sample['xrd']['state']='PASS'
        record_endpoint_times(sample,older,'2026-01-06T00:00:00Z')
        self.assertEqual(sample['endpoint_reported_at']['icp'],'2026-01-05T00:00:00Z')
        imported=specimen();imported.pop('endpoint_reported_at')
        record_endpoint_times(imported,copy.deepcopy(imported),'2026-01-10T00:00:00Z')
        self.assertNotIn('endpoint_reported_at',imported)

    def test_manual_partial_feedback_records_icp_time_before_structural_result(self):
        from sacfactory.storage import Store
        from sacfactory.manual import save_manual_revisions
        from sacfactory.pending_prediction import refresh_pending_prediction
        from sacfactory.planning import fingerprint
        with tempfile.TemporaryDirectory() as root:
            store=Store(root,'REAL')
            pending=copy.deepcopy(specimen()['proposal']['recipe'])
            store.import_bundle([pending],[])
            old=store.get('sample',pending['sample_id'])
            edited=copy.deepcopy(old)
            edited['icp']=[{'element':'Fe','value':'2.5','flag':'MEASURED','unit':'wt%'}]
            save_manual_revisions(store,[edited],{},'feedback',{old['sample_id']:fingerprint(old)})
            current=store.get('sample',old['sample_id'])
            self.assertIn('icp',current['endpoint_reported_at'])
            self.assertNotIn('structure',current['endpoint_reported_at'])
            self.assertFalse(current.get('result_reported_at'))
            current['icp']=[]
            store.put_many('sample',[current],'sample_id',overwrite=True)
            current=store.get('sample',old['sample_id'])
            with self.assertRaisesRegex(ValueError,'unmeasured'):
                refresh_pending_prediction(store,current['sample_id'],fingerprint(current))

    def test_training_overlap_and_changed_carbonization_inputs_are_excluded(self):
        for reason,mutate in [('training',lambda s:s['proposal']['evaluation_capture'].update(training_batch_ids=['BATCH_2'])),
                              ('Changed',lambda s:s.update(carbonization_loss_fraction=.15))]:
            sample=specimen();mutate(sample)
            pairs,structures,audit=follow_up([sample],'REAL')
            self.assertTrue(pairs.empty);self.assertTrue(structures.empty)
            self.assertTrue(audit.reason.str.contains(reason).all())

    def test_latest_matching_revision_counts_once_and_preserves_original(self):
        sample=specimen();original=copy.deepcopy(sample['proposal'])
        manual=copy.deepcopy(original);manual['created']='2026-01-02T00:00:00Z'
        manual['sac_gen_prediction']['icp_predictions'][0]['predicted_mean_w_wt_percent']=2.6
        sample['manual_prediction_history']=[copy.deepcopy(manual)]
        sample['manual_prediction']=manual
        pairs,structures,audit=follow_up([sample],'REAL')
        self.assertEqual(len(pairs),1);self.assertEqual(len(structures),1)
        self.assertEqual(pairs.predicted_wt_percent.iloc[0],2.6)
        self.assertEqual(sample['proposal'],original)
        self.assertEqual((~audit.eligible).sum(),2)

    def test_post_result_revision_does_not_hide_an_earlier_valid_forecast(self):
        sample=specimen()
        late=copy.deepcopy(sample['proposal']);late['created']='2026-01-08T00:00:00Z'
        late['sac_gen_prediction']['icp_predictions'][0]['predicted_mean_w_wt_percent']=99.
        sample['manual_prediction']=late
        pairs,structures,audit=follow_up([sample],'REAL')
        self.assertEqual(len(pairs),1);self.assertEqual(len(structures),1)
        self.assertEqual(pairs.predicted_wt_percent.iloc[0],2.4)
        self.assertEqual((~audit.eligible).sum(),2)

    def test_older_pending_rows_recover_complete_forecasts_from_the_queue(self):
        sample=specimen();queue=copy.deepcopy(sample['proposal'])
        queue['recommendation_id']='QUEUE_1';sample['recommendation_id']='QUEUE_1'
        sample['proposal'].pop('sac_gen_prediction')
        pairs,structures,_=follow_up([sample],'REAL',[queue])
        self.assertEqual(len(pairs),1);self.assertEqual(len(structures),1)

    def test_metrics_use_equal_specimen_weight_and_handle_constant_results(self):
        pairs=pd.DataFrame([{'model':'A','model_id':'A','sample_id':s,'element':e,
                             'predicted_wt_percent':0.,'observed_wt_percent':value,'specimen_weight':weight}
                            for s,e,value,weight in [('s1','Fe',0.,1.),('s2','Fe',10.,.5),('s2','Co',10.,.5)]])
        row=metrics(pairs,pd.DataFrame()).iloc[0]
        self.assertEqual(row.ICP_MAE,5.)
        pairs['observed_wt_percent']=2.
        self.assertIsNone(metrics(pairs,pd.DataFrame()).iloc[0].ICP_R_squared)

    def test_same_recipe_comparison_is_read_only_and_has_aligned_snapshots(self):
        class Model:
            is_public_release=True;training_samples=[];process_defaults={}
            def __init__(self,name):self.model_id=name
            def predict(self,recipes):
                return [{'model_id':self.model_id,'p_xrd_pass':.4,'predicted_total_wt_percent':2.,
                         'icp_predictions':[{'element':'Fe','predicted_mean_w_wt_percent':2.}]} for _ in recipes]
        rows=[{'recommendation_id':'PROPOSAL_1','origin':'REAL','recipe':initial_context()}]
        original=copy.deepcopy(rows)
        with patch('sacfactory.planning_models.load',side_effect=lambda root,origin,name,samples:Model(name)):
            captured,table=compare_batch(rows,'unused','REAL',['A','B'],[])
        self.assertEqual(rows,original)
        self.assertEqual(len(table),2)
        self.assertEqual(len(captured['PROPOSAL_1']),2)
        self.assertTrue(all(e['recipe']['concentrations_mM']=={'Fe':50.} for e in captured['PROPOSAL_1']))

    def test_follow_up_ui_renders_regression_and_endpoint_exports_without_writes(self):
        samples=[specimen('SYNTHETIC_'+str(i),measured=i+1,state='FAIL' if i==0 else 'PASS') for i in range(3)]
        for i,s in enumerate(samples):
            s['proposal']['sac_gen_prediction']['icp_predictions'][0]['predicted_mean_w_wt_percent']=i+.9
        original=copy.deepcopy(samples)
        app=AppTest.from_string("from sacfactory.ui_forecast_review import render_follow_up\nimport streamlit as st\nrender_follow_up(st.session_state['rows'],'REAL')")
        app.session_state['rows']=samples;app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.multiselect(key='followup_elements_REAL').options),118)
        self.assertGreaterEqual(len(app.get('vega_lite_chart')),2)
        self.assertTrue(any(x.label=='Export ICP prediction pairs CSV' for x in app.get('download_button')))
        self.assertEqual(app.session_state['rows'],original)
        app.checkbox(key='followup_separation_REAL').set_value(True).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value,'0')


if __name__=='__main__':unittest.main()
