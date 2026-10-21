"""Reader-facing validation has current controls and no development menus."""
import json
import unittest
from streamlit.testing.v1 import AppTest
from test_pulseicp_bridge import specimens


def app(rows):
    ui=AppTest.from_string("from sacfactory.ui_weighted_validation import render_weighted_validation\nimport streamlit as st\nrender_weighted_validation(st.session_state['rows'],{},'REAL')",default_timeout=90)
    ui.session_state['rows']=rows
    return ui.run()


class PublicValidationTests(unittest.TestCase):
    def test_architectures_and_nested_validation(self):
        ui=app(specimens(60));self.assertFalse(ui.exception)
        self.assertEqual(len(ui.selectbox(key='weighted_icp_family_REAL').options),4)
        self.assertFalse(any(w.key=='weighted_show_utility_REAL' for w in ui.checkbox))
        ui.selectbox(key='weighted_icp_family_REAL').set_value('Diffusion Transformer').run()
        ui.selectbox(key='weighted_device_REAL').set_value('cpu')
        ui.number_input(key='weighted_aux_members_REAL').set_value(1)
        ui.number_input(key='weighted_aux_epochs_REAL').set_value(1)
        ui.number_input(key='weighted_aux_draws_REAL').set_value(4)
        ui.number_input(key='weighted_aux_steps_REAL').set_value(10).run()
        ui.button(key='weighted_run_REAL').click().run()
        self.assertFalse(ui.exception)
        self.assertFalse(ui.error)
        points,_,_,_,info=ui.session_state['weighted_result_REAL'][1]
        self.assertFalse(info['system_score']['enabled'])
        self.assertEqual(points.xrd_pass_predicted.notna().sum(),60)
        self.assertTrue(all(p['pulseicp']['xrd']['inner_folds'] for p in info['folds']))
        charts=[json.loads(c.proto.spec) for c in ui.get('vega_lite_chart')]
        parity=next(c for c in charts if c.get('layer',[{}])[0].get('encoding',{}).get('x',{}).get('field')=='predicted_w_wt_percent')
        self.assertEqual(len(parity['layer']),3)
        self.assertEqual(parity['title'],'Held-out ICP predictions vs measurements')
        self.assertTrue(any(w.value=='ICP regression: predicted vs measured' for w in ui.subheader))
        self.assertTrue(any(w.label=='ICP RMSE / wt%' for w in ui.metric))
        self.assertEqual(ui.subheader[0].value,'Composite outcome score: predicted vs measured')
        self.assertTrue(any(w.label=='Score predictive R²' for w in ui.metric))
        stored=ui.session_state['weighted_result_REAL']
        ui.number_input(key='validation_outcome_REAL_score_weight').set_value(.7).run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.session_state['weighted_result_REAL'][0],stored[0])
        self.assertEqual(len(ui.session_state['weighted_result_REAL'][1][0]),60)
        agreement=next(c for c in charts if c.get('encoding',{}).get('y',{}).get('field')=='validation_score')
        self.assertEqual(agreement['encoding']['y']['title'],'Prediction agreement score / 100')
        self.assertFalse(any('Individual prediction accuracy' in w.value for w in ui.markdown))

    def test_batch_control_adapts_to_selected_data(self):
        ui=app(specimens(60));ui.number_input(key='weighted_cv_batches_REAL').set_value(4).run()
        ui.session_state['rows']=specimens(30)
        ui.run();self.assertFalse(ui.exception)
        self.assertEqual(ui.number_input(key='weighted_cv_batches_REAL').value,1)

    def test_xrd_only_result_explains_why_icp_regression_is_unavailable(self):
        ui=app(specimens(60))
        ui.number_input(key='weighted_aux_epochs_REAL').set_value(1)
        ui.number_input(key='weighted_aux_draws_REAL').set_value(4)
        ui.button(key='weighted_run_REAL').click().run()
        self.assertFalse(ui.exception)
        result=list(ui.session_state['weighted_result_REAL'][1])
        result[0]=result[0].copy()
        result[0]['predicted_w_wt_percent']=float('nan')
        view=AppTest.from_string("from sacfactory.ui_validation import render_results\nimport streamlit as st\nrender_results(st.session_state['result'],'REAL')")
        view.session_state['result']=tuple(result)
        view.run()
        self.assertFalse(view.exception)
        self.assertTrue(any('No paired numeric ICP' in w.value for w in view.info))


if __name__=='__main__':unittest.main()
