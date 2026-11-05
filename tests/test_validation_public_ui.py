"""Current multi-model validation controls and retained legacy result rendering."""
import json
import unittest
from streamlit.testing.v1 import AppTest
from test_pulseicp_bridge import specimens
from sacfactory.validation_groups import RANDOM_RECIPE_CV


def app(rows):
    ui=AppTest.from_string("from sacfactory.ui_weighted_validation import render_weighted_validation\nimport streamlit as st\nrender_weighted_validation(st.session_state['rows'],{},'REAL')",default_timeout=90)
    ui.session_state['rows']=rows
    return ui.run()


class PublicValidationTests(unittest.TestCase):
    def test_multiple_models_use_current_controls_and_matched_masks(self):
        ui=app(specimens(60));self.assertFalse(ui.exception)
        options=ui.multiselect(key='weighted_models_REAL').options
        self.assertIn('SAC-Gen-Diffusion',options);self.assertIn('SAC-Gen-GB',options)
        ui.multiselect(key='weighted_models_REAL').set_value(['Data-only Gradient boosting','Science prior (fixed)']).run()
        ui.selectbox(key='weighted_device_REAL').set_value('cpu')
        ui.number_input(key='weighted_aux_epochs_REAL').set_value(1)
        ui.selectbox(key='weighted_protocol_REAL').set_value(RANDOM_RECIPE_CV).run()
        ui.number_input(key='weighted_recipe_folds_REAL').set_value(3).run()
        ui.button(key='weighted_run_REAL').click().run()
        self.assertFalse(ui.exception);self.assertFalse(ui.error)
        result=ui.session_state['weighted_result_REAL'][1]
        self.assertEqual(result['kind'],'multi_model_validation')
        self.assertEqual(set(result['info']['models']),{'Data-only Gradient boosting','Science prior (fixed)'})
        self.assertEqual(result['points'].sample_id.nunique(),60)
        for fold in result['info']['partitions']:
            self.assertFalse(set(fold['train_ids']) & set(fold['test_ids']))
        self.assertTrue(ui.get('vega_lite_chart'))
        self.assertTrue(any('Composite outcome score' in w.value for w in ui.subheader))

    def test_batch_control_adapts_when_batch_protocol_is_selected(self):
        ui=app(specimens(60))
        ui.selectbox(key='weighted_protocol_REAL').set_value('Batch holdout').run()
        ui.number_input(key='weighted_cv_batches_REAL').set_value(4).run()
        ui.session_state['rows']=specimens(30)
        ui.run();self.assertFalse(ui.exception)
        self.assertEqual(ui.number_input(key='weighted_cv_batches_REAL').value,1)

    def test_xrd_only_legacy_result_explains_missing_numeric_icp(self):
        from sacfactory.ui_validation import calculate_validation
        from sacfactory.compute_runtime import execution
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            result=list(calculate_validation(json.dumps({'samples':specimens(60),'methods':{},'origin':'REAL',
                'targets':['Fe'],'protocol':RANDOM_RECIPE_CV,'profile':'chemistry_pulseicp','max_folds':2,'trees':12,
                'include_system_score':False})))
        result[0]=result[0].copy();result[0]['predicted_w_wt_percent']=float('nan')
        view=AppTest.from_string("from sacfactory.ui_validation import render_results\nimport streamlit as st\nrender_results(st.session_state['result'],'REAL')")
        view.session_state['result']=tuple(result);view.run()
        self.assertFalse(view.exception)
        self.assertTrue(any('No paired numeric ICP' in w.value for w in view.info))


if __name__=='__main__':unittest.main()
