"""Primary SAC-Gen UI selects prediction accuracy, not synthesis utility."""
import json
import unittest
from streamlit.testing.v1 import AppTest
from test_pulseicp_bridge import specimens


class SACGenUITests(unittest.TestCase):
    def test_default_trains_icp_and_xrd_without_a_quality_score(self):
        rows=specimens(60,'DEMO')
        script='import json\nfrom sacfactory.ui_weighted_validation import render_weighted_validation\n'
        script+=f"rows=json.loads({json.dumps(rows)!r})\nrender_weighted_validation(rows,{{}},'DEMO')"
        ui=AppTest.from_string(script,default_timeout=90).run()
        self.assertFalse(ui.exception)
        self.assertFalse(any(w.key=='weighted_show_utility_DEMO' for w in ui.checkbox))
        self.assertEqual(ui.selectbox(key='weighted_icp_family_DEMO').value,'Gaussian residual')
        ui.number_input(key='weighted_aux_members_DEMO').set_value(1)
        ui.number_input(key='weighted_aux_epochs_DEMO').set_value(1)
        ui.number_input(key='weighted_aux_draws_DEMO').set_value(4)
        ui.selectbox(key='weighted_device_DEMO').set_value('cpu').run()
        ui.button(key='weighted_run_DEMO').click().run()
        self.assertFalse(ui.exception)
        self.assertFalse(ui.error)
        points,_,_,_,info=ui.session_state['weighted_result_DEMO'][1]
        self.assertEqual(info['model_name'],'SAC-Gen')
        self.assertFalse(info['system_score']['enabled'])
        self.assertEqual(points.xrd_pass_predicted.notna().sum(),60)
        self.assertTrue(points.icp_pass_predicted.isna().all())


if __name__=='__main__':unittest.main()
