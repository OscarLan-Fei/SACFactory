"""Public benchmark controls, results, charts and download-only methods."""
import json
import unittest
from streamlit.testing.v1 import AppTest
from test_pulseicp_bridge import specimens


class BenchmarkUITests(unittest.TestCase):
    def test_model_and_replay_results_render(self):
        script='import json\nfrom sacfactory.ui_benchmarks import render_benchmarks\n'
        script+=f'rows=json.loads({json.dumps(specimens())!r})\n'
        script+="render_benchmarks(rows,{},'REAL')"
        ui=AppTest.from_string(script,default_timeout=90).run()
        self.assertFalse(ui.exception)
        ui.multiselect(key='benchmark_models_REAL').set_value(['Mean / prevalence','Ridge / logistic'])
        ui.number_input(key='benchmark_groups_REAL').set_value(2)
        ui.button(key='benchmark_run_models_REAL').click().run()
        self.assertFalse(ui.exception)
        self.assertFalse(ui.error)
        _,table,_=ui.session_state['benchmark_models_result_REAL'][1]
        self.assertEqual(set(table.ICP_n),{60})
        self.assertTrue(any(w.value=='ICP regression fits by model' for w in ui.subheader))
        self.assertTrue(any(w.value=='Composite outcome score: predicted vs measured' for w in ui.subheader))
        self.assertGreaterEqual(len(ui.get('vega_lite_chart')),8)
        self.assertTrue(any('fit_slope' in w.value.columns for w in ui.dataframe))
        self.assertTrue(any(w.label=='Download ROC and precision-recall curves' for w in ui.get('download_button')))
        ui.multiselect(key='benchmark_policies_REAL').set_value(['Random'])
        ui.number_input(key='benchmark_initial_REAL').set_value(24)
        ui.number_input(key='benchmark_budget_REAL').set_value(28)
        ui.number_input(key='benchmark_interval_REAL').set_value(4)
        ui.button(key='benchmark_run_acquisition_REAL').click().run()
        self.assertFalse(ui.exception)
        self.assertFalse(ui.error)
        self.assertEqual(len(ui.session_state['benchmark_acquisition_result_REAL'][1][0]),2)
        self.assertEqual(len(ui.get('json')),0)


if __name__=='__main__':unittest.main()
