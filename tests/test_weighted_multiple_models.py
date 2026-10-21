"""Multiple conventional models share masked folds and one outcome definition."""
import copy
import json
import unittest
from streamlit.testing.v1 import AppTest
from test_pulseicp_bridge import specimens
from sacfactory.model_benchmarks import compare
from sacfactory.compute_runtime import execution


def view(both=False):
    script='import streamlit as st\nfrom sacfactory.ui_validation import render_weighted_validation\n'
    if both:script+="from sacfactory.ui_benchmarks import render_benchmarks\nrender_benchmarks(st.session_state['rows'],{},'REAL')\n"
    script+="render_weighted_validation(st.session_state['rows'],{},'REAL')"
    ui=AppTest.from_string(script,default_timeout=90);ui.session_state['rows']=specimens(60)
    return ui.run()


class WeightedModelTests(unittest.TestCase):
    def test_neural_and_conventional_models_share_masks_and_score_plot(self):
        ui=view();ui.multiselect(key='weighted_models_REAL').set_value(['SAC-Gen','Ridge / logistic','Random forest'])
        ui.number_input(key='weighted_aux_epochs_REAL').set_value(1)
        ui.number_input(key='weighted_aux_draws_REAL').set_value(4)
        ui.selectbox(key='weighted_device_REAL').set_value('cpu').run()
        ui.button(key='weighted_run_REAL').click().run()
        self.assertFalse(ui.exception);self.assertFalse(ui.error)
        saved=ui.session_state['weighted_result_REAL'];result=saved[1]
        self.assertEqual(result['kind'],'multi_model_validation')
        self.assertEqual(set(result['summary'].ICP_n),{60})
        self.assertEqual(set(result['summary'].model),{'SAC-Gen','Ridge / logistic','Random forest'})
        for fold in result['info']['partitions']:
            self.assertFalse(set(fold['train_batches'])&set(fold['test_batches']))
            self.assertEqual(len(fold['models']),3)
        self.assertEqual(ui.subheader[0].value,'Composite outcome score: predicted vs measured')
        ui.number_input(key='validation_outcome_REAL_score_weight').set_value(.7).run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.session_state['weighted_result_REAL'][0],saved[0])
        json.dumps(result['info'],allow_nan=False)

    def test_baselines_only_and_benchmark_results_coexist(self):
        ui=view(both=True)
        ui.multiselect(key='benchmark_models_REAL').set_value(['Mean / prevalence'])
        ui.button(key='benchmark_run_models_REAL').click().run()
        self.assertFalse(ui.exception)
        ui.multiselect(key='weighted_models_REAL').set_value(['Mean / prevalence','Ridge / logistic']).run()
        self.assertTrue(ui.selectbox(key='weighted_icp_family_REAL').disabled)
        ui.selectbox(key='weighted_protocol_REAL').set_value('Batch holdout').run()
        ui.button(key='weighted_run_REAL').click().run()
        self.assertFalse(ui.exception);self.assertFalse(ui.error)
        result=ui.session_state['weighted_result_REAL'][1]
        self.assertEqual(result['info']['protocol'],'Batch holdout')
        self.assertEqual(set(result['summary'].ICP_n),{60})
        ui.multiselect(key='weighted_models_REAL').set_value([]).run()
        self.assertTrue(ui.button(key='weighted_run_REAL').disabled)

    def test_time_masks_and_insufficient_source_folds_are_explicit(self):
        rows=specimens(60);before=copy.deepcopy(rows)
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            _,_,info=compare(rows,{},'REAL',['Fe'],models=['Mean / prevalence','Ridge / logistic'],
                protocol='Chronological batch replay',max_folds=6)
            for part in info['partitions']:
                self.assertFalse(set(part['train_batches'])&set(part['test_batches']))
                self.assertTrue(all(a<b for a in part['train_batches'] for b in part['test_batches']))
            empty,summary,info=compare(rows,{},'REAL',['Fe'],models=['Mean / prevalence'],protocol='Unseen element holdout')
            self.assertTrue(empty.empty)
            self.assertEqual(summary.iloc[0].ICP_n,0)
            self.assertTrue(info['partitions'][0]['models']['Mean / prevalence']['unavailable_reason'])
            with self.assertRaises(ValueError):compare(rows,{},'REAL',['Fe'],models=[])
        self.assertEqual(rows,before)


if __name__=='__main__':unittest.main()
