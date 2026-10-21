"""Live progress is session-local and never replays cached layout closures."""
import json
import unittest
from streamlit.testing.v1 import AppTest
from test_pulseicp_bridge import specimens


VALIDATION_SCRIPT='''
import streamlit as st
import pandas as pd
from unittest.mock import patch
from sacfactory.compute_runtime import emit
from sacfactory import ui_validation as view

def backend(**parameters):
    st.session_state['fit_calls']=st.session_state.get('fit_calls',0)+1
    emit({'fold':1,'folds':2,'stage':'First training fold'})
    emit({'fold':2,'folds':2,'stage':'Second training fold'})
    emit({'completed':True,'folds':2})
    return (pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),
            {'outcome_metrics':[],'sample_weighted_score':None})

with patch.object(view,'weighted_validation',backend):
    view.render_weighted_validation(st.session_state['rows'],{},'REAL')
'''

BENCHMARK_SCRIPT='''
import json
import streamlit as st
from unittest.mock import patch
from sacfactory.compute_runtime import emit
from sacfactory import ui_benchmarks as view

def backend(**parameters):
    st.session_state['fit_calls']=st.session_state.get('fit_calls',0)+1
    emit({'fold':1,'folds':2,'stage':'First benchmark fold'})
    emit({'fold':2,'folds':2,'stage':'Second benchmark fold'})
    return {'value':parameters['value']}

value=st.number_input('Fixture value',value=1,key='value')
kind=st.session_state['kind']
payload=json.dumps({'task':kind,'runtime':{'device':'cpu','threads':1,'cpu_jobs':1},'value':value})
with patch.object(view,'compare',backend),patch.object(view,'replay',backend):
    view._run(payload,kind,'REAL')
'''


class LiveProgressTests(unittest.TestCase):
    def assert_clean(self,ui):
        self.assertFalse(ui.exception)
        self.assertFalse(ui.error)
        self.assertTrue(any(w.proto.value==100 for w in ui.get('progress')))

    def test_validation_progress_reuse_invalidation_and_new_session(self):
        for session in range(2):
            ui=AppTest.from_string(VALIDATION_SCRIPT,default_timeout=30)
            ui.session_state['rows']=specimens(30)
            ui.run().button(key='weighted_run_REAL').click().run()
            self.assert_clean(ui)
            self.assertEqual(ui.session_state['fit_calls'],1)
            ui.button(key='weighted_run_REAL').click().run()
            self.assert_clean(ui)
            self.assertEqual(ui.session_state['fit_calls'],1)
            ui.number_input(key='weighted_delta_REAL').set_value(.04).run()
            ui.button(key='weighted_run_REAL').click().run()
            self.assert_clean(ui)
            self.assertEqual(ui.session_state['fit_calls'],2)
            ui.session_state['rows']=specimens(29)
            ui.run().button(key='weighted_run_REAL').click().run()
            self.assert_clean(ui)
            self.assertEqual(ui.session_state['fit_calls'],3)

    def test_benchmark_progress_for_both_tasks_and_new_sessions(self):
        for kind in ('models','acquisition'):
            for session in range(2):
                ui=AppTest.from_string(BENCHMARK_SCRIPT,default_timeout=30)
                ui.session_state['kind']=kind
                ui.run().button(key=f'benchmark_run_{kind}_REAL').click().run()
                self.assert_clean(ui)
                self.assertEqual(ui.session_state['fit_calls'],1)
                ui.button(key=f'benchmark_run_{kind}_REAL').click().run()
                self.assert_clean(ui)
                self.assertEqual(ui.session_state['fit_calls'],1)
                ui.number_input(key='value').set_value(2).run()
                ui.button(key=f'benchmark_run_{kind}_REAL').click().run()
                self.assert_clean(ui)
                self.assertEqual(ui.session_state['fit_calls'],2)


if __name__=='__main__':unittest.main()
