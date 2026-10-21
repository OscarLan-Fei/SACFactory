"""Selected experiment and element pools execute without changing measurements."""
import copy
import json
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from test_element_campaign import singles


class MultiSelectionUITests(unittest.TestCase):
    def test_analysis_scope_and_shared_calculation_pool_stay_distinct(self):
        rows=singles()
        app=AppTest.from_string('''
import streamlit as st
from sacfactory.ui_analysis import render_analysis
render_analysis(st.session_state['rows'],{},'DEMO')
''',default_timeout=30)
        app.session_state['rows']=rows;app.run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(next(w.value for w in app.metric if w.label=='Selected records'),'24')
        self.assertEqual(next(w.value for w in app.metric if w.label=='Shared ICP input records'),'48')
        app.selectbox(key='analysis_cohort_DEMO').set_value(1).run()
        app.multiselect(key='calculation_elements_DEMO').set_value(['Fe']).run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.selectbox(key='analysis_cohort_DEMO').value,0)
        self.assertEqual(next(w.value for w in app.metric if w.label=='Shared ICP input records'),'24')
        app.multiselect(key='calculation_samples_DEMO').set_value([rows[0]['sample_id'],rows[1]['sample_id']]).run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(next(w.value for w in app.metric if w.label=='Shared ICP input records'),'2')
        app.button(key='calculation_all_DEMO').click().run()
        self.assertEqual(next(w.value for w in app.metric if w.label=='Shared ICP input records'),'48')
        self.assertEqual(app.session_state['rows'],rows)

    def test_shared_icp_default_all_elements_and_subset_really_refits(self):
        rows=singles()
        app=AppTest.from_string('''
import streamlit as st
from sacfactory.ui_loading import render_loading
render_loading(st.session_state['rows'][:24], 'DEMO', all_samples=st.session_state['rows'])
''',default_timeout=30)
        app.session_state['rows']=rows
        app.run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.multiselect(key='pooled_loading_elements_DEMO').value,['Fe','Co'])
        self.assertEqual(next(w.value for w in app.metric if w.label=='Shared ICP input records'),'48')
        app.selectbox(key='pooled_loading_model_DEMO').set_value('Random forest').run()
        app.button(key='pooled_loading_run_DEMO').click().run()
        self.assertEqual(len(app.exception),0)
        points=app.session_state['pooled_loading_result_DEMO'][1][0]
        self.assertEqual(len(points),48)
        self.assertEqual(set(points.element),{'Fe','Co'})
        app.multiselect(key='pooled_loading_elements_DEMO').set_value(['Co']).run()
        self.assertFalse(any(w.label=='Shared held-out ICP pairs' for w in app.metric))
        app.button(key='pooled_loading_run_DEMO').click().run()
        self.assertEqual(len(app.session_state['pooled_loading_result_DEMO'][1][0]),24)
        self.assertEqual(app.session_state['rows'],rows)

    def test_shared_experiment_filters_retain_complete_mixtures_and_reset(self):
        rows=singles()
        rows[0]['elements']=['Fe','Ni'];rows[0]['concentrations_mM']['Ni']=.04
        app=AppTest.from_string('''
import streamlit as st
from sacfactory.ui_data_selection import select_experiments
st.session_state['chosen']=select_experiments(st.session_state['rows'],'DEMO')
''')
        app.session_state['rows']=copy.deepcopy(rows);app.run()
        self.assertEqual(len(app.session_state['chosen']),48)
        app.multiselect(key='calculation_elements_DEMO').set_value(['Ni','Co']).run()
        self.assertEqual(len(app.session_state['chosen']),25)
        self.assertEqual(app.session_state['chosen'][0]['elements'],['Fe','Ni'])
        app.multiselect(key='calculation_batches_DEMO').set_value([rows[0]['batch_id']]).run()
        app.multiselect(key='calculation_samples_DEMO').set_value([rows[0]['sample_id']]).run()
        self.assertEqual(app.session_state['chosen'],rows[:1])
        app.button(key='calculation_all_DEMO').click().run()
        self.assertEqual(app.session_state['chosen'],rows)
        self.assertEqual(app.session_state['rows'],rows)
        self.assertEqual(len(app.exception),0)

    def test_multi_target_dispatch_attempts_every_job_and_exports_failures(self):
        app=AppTest.from_string('''
import streamlit as st
import pandas as pd
from sacfactory.ui_calculation_batch import run_jobs,detail,select_targets
targets=select_targets('Targets',[0,1,2],str,'targets',default_all=True)
def calculate(value):
    st.session_state['calls'].append(value)
    if value==1:raise ValueError('Unavailable reference')
    return pd.DataFrame({'prediction':[value]}),{'target':value}
if st.button('Calculate',key='calculate'):
    st.session_state['results']=run_jobs([(str(i),i) for i in targets],calculate)
if 'results' in st.session_state:detail(st.session_state['results'],'jobs')
''')
        app.session_state['calls']=[];app.run()
        app.button(key='calculate').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.session_state['calls'],[0,1,2])
        self.assertEqual(app.session_state['results']['1'][1]['batch_status'],'Unavailable')
        self.assertEqual(len(app.session_state['results']),3)
        self.assertTrue(any(w.label=='Download all selected calculation results' for w in app.get('download_button')))
        app.selectbox(key='jobs_inspect').set_value('2').run()
        self.assertEqual(app.session_state['calls'],[0,1,2])

    def test_chemistry_targets_and_held_elements_all_run(self):
        rows=singles()
        app=AppTest.from_string('''
import streamlit as st
from sacfactory.ui_chemistry import render_chemistry_analysis
render_chemistry_analysis(st.session_state['rows'],st.session_state['rows'],{},'DEMO')
''',default_timeout=30)
        app.session_state['rows']=rows;app.run()
        self.assertEqual(len(app.exception),0)
        app.multiselect(key='chemistry_targets_DEMO').set_value([0,1])
        app.selectbox(key='chemistry_protocol_DEMO').set_value('unseen_element')
        app.multiselect(key='chemistry_elements_DEMO').set_value(['Fe','Co']).run()
        with patch('sacfactory.ui_chemistry.cached_ablation',return_value=(pd.DataFrame(),{'reason':'Test dispatch'})) as mocked:
            app.button(key='chemistry_run_DEMO').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(mocked.call_count,4)
        self.assertEqual({c.args[2] for c in mocked.call_args_list},{'Fe','Co'})
        self.assertEqual(len(app.session_state['chemistry_result_DEMO'][1]),4)
        self.assertEqual(app.session_state['rows'],rows)

    def test_pulseicp_contexts_and_structural_samples_dispatch_all_selected(self):
        import tempfile
        from test_pulseicp_bridge import specimens
        rows=specimens(origin='DEMO')
        for s in rows[30:]:s['support_batch']='SECOND'
        app=AppTest.from_string('''
import streamlit as st
from sacfactory.ui_pulseicp import render_pulseicp
render_pulseicp(st.session_state['rows'],'DEMO',st.session_state['root'])
''',default_timeout=30)
        with tempfile.TemporaryDirectory() as root:
            app.session_state['rows']=rows;app.session_state['root']=root;app.run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.multiselect(key='picp_contexts_DEMO').value,[0,1])
            with patch('sacfactory.ui_pulseicp.bridge.validate',side_effect=ValueError('Independent context unavailable')) as mocked:
                app.button(key='picp_run_DEMO').click().run()
            self.assertEqual(mocked.call_count,2)
            self.assertEqual(len(app.exception),0)
            results=app.session_state['picp_batch_result_DEMO'][1]
            self.assertEqual(len(results),2)
            self.assertTrue(all(r[2] for r in results))
            ids=[rows[0]['sample_id'],rows[30]['sample_id']]
            app.multiselect(key='picp_prior_samples_DEMO').set_value(ids).run()
            with patch('sacfactory._pulseicp.screening_prior.predict_xrd_prior',return_value=pd.DataFrame({'prior':[.5,.6]})) as mocked:
                app.button(key='picp_prior_run_DEMO').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(mocked.call_args.args[0]),2)
            self.assertTrue(any('sample_id' in w.value and list(w.value.sample_id)==ids for w in app.dataframe))
            self.assertEqual(app.session_state['rows'],rows)

    def test_pending_batch_reforecast_uses_selected_saved_samples(self):
        import tempfile
        from sacfactory.storage import Store
        from sacfactory.planning import fingerprint
        from sacfactory.domain import validate_sample
        rows=singles()[:3]
        for i,s in enumerate(rows):
            s.update(icp=[],icp_review={'state':'PENDING'},xrd={'state':'PENDING'},
                result_reported_at=None,batch_id=f'BATCH_{i+1:04d}',temperature_C=None,
                sample_condition='NOT_RECORDED',stem={'state':'NOT_TESTED'},electrical_input_kind='SETTING')
        app=AppTest.from_string('''
import streamlit as st
from sacfactory.ui_manual import render_pending,render_feedback
from sacfactory.storage import Store
store=Store(st.session_state['root'],'DEMO')
render_pending(store,store.all('sample'),{})
render_feedback(store,store.all('sample'),{})
''',default_timeout=30)
        with tempfile.TemporaryDirectory() as root:
            store=Store(root,'DEMO');store.put_many('sample',[validate_sample(s) for s in rows],'sample_id')
            before=copy.deepcopy(store.all('sample'))
            app.session_state['root']=root;app.run()
            self.assertEqual(len(app.exception),0)
            chosen=[rows[0]['sample_id'],rows[2]['sample_id']]
            app.multiselect(key='pending_reforecast_samples_DEMO').set_value(chosen).run()
            with patch('sacfactory.pending_prediction.refresh_pending_prediction',side_effect=[{},ValueError('Missing training coverage')]) as mocked:
                app.button(key='pending_predict_batch_DEMO').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(mocked.call_count,2)
            self.assertEqual([c.args[1] for c in mocked.call_args_list],chosen)
            original={s['sample_id']:s for s in before}
            self.assertTrue(all(c.args[2]==fingerprint(original[c.args[1]]) for c in mocked.call_args_list))
            report=app.session_state['pending_batch_forecast_report_DEMO']
            self.assertEqual([r['status'] for r in report],['Forecast metadata saved','Unavailable'])
            key='feedback_filters_DEMO_Pending experiments'
            app.multiselect(key=key).set_value(['BATCH_0001','BATCH_0003']).run()
            self.assertEqual(len(app.exception),0)
            options=next(w.options for w in app.multiselect if w.label=='Samples to return')
            self.assertEqual(set(options),set(chosen))
            self.assertEqual(store.all('sample'),before)


if __name__=='__main__':unittest.main()
