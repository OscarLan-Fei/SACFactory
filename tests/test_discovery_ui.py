"""Lazy Discovery lab navigation and source-linked chat integration."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from test_pulseicp_bridge import specimens


class DiscoveryInterfaceTests(unittest.TestCase):
    def test_inverse_preview_pending_and_returned_feedback(self):
        from sacfactory.storage import Store
        from sacfactory.discovery.campaigns import CampaignLedger
        from test_discovery import forecast
        class Model:
            model_id='test-model'
            variant_name='Test model'
            training_keys=[]
            def predict(self,rows):return forecast(rows)
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder,'REAL');store.import_bundle(specimens(20),[])
            before=len(store.all('sample'))
            app=AppTest.from_string('''
from sacfactory.storage import Store
from sacfactory.ui_discovery import render_discovery
import streamlit as st
store=Store(st.session_state['root'],'REAL')
rows=store.all('sample')
render_discovery(store,rows,{},rows[0])
''',default_timeout=30)
            app.session_state['root']=folder
            app.session_state['discovery_tabs_REAL']='Inverse design'
            with patch('sacfactory.ui_discovery.cached_model',return_value=Model()):
                app.run();app.button(key='discovery_inverse_run_REAL').click().run()
                self.assertEqual(len(app.exception),0)
                app.button(key='discovery_inverse_preview_REAL_pending').click().run()
                self.assertEqual(len(app.exception),0)
                self.assertTrue(any('Added pending samples' in item.value for item in app.success))
                # A repeated save of the same rendered proposal cannot create a second batch.
                app.button(key='discovery_inverse_run_REAL').click().run()
                self.assertTrue(app.button(key='discovery_inverse_preview_REAL_pending').disabled)
            ledger=CampaignLedger(folder,'REAL');links=ledger.records('link')
            self.assertEqual(len(links),1);self.assertEqual(len(store.all('sample')),before+6)
            pending=next(s for s in store.all('sample') if s.get('discovery_proposal_id'))
            pending['icp']=[{'element':e,'value':1.,'unit':'wt%','flag':'MEASURED'} for e in pending['elements']]
            pending['xrd']={'mode':'MANUAL','state':'PASS'}
            store.import_bundle([pending],[],overwrite=True)
            self.assertEqual(ledger.evaluate(store.all('sample'))[0]['quantified_specimens'],1)

    def test_every_tab_renders_without_fitting_or_touching_experiments(self):
        from sacfactory.storage import Store
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder,'REAL');store.import_bundle(specimens(30),[]);before=store.all('sample')
            app=AppTest.from_string('''
from sacfactory.storage import Store
from sacfactory.ui_discovery import render_discovery
import streamlit as st
store=Store(st.session_state['root'],'REAL')
rows=store.all('sample')
render_discovery(store,rows,{},rows[0])
''',default_timeout=30)
            app.session_state['root']=folder
            with patch('sacfactory.ui_discovery.train_candidate') as train,patch('sacfactory.ui_discovery.cached_model') as load:
                app.run();self.assertEqual(len(app.exception),0)
                for name in ['Evolution','Knowledge','Model disagreement','Set model & updates','Inverse design','Hypotheses','Characterization','Validation']:
                    app.session_state['discovery_tabs_REAL']=name
                    app.run();self.assertEqual(len(app.exception),0,[str(e) for e in app.exception])
                train.assert_not_called();load.assert_not_called()
            self.assertEqual(before,store.all('sample'))

    def test_chat_memory_never_exposes_unshared_experiments(self):
        from types import SimpleNamespace
        from sacfactory.research_chat import ResearchTools
        rows=specimens(30)
        tools=ResearchTools(rows,{},SimpleNamespace(info={}),rows[0],{},'REAL',[rows[0]['sample_id']])
        result=tools.run('search_research_memory',{'query':'Fe SAC_0001 precursor','limit':8})
        self.assertTrue(result['evidence'])
        self.assertNotIn('SAC_0002',str(result))


if __name__=='__main__':unittest.main()
