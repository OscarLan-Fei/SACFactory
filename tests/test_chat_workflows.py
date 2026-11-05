"""Conversation-driven template revisions, units, resources and board targets."""
from contextlib import closing
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from streamlit.testing.v1 import AppTest
from sacfactory.language import ApiConfig,LanguageError
from sacfactory.research_chat import ResearchAdapter,ResearchTools
from sacfactory.storage import Store
from sacfactory.workflow_service import WorkflowService
from sacfactory.workflow_chat import WorkflowChat,local_chat_workflow,scoped_state
from sacfactory.workflow_worker import worker_status
from sacfactory.ui_chat_workflows import queue_board,apply_board_request
from test_pulseicp_bridge import specimens


def call(name,arguments,ident='tool1'):
    return {'status':'completed','output':[{'type':'function_call','call_id':ident,'name':name,'arguments':json.dumps(arguments)}]}


def final(text='Workflow changes saved.'):
    return {'status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':text}]}]}


class ChatWorkflowTests(unittest.TestCase):
    def service(self,folder,origin='REAL',count=6,selected=None):
        store=Store(folder,origin)
        if not store.all('sample'):
            rows=specimens(count,origin)
            for i,row in enumerate(rows):row['record_uid']='chat_fixture_'+str(i)
            store.import_bundle(rows,[])
        return WorkflowService(store,selected_ids=selected)

    def tools(self,service,state):
        return ResearchTools(service.store.all('sample'),{},SimpleNamespace(info={'deferred':True}),service.context,{},
                             service.store.origin,service.selected_ids,workflow_service=service,workflow_state=state)

    def wait_worker(self,service):
        deadline=time.monotonic()+8
        while worker_status(service)['active'] and time.monotonic()<deadline:time.sleep(.02)
        self.assertFalse(worker_status(service)['active'])

    def test_multiturn_provider_controls_template_device_preparation_and_run(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);before=service.store.all('sample')
            state=scoped_state(None,'REAL',service.selected_ids);config=ApiConfig('https://example.org/v1','mock','fake-private-key','responses')
            first=[call('workflow_template',{'action':'preview','template_id':'real_feedback_loop','workflow_id':None,
                     'options_json':json.dumps({'elements':['Fe'],'bounds_mM':{'Fe':[50.,50.]},'pulses':[4,8]})}),final('Preview ready.')]
            ResearchAdapter(config,True,lambda *args:first.pop(0),max_calls=6).converse('Choose an operator-assisted feedback loop.',[],{},self.tools(service,state))
            self.assertEqual(state['preview']['options']['bounds_mmol'],{'Fe':[1.,1.]})
            self.assertFalse(service.db.all('definitions'));self.assertFalse(service.db.all('devices'))
            requests=[]
            replies=[call('workflow_devices',{'action':'register','definition_json':json.dumps({'profile_id':'icp_ms','id':'lab_icp_ms','mode':'INVENTORY_ONLY'})}),
                call('workflow_template',{'action':'preview','template_id':None,'workflow_id':None,'options_json':json.dumps({'solvent_volume_mL':40.,'bounds_mM':{'Fe':[50.,50.]},'bindings':{'icp':'lab_icp_ms'}})},'tool2'),
                call('workflow_template',{'action':'create','template_id':None,'workflow_id':None,'options_json':None},'tool3'),
                call('workflow_control',{'action':'start','workflow_id':None,'run_id':None,'spec_json':None},'tool4'),final('Saved and ready; execution has not started.')]
            def transport(request,timeout):requests.append(json.loads(request.data));return replies.pop(0)
            result=ResearchAdapter(config,True,transport,max_calls=6).converse('Keep 50 mM in 40 mL DMF, use the ICP-MS, save and start the run.',[],{},self.tools(service,state))
            self.assertEqual(result.api_calls,5)
            self.assertIn('real_feedback_loop',requests[0]['instructions'])
            row=service.db.get('definitions',state['workflow_id']);options=row['template']['options']
            self.assertEqual(options['bindings']['icp'],'lab_icp_ms');self.assertEqual(options['bounds_mmol'],{'Fe':[2.,2.]})
            run=service.db.get('runs',state['run_id'])
            self.assertEqual(run['state'],'READY');self.assertEqual(run['context']['solvent_volume_mL'],40.)
            from sacfactory.discovery.inverse import candidates
            pool=candidates(run['context'],['Fe'],options['bounds_mmol'],options['temperature'],options['pulses'],8)
            self.assertTrue(all(abs(r['concentrations_mM']['Fe']-50.)<1e-10 for r in pool))
            self.assertEqual(service.store.all('sample'),before);self.assertFalse(service.db.all('jobs'))

    def test_revisions_keep_prior_runs_and_preserve_earlier_preview_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);state={'origin':'REAL'};chat=WorkflowChat(service,state)
            created=chat.template('create',template_id='manual_synthesis',options_json=json.dumps({'solvent_volume_mL':30.,'pulses':[2]}))
            chat.control('start');old_run=copy.deepcopy(service.db.get('runs',state['run_id']))
            next_turn=WorkflowChat(service,state)
            next_turn.template('preview',options_json=json.dumps({'pulses':[4,8]}))
            next_turn.template('preview',options_json=json.dumps({'pan_mass_g':3.}))
            revised=next_turn.template('revise')
            row=service.db.get('definitions',revised['workflow_id'])
            self.assertEqual(row['template']['options']['pulses'],[4,8]);self.assertEqual(row['template']['options']['pan_mass_g'],3.)
            self.assertEqual(row['template']['options']['solvent_volume_mL'],30.)
            self.assertEqual(row['spec']['version'],2);self.assertEqual(revised['revision_of'],created['workflow_id'])
            self.assertEqual(service.db.get('runs',old_run['id']),old_run)
            self.assertEqual(next_turn.template('revise'),revised)
            self.assertEqual(len(service.db.all('definitions')),2)

    def test_source_context_changes_do_not_override_saved_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);created=service.instantiate_template('inverse_screening',{'solvent_volume_mL':40.,'pan_mass_g':3.,'voltage_V':32.,'current_A':80.})
            other=WorkflowService(service.store,{**service.context,'solvent_volume_mL':10.,'pan_mass_g':1.})
            run=other.start(created['workflow_id'])
            self.assertEqual(run['context']['solvent_volume_mL'],40.);self.assertEqual(run['context']['pan_mass_g'],3.)

    def test_concentration_alias_uses_current_volume_and_preserves_other_elements(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            service=WorkflowService(service.store,{**service.context,'solvent_volume_mL':40.})
            state={'origin':'REAL'};chat=WorkflowChat(service,state)
            chat.template('preview',template_id='manual_synthesis',options_json=json.dumps({'bounds_mM':{'Fe':[50.,50.]}}))
            self.assertEqual(state['preview']['options']['bounds_mmol'],{'Fe':[2.,2.]})
            chat.template('preview',options_json=json.dumps({'elements':['Fe','Co'],'bounds_mmol':{'Fe':[2.,2.],'Co':[.5,.5]}}))
            chat.template('preview',options_json=json.dumps({'bounds_mM':{'Fe':[25.,25.]}}))
            self.assertEqual(state['preview']['options']['bounds_mmol'],{'Fe':[1.,1.],'Co':[.5,.5]})

    def test_invalid_edits_and_wrong_modality_do_not_change_saved_workflow(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,'DEMO');chat=WorkflowChat(service,{'origin':'DEMO'})
            created=chat.template('create',template_id='simulation_loop');before=copy.deepcopy(service.db.get('definitions',created['workflow_id']))
            for patch in [{'bindings':{'xrd':'demo_icp'}},{'solvent_volume_mL':0},{'bounds_mM':{'Fe':[True,50.]}},{'api_key':'unsupported'}]:
                with self.assertRaises(ValueError):chat.template('revise',options_json=json.dumps(patch))
            self.assertEqual(service.db.get('definitions',created['workflow_id']),before)
            self.assertEqual(len(service.db.all('definitions')),1)

    def test_background_controls_acknowledgement_and_live_status(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);state={'origin':'REAL'};chat=WorkflowChat(service,state)
            chat.template('create',template_id='precursor_preparation');chat.control('start')
            started=chat.control('run_async');self.assertTrue(started['board']['run_id'])
            self.wait_worker(service)
            self.assertEqual(chat.control('status')['state'],'WAITING')
            with closing(service.db.connect()) as db:self.assertEqual(db.execute('SELECT count(*) FROM locks').fetchone()[0],1)
            with self.assertRaises(ValueError):chat.control('acknowledge',spec_json=json.dumps({'step_id':'weigh','note':''}))
            chat.control('acknowledge',spec_json=json.dumps({'step_id':'weigh','note':'Operator recorded actual masses.'}))
            self.assertEqual(chat.control('status')['completed_steps'],1)
            chat.control('cancel')
            with closing(service.db.connect()) as db:self.assertFalse(db.execute('SELECT * FROM locks').fetchall())
            self.assertFalse(service.db.all('jobs'))

    def test_provider_failure_after_creation_preserves_the_local_definition(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);state={'origin':'REAL'}
            replies=[call('workflow_template',{'action':'create','template_id':'precursor_preparation','workflow_id':None,'options_json':None})]
            def transport(request,timeout):
                if replies:return replies.pop(0)
                raise HTTPError('https://example.org/v1/responses',503,'Unavailable',{},None)
            with self.assertRaises(LanguageError):ResearchAdapter(ApiConfig('https://example.org/v1','mock','fake-key','responses'),True,transport).converse('Create the preparation workflow.',[],{},self.tools(service,state))
            self.assertTrue(state['workflow_id']);self.assertEqual(len(service.db.all('definitions')),1)
            self.assertEqual(WorkflowChat(service,state).control('get')['workflow_id'],state['workflow_id'])
            self.assertFalse(service.db.all('runs'))

    def test_inspection_respects_selection_and_templates_respect_namespace(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);created=service.instantiate_template('data_benchmark')
            restricted=WorkflowService(service.store,selected_ids=service.selected_ids[:1])
            with self.assertRaises(ValueError):WorkflowChat(restricted,{'origin':'REAL'}).control('get',workflow_id=created['workflow_id'])
            with self.assertRaises(ValueError):WorkflowChat(service,{'origin':'REAL'}).template('create',template_id='simulation_loop')
            with self.assertRaises(ValueError):WorkflowChat(service,{'origin':'DEMO'})

    def test_scope_expansion_keeps_active_work_while_narrowing_or_new_discussion_resets(self):
        state=scoped_state(None,'REAL',['one'],'discussion1');state.update(workflow_id='known',selected_ids=['one'])
        self.assertIs(scoped_state(state,'REAL',['one','two'],'discussion1'),state)
        self.assertNotIn('workflow_id',scoped_state(state,'REAL',[],'discussion1'))
        self.assertNotIn('workflow_id',scoped_state(state,'REAL',['one'],'discussion2'))
        self.assertNotIn('workflow_id',scoped_state(state,'DEMO',['one'],'discussion1'))

    def test_legacy_template_copy_can_be_recovered_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);plan=service.template_plan('manual_synthesis',{'register_resources':True,'pulses':[2,4]})
            for device in plan['resources_to_register']:service.devices.register(device)
            original=service.db.save(plan['spec'])
            with closing(service.db.connect()) as db,db:service.db.event(db,action='INSTANTIATE_TEMPLATE',workflow_id=original['id'],template_id='manual_synthesis')
            chat=WorkflowChat(service,{'origin':'REAL'})
            updated=chat.template('revise',workflow_id=original['id'],options_json=json.dumps({'pulses':[8]}))
            self.assertEqual(service.db.get('definitions',original['id']),original)
            self.assertEqual(service.db.get('definitions',updated['workflow_id'])['template']['options']['pulses'],[8])

    def test_local_commands_share_state_and_board_navigation_preserves_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,selected=None);state={'origin':'REAL'};chat=WorkflowChat(service,state)
            local_chat_workflow('/workflow preview manual_synthesis',chat)
            local_chat_workflow('/workflow edit {"solvent_volume_mL":40,"bounds_mM":{"Fe":[50,50]}}',chat)
            session={};queue_board(session,state);apply_board_request(session,service.store.all('sample'),'REAL')
            self.assertEqual(session['workflow_REAL_graph_scope'],'Chat preview')
            self.assertEqual(session['workflow_chat_preview_REAL']['options']['bounds_mmol'],{'Fe':[2.,2.]})
            self.assertTrue(session['workflow_REAL_all_sources'])
            saved=local_chat_workflow('/workflow create',chat)
            local_chat_workflow('/workflow start',chat)
            queue_board(session,state);apply_board_request(session,service.store.all('sample'),'REAL')
            self.assertEqual(session['workflow_REAL_board_run_selection'],state['run_id'])
            self.assertEqual(session['workflow_REAL_graph_scope'],'Run')
            self.assertFalse(session.get('workflow_chat_preview_REAL'))
            self.assertEqual(saved['workflow_id'],state['workflow_id'])

    def test_shared_record_widget_survives_board_roundtrip(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'SACFACTORY_DATA_DIR':folder}):
            service=self.service(folder);ids=service.selected_ids
            app=AppTest.from_file(str(root/'app.py'),default_timeout=60)
            app.session_state['assistant_mode_setting']='Research conversation'
            app.session_state['seen_data_epoch_REAL']=(service.store.get('meta','data-epoch') or {}).get('value','initial')
            app.session_state['research_samples_REAL']=ids
            app.run();self.assertFalse(app.exception)
            app.chat_input[0].set_value('/workflow preview precursor_preparation').run()
            app.button(key='chat_flow_REAL_open').click().run()
            self.assertEqual(app.session_state['research_samples_REAL'],ids)
            app.button(key='workflow_REAL_chat_preview_save').click().run()
            next(b for b in app.button if b.label=='Continue in conversation').click().run()
            self.assertFalse(app.exception,[e.value for e in app.exception])
            self.assertEqual(app.multiselect(key='research_samples_REAL').value,ids)
            self.assertEqual(app.session_state['chat_workflow_state_REAL']['selected_ids'],ids)

    def test_chat_preview_does_not_fit_the_planner_and_opens_the_corresponding_board(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'SACFACTORY_DATA_DIR':folder}):
            service=self.service(folder);before=service.store.all('sample')
            app=AppTest.from_file(str(root/'app.py'),default_timeout=60)
            with patch('sacfactory.planning.fit_model',side_effect=AssertionError('Workflow requests must not fit the planner')):
                app.run();self.assertFalse(app.exception)
                app.chat_input[0].set_value('/workflow preview precursor_preparation {"pan_mass_g":3,"solvent_volume_mL":30}').run()
                self.assertFalse(app.exception,[e.value for e in app.exception])
                self.assertEqual(app.session_state['chat_workflow_state_REAL']['preview']['options']['solvent_volume_mL'],30.)
                app.button(key='chat_flow_REAL_open').click().run()
                self.assertFalse(app.exception,[e.value for e in app.exception])
                self.assertEqual(app.session_state['workflow_REAL_graph_scope'],'Chat preview')
                app.button(key='workflow_REAL_chat_preview_save').click().run()
                self.assertFalse(app.exception,[e.value for e in app.exception])
                self.assertEqual(app.session_state['workflow_REAL_graph_scope'],'Saved workflow')
                next(b for b in app.button if b.label=='Continue in conversation').click().run()
                self.assertFalse(app.exception,[e.value for e in app.exception])
                app.chat_input[0].set_value('/workflow edit {"pan_mass_g":4}').run()
                self.assertFalse(app.exception,[e.value for e in app.exception])
                self.assertEqual(app.session_state['chat_workflow_state_REAL']['preview']['options']['pan_mass_g'],4)
            self.assertEqual(service.store.all('sample'),before)
            self.assertEqual(len(service.db.all('definitions')),1);self.assertFalse(service.db.all('runs'))


if __name__=='__main__':unittest.main()
