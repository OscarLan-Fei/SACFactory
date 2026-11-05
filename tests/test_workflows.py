"""Workflow persistence, scope, device contracts and complete simulated feedback."""
import copy
from contextlib import closing
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch,Mock

from streamlit.testing.v1 import AppTest
from sacfactory.storage import Store
from sacfactory.workflow_engine import WorkflowDB,validate_spec
from sacfactory.workflow_service import WorkflowService,simulation_template,local_workflow_message
from sacfactory.workflow_devices import SimulatedDeviceFailure
from sacfactory.discovery.common import outcomes,digest
from test_pulseicp_bridge import specimens


def minimal(operation='knowledge.search',params=None):
    return {'name':'Small workflow','steps':[{'id':'first','operation':operation,'parameters':params or {'query':'mass balance'}}]}


class WorkflowTests(unittest.TestCase):
    def service(self,folder,origin='DEMO',count=0):
        store=Store(folder,origin)
        if count:store.import_bundle(specimens(count,origin),[])
        return WorkflowService(store)

    def test_definition_rejects_code_cycles_forward_and_undeclared_references(self):
        validate_spec(simulation_template())
        for op in ['exec','shell','sql','mcp.call_anything']:
            with self.assertRaises(ValueError):validate_spec(minimal(op))
        spec=minimal();spec['steps'][0]['depends_on']=['first']
        with self.assertRaises(ValueError):validate_spec(spec)
        spec=minimal();spec['steps'].append({'id':'second','operation':'knowledge.search','parameters':{'query':{'$ref':'first.evidence.0.title'}}})
        with self.assertRaises(ValueError):validate_spec(spec)
        spec['steps'][1]['depends_on']=['first'];validate_spec(spec)
        with self.assertRaises(ValueError):validate_spec(minimal('inverse.design',{'allow_extrapolation':'false'}))
        with self.assertRaises(ValueError):validate_spec(minimal('model.train',{'epochs':'100'}))

    def test_frozen_definition_and_completed_outputs_survive_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);spec=minimal();definition=service.db.save(spec);run=service.start(definition['id'])
            spec['steps'][0]['operation']='exec'
            service.db.run_ready(run['id'],service.execute)
            other=self.service(folder)
            self.assertEqual(other.db.summary(run['id'])['state'],'COMPLETED')
            self.assertEqual(other.db.get('runs',run['id'])['spec']['steps'][0]['operation'],'knowledge.search')
            self.assertTrue(other.db.steps(run['id'])[0]['output']['evidence'])

    def test_pause_resume_and_cancellation_stop_following_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);definition=service.db.save(minimal());run=service.start(definition['id'])
            service.db.control(run['id'],'pause')
            with self.assertRaises(ValueError):service.db.advance(run['id'],service.execute)
            service.db.control(run['id'],'resume');service.db.control(run['id'],'cancel')
            self.assertEqual(service.db.run_ready(run['id'],service.execute)['state'],'CANCELLED')
            self.assertEqual(service.db.steps(run['id'])[0]['attempts'],0)

    def test_manual_checkpoint_requires_documented_acknowledgement(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);definition=service.db.save(minimal('manual.checkpoint',{'instruction':'Transfer the sample.'}));run=service.start(definition['id'])
            service.db.run_ready(run['id'],service.execute)
            self.assertEqual(service.db.summary(run['id'])['state'],'WAITING')
            with self.assertRaises(ValueError):service.db.acknowledge(run['id'],'first','')
            service.db.acknowledge(run['id'],'first','Sample transfer recorded.')
            self.assertEqual(service.db.run_ready(run['id'],service.execute)['state'],'COMPLETED')

    def test_concurrent_worker_cannot_duplicate_a_running_action(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);run=service.start(service.db.save(minimal())['id'])
            claimed=threading.Event();release=threading.Event();calls=[]
            def execute(*args):calls.append(1);claimed.set();release.wait(10);return {'ok':True}
            thread=threading.Thread(target=lambda:service.db.advance(run['id'],execute));thread.start()
            self.assertTrue(claimed.wait(10))
            try:
                with self.assertRaises(ValueError):service.db.advance(run['id'],execute)
            finally:release.set();thread.join(10)
            self.assertEqual(len(calls),1);self.assertEqual(service.db.summary(run['id'])['state'],'COMPLETED')

    def test_expired_worker_requires_explicit_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);run=service.start(service.db.save(minimal())['id'])
            with closing(service.db.connect()) as db,db:
                step=service.db.steps(run['id'])[0];step['lease_until']='2000-01-01T00:00:00+00:00'
                db.execute("UPDATE steps SET state='RUNNING',payload=? WHERE run_id=?",(json.dumps(step),run['id']))
            with self.assertRaises(ValueError):service.db.advance(run['id'],service.execute)
            service.db.control(run['id'],'recover');service.db.control(run['id'],'retry')
            self.assertEqual(service.db.run_ready(run['id'],service.execute)['state'],'COMPLETED')

    def test_real_space_cannot_receive_simulated_acquisition(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,'REAL',1);service.devices.defaults();before=service.store.all('sample')
            with self.assertRaises(ValueError):service.devices.submit('demo_heater','synthesize',before,'test-key')
            with self.assertRaises(ValueError):service.conversation('create_demo')
            self.assertEqual(before,service.store.all('sample'));self.assertEqual(service.db.all('jobs'),[])

    def test_device_job_is_idempotent_and_limits_are_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,count=1);service.devices.defaults();samples=service.store.all('sample')
            a=service.devices.submit('demo_heater','synthesize',samples,'key');b=service.devices.submit('demo_heater','synthesize',samples,'key')
            self.assertEqual(a,b);self.assertEqual(len(service.db.all('jobs')),1)
            samples[0]['pulse_limit_C']=3000
            with self.assertRaises(ValueError):service.devices.submit('demo_heater','synthesize',samples,'new-key')

    def test_injected_device_failure_retries_same_job_without_duplicate_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,count=1)
            service.devices.register({'id':'faulty','name':'Fault test','mode':'SIMULATOR','capabilities':['synthesize'],'fail_once':True})
            spec=minimal('device.synthesize',{'device_id':'faulty','sample_ids':[service.selected_ids[0]]})
            run=service.start(service.db.save(spec)['id']);service.db.run_ready(run['id'],service.execute)
            self.assertEqual(service.db.summary(run['id'])['state'],'FAILED')
            first=service.db.all('jobs')[0]['id'];service.db.control(run['id'],'retry');service.db.run_ready(run['id'],service.execute)
            self.assertEqual(service.db.summary(run['id'])['state'],'COMPLETED')
            jobs=service.db.all('jobs');self.assertEqual(len(jobs),1);self.assertEqual(jobs[0]['id'],first);self.assertEqual(jobs[0]['attempts'],2)

    def test_device_lock_and_disconnected_mcp_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,count=1);service.devices.defaults()
            with closing(service.db.connect()) as db,db:db.execute('INSERT INTO locks VALUES(?,?)',('demo_heater','another_job'))
            with self.assertRaises(ValueError):service.devices.submit('demo_heater','synthesize',service.store.all('sample'),'key')
            service.devices.register({'id':'future','name':'Future MCP','mode':'MCP_NOT_CONNECTED','capabilities':['synthesize'],'mcp_endpoint':'https://example.org/mcp'})
            with self.assertRaises(ValueError):service.devices.submit('future','synthesize',service.store.all('sample'),'key2')
            with self.assertRaises(ValueError):service.devices.register({'id':'bad','name':'Bad','mode':'MCP_NOT_CONNECTED','capabilities':['synthesize'],'mcp_endpoint':'https://secret:token@example.org/mcp'})

    def test_forged_synthesis_receipt_cannot_create_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,count=1);service.devices.defaults()
            forged={'simulated':True,'completed':True,'job_id':'invented'}
            with self.assertRaises(ValueError):service.devices.submit('demo_characterizer','characterize',service.store.all('sample'),'key',forged)

    def test_scoped_run_and_training_cannot_use_unselected_records(self):
        with tempfile.TemporaryDirectory() as folder:
            all_service=self.service(folder,count=2);selected=all_service.selected_ids[:1]
            service=WorkflowService(all_service.store,selected_ids=selected)
            run=service.start(service.db.save(minimal())['id'])
            with self.assertRaises(ValueError):service._samples(all_service.selected_ids,run)
            denied=WorkflowService(all_service.store,selected_ids=[])
            with self.assertRaises(ValueError):denied.conversation('status',run_id=run['id'])

    def test_fabricated_forecasts_cannot_be_frozen_as_model_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);run=service.start(service.db.save(minimal())['id'])
            with self.assertRaises(ValueError):service.execute('campaign.plan',{'design':{'rows':[],'model_id':'invented'}},run,'forged')
            self.assertEqual(service.store.all('sample'),[])

    def test_local_conversation_commands_use_persistent_service(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);result=local_workflow_message('/workflow demo',service)
            started=local_workflow_message('/workflow start '+result['workflow_id'],service)
            self.assertEqual(local_workflow_message('/workflow status '+started['run_id'],service)['state'],'READY')
            self.assertIsNone(local_workflow_message('Discuss chemistry',service))

    def test_incomplete_planning_context_does_not_block_workflow_management(self):
        with tempfile.TemporaryDirectory() as folder:
            context={'origin':'REAL','elements':['Fe'],'concentrations_mM':{'Fe':None},'pulse_limit_C':None,'pulse_count':1}
            service=WorkflowService(Store(folder,'REAL'),context,[])
            self.assertIsNotNone(service.capabilities()['context_warning'])
            definition=service.db.save(minimal());run=service.start(definition['id'])
            self.assertEqual(service.db.run_ready(run['id'],service.execute)['state'],'COMPLETED')

    def test_complete_simulated_loop_covers_all_seven_functions(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);service.devices.defaults();definition=service.db.save(simulation_template())
            run=service.start(definition['id']);service.db.run_ready(run['id'],service.execute)
            summary=service.db.summary(run['id']);self.assertEqual(summary['state'],'COMPLETED',summary)
            steps={s['id']:s for s in service.db.steps(run['id'])};samples=service.store.all('sample')
            self.assertEqual(len(samples),32);self.assertTrue(all(s['origin']=='DEMO' and s['simulation']['synthetic'] for s in samples))
            self.assertTrue(all(outcomes(s)[0] and outcomes(s)[1] is not None for s in samples))
            self.assertEqual(len(service.ledger.records('characterization')),96)
            self.assertEqual(len(service.ledger.records('characterization_assessment')),32)
            self.assertEqual(steps['feedback']['output']['evaluation'][0]['structural_specimens'],32)
            validation=steps['validation']['output'];self.assertEqual(len(validation['folds']),3)
            for fold in validation['folds']:
                pools=[set(fold[k]) for k in ['fit_keys','weight_keys','calibration_keys','test_keys']]
                self.assertTrue(all(not a&b for i,a in enumerate(pools) for b in pools[i+1:]))
            model_id=steps['update']['output']['candidate']['model_id']
            self.assertEqual(steps['next_design']['output']['model_id'],model_id)
            self.assertNotEqual(steps['design']['output']['model_id'],model_id)
            self.assertEqual(len(service.ledger.records('hypothesis')),1)
            self.assertFalse((Path(folder)/'sac_gen/demo/latest.json').exists())
            # Receipt recovery must not create a second training directory or duplicate records.
            step=next(n for n in run['spec']['steps'] if n['id']=='update')
            from sacfactory.workflow_engine import resolve
            params=resolve(step['parameters'],{k:v['output'] for k,v in steps.items()})
            old=len(list((Path(folder)/'discovery_models/demo').glob('workflow_*')))
            service.execute('model.train',params,run,run['id']+':update')
            self.assertEqual(old,len(list((Path(folder)/'discovery_models/demo').glob('workflow_*'))))
            # Recover a crash after checkpoint publication but before the step receipt.
            with closing(service.db.connect()) as db,db:db.execute('DELETE FROM receipts WHERE id=?',(run['id']+':update',))
            with patch('sacfactory.discovery.set_model.SetResidual.fit',side_effect=AssertionError('Must load the completed checkpoint')):
                recovered=service.execute('model.train',params,run,run['id']+':update')
                self.assertEqual(recovered['candidate']['model_id'],model_id)

    def test_expired_simulator_lock_can_be_reconciled_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,count=1);service.devices.defaults()
            result=service.devices.submit('demo_heater','synthesize',service.store.all('sample'),'run_expired:step')
            with closing(service.db.connect()) as db,db:
                job=service.db.get('jobs',result['job_id']);job['state']='RUNNING';job['lease_until']='2000-01-01T00:00:00+00:00'
                db.execute('UPDATE jobs SET payload=? WHERE id=?',(json.dumps(job),job['id']))
                db.execute('INSERT INTO locks VALUES(?,?)',('demo_heater',job['id']))
            self.assertEqual(service.devices.reconcile_expired('run_expired'),[job['id']])
            retried=service.devices.submit('demo_heater','synthesize',service.store.all('sample'),'run_expired:step')
            self.assertEqual(retried['job_id'],job['id'])

    def test_workflow_interface_renders_without_running_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            app=AppTest.from_string('''
import streamlit as st
from sacfactory.storage import Store
from sacfactory.workflow_service import WorkflowService
from sacfactory.ui_workflows import render_workflows
store=Store(st.session_state['root'],'DEMO')
service=WorkflowService(store)
render_workflows(store,store.all('sample'),service.context)
''',default_timeout=30)
            app.session_state['root']=folder
            with patch('sacfactory.workflow_service.WorkflowService.execute') as execute:
                for tab in ['Workflow library','Runs','Devices']:
                    app.session_state['workflow_DEMO_tabs']=tab;app.run();self.assertEqual(len(app.exception),0)
                execute.assert_not_called()
            self.assertEqual(service.db.all('jobs'),[])


class ConversationWorkflowTests(unittest.TestCase):
    def test_llm_tools_can_save_and_start_a_workflow_without_arbitrary_code(self):
        from sacfactory.research_chat import ResearchTools,ResearchAdapter
        from sacfactory.language import ApiConfig
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            service=WorkflowService(Store(folder,'DEMO'))
            tools=ResearchTools([],{},SimpleNamespace(info={}),service.context,{},'DEMO',[],workflow_service=service)
            self.assertTrue(any(t['name']=='workflow_control' for t in tools.definitions))
            captured=[]
            def transport(request,timeout):
                body=json.loads(request.data);captured.append(body)
                if len(captured)==1:return {'status':'completed','output':[{'type':'function_call','call_id':'w1','name':'workflow_control',
                    'arguments':json.dumps({'action':'save','workflow_id':None,'run_id':None,'spec_json':json.dumps(minimal())})}]}
                if len(captured)==2:
                    saved=json.loads(body['input'][-1]['output'])
                    return {'status':'completed','output':[{'type':'function_call','call_id':'w2','name':'workflow_control',
                        'arguments':json.dumps({'action':'start','workflow_id':saved['workflow_id'],'run_id':None,'spec_json':None})}]}
                return {'status':'completed','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'Workflow saved and started.'}]}]}
            result=ResearchAdapter(ApiConfig('https://example.org/v1','mock','credential','responses'),True,transport=transport).converse('Save a research-memory workflow.',[],{},tools)
            self.assertEqual(len(service.db.all('definitions')),1);self.assertEqual(result.trace[0]['tool'],'workflow_control')
            self.assertEqual(len(service.db.all('runs')),1)
            self.assertTrue(any(t['name']=='workflow_control' for t in captured[0]['tools']))
            with self.assertRaises(ValueError):service.db.save(minimal('exec',{'code':'bad'}))


if __name__=='__main__':unittest.main()
