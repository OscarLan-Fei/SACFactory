"""Live graph semantics, populated-run widgets and real scheduling decisions."""
from contextlib import closing
import copy
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from sacfactory.storage import Store
from sacfactory.workflow_service import WorkflowService
from sacfactory.workflow_graph import snapshot
from sacfactory.workflow_schedule import bind_devices,schedule_rows
from sacfactory.workflow_engine import validate_spec
from sacfactory.workflow_worker import launch,worker_status
from test_workflows import minimal
from test_pulseicp_bridge import specimens


class WorkflowBoardTests(unittest.TestCase):
    def service(self,folder):return WorkflowService(Store(folder,'DEMO'))

    def test_populated_runs_render_without_duplicate_widget_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);run=service.start(service.db.save(minimal())['id'])
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
                for tab in ['Runs','Live board','Devices','Workflow library']:
                    app.session_state['workflow_DEMO_tabs']=tab;app.run()
                    self.assertEqual(len(app.exception),0,[e.value for e in app.exception])
                    keys=[w.key for widgets in [app.button,app.selectbox,app.radio,app.checkbox] for w in widgets if w.key]
                    self.assertEqual(len(keys),len(set(keys)))
                execute.assert_not_called()

    def test_all_sources_do_not_render_thousands_of_selected_tags(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder,'DEMO');store.import_bundle(specimens(80,'DEMO'),[])
            app=AppTest.from_string('''
import streamlit as st
from sacfactory.storage import Store
from sacfactory.ui_workflows import render_workflows
store=Store(st.session_state['root'],'DEMO');rows=store.all('sample')
render_workflows(store,rows,rows[0])
''',default_timeout=30)
            app.session_state['root']=folder;app.session_state['workflow_DEMO_tabs']='Devices';app.run()
            self.assertEqual(len(app.exception),0)
            self.assertFalse(any(w.label=='Source experiment records' for w in app.multiselect))
            self.assertTrue(any('80 selected source records' in c.value for c in app.caption))

    def test_graph_retains_dependency_and_actual_device_edges(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);service.devices.defaults()
            spec={'name':'Graph','steps':[{'id':'a','operation':'knowledge.search','parameters':{'query':'Fe'}},
                {'id':'heater','operation':'device.synthesize','parameters':{'device_id':'demo_heater','sample_ids':[]},'depends_on':['a']}]}
            run=service.start(service.db.save(spec)['id']);view=snapshot(service,run_id=run['id'])
            self.assertIn({'source':'step:a','target':'step:heater','kind':'dependency','label':'depends on'},view['edges'])
            self.assertTrue(any(e['source']=='device:demo_heater' and e['target']=='step:heater' and e['kind']=='uses' for e in view['edges']))
            self.assertEqual(next(n for n in view['nodes'] if n['id']=='step:a')['state'],'READY')
            self.assertEqual(next(n for n in view['nodes'] if n['id']=='step:heater')['state'],'WAITING_DEPENDENCIES')
            for index,left in enumerate(view['nodes']):
                for right in view['nodes'][index+1:]:
                    self.assertFalse(abs(left['mind_x']-right['mind_x'])<182 and abs(left['mind_y']-right['mind_y'])<68)

    def test_ready_priority_changes_the_executed_step(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            spec={'name':'Priorities','steps':[{'id':'low','operation':'knowledge.search','parameters':{'query':'Fe'},'schedule':{'priority':10,'estimated_seconds':20}},
                {'id':'high','operation':'knowledge.search','parameters':{'query':'Co'},'schedule':{'priority':90,'estimated_seconds':10}}]}
            run=service.start(service.db.save(spec)['id']);calls=[]
            service.db.advance(run['id'],lambda op,p,r,k:(calls.append(k) or {}))
            self.assertTrue(calls[0].endswith(':high'))

    def test_critical_path_breaks_equal_priority_ties(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            spec={'name':'Paths','steps':[{'id':'short','operation':'knowledge.search','parameters':{'query':'Fe'},'schedule':{'estimated_seconds':10}},
                {'id':'long','operation':'knowledge.search','parameters':{'query':'Co'},'schedule':{'estimated_seconds':30}},
                {'id':'tail','operation':'knowledge.search','parameters':{'query':'Ni'},'depends_on':['long'],'schedule':{'estimated_seconds':100}}]}
            run=service.start(service.db.save(spec)['id']);calls=[]
            service.db.advance(run['id'],lambda op,p,r,k:(calls.append(k) or {}));self.assertTrue(calls[0].endswith(':long'))

    def test_busy_device_queues_without_claiming_or_failing_task(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);service.devices.defaults()
            run=service.start(service.db.save(minimal('device.synthesize',{'device_id':'demo_heater','sample_ids':[]}))['id'])
            with closing(service.db.connect()) as db,db:db.execute('INSERT INTO locks VALUES(?,?)',('demo_heater','external_job'))
            executor=unittest.mock.Mock(return_value={});service.db.run_ready(run['id'],executor)
            self.assertEqual(service.db.summary(run['id'])['state'],'QUEUED');executor.assert_not_called()
            view=snapshot(service,run_id=run['id']);self.assertEqual(view['schedule'][0]['state'],'DEVICE_BUSY')
            self.assertIsNone(view['schedule'][0]['estimated_start_seconds'])
            with closing(service.db.connect()) as db,db:db.execute('DELETE FROM locks')
            service.db.run_ready(run['id'],executor);self.assertEqual(service.db.summary(run['id'])['state'],'COMPLETED')

    def test_unconnected_device_is_blocked_and_independent_work_can_proceed(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            service.devices.register({'id':'future','name':'Future instrument','mode':'MCP_NOT_CONNECTED','capabilities':['synthesize']})
            spec={'name':'Mixed','steps':[{'id':'blocked','operation':'device.synthesize','parameters':{'device_id':'future','sample_ids':[]}},
                 {'id':'local','operation':'knowledge.search','parameters':{'query':'mass'}}]}
            run=service.start(service.db.save(spec)['id']);calls=[]
            service.db.run_ready(run['id'],lambda op,p,r,k:(calls.append(op) or {}))
            self.assertEqual(calls,['knowledge.search']);self.assertEqual(service.db.summary(run['id'])['state'],'QUEUED')

    def test_binding_creates_a_revision_and_preserves_frozen_run(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);service.devices.defaults()
            service.devices.register({'id':'other_heater','name':'Second heater','mode':'SIMULATOR','capabilities':['synthesize']})
            original=minimal('device.synthesize',{'device_id':'demo_heater','sample_ids':[]})
            definition=service.db.save(original);run=service.start(definition['id'])
            revised=bind_devices(original,{'first':'other_heater'},service.db.all('devices'),{'first':{'priority':80,'estimated_seconds':30}})
            self.assertEqual(revised['steps'][0]['parameters']['device_id'],'other_heater')
            self.assertEqual(service.db.get('runs',run['id'])['spec']['steps'][0]['parameters']['device_id'],'demo_heater')
            view=snapshot(service,run_id=run['id'],bindings={'first':'other_heater'})
            self.assertTrue(any(e['kind']=='proposed' for e in view['edges']))
            with self.assertRaises(ValueError):bind_devices(original,{'first':'demo_characterizer'},service.db.all('devices'))

    def test_duration_is_an_estimate_and_dependency_end_precedes_start(self):
        spec={'name':'Timing','steps':[{'id':'a','operation':'knowledge.search','parameters':{'query':'Fe'},'schedule':{'estimated_seconds':10}},
              {'id':'b','operation':'knowledge.search','parameters':{'query':'Co'},'depends_on':['a'],'schedule':{'estimated_seconds':20}}]}
        states=[{'id':n['id'],'state':'PENDING','output':None} for n in spec['steps']]
        rows=schedule_rows(spec,states,[],{},'DEMO')
        self.assertEqual(rows[1]['estimated_start_seconds'],rows[0]['estimated_end_seconds'])
        self.assertIsNone(rows[0]['actual_seconds'])
        spec['steps'][0]['schedule']['estimated_seconds']=float('nan')
        with self.assertRaises(ValueError):validate_spec(spec)

    def test_background_worker_exposes_running_state_and_prevents_double_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);run=service.start(service.db.save(minimal())['id'])
            entered=threading.Event();release=threading.Event()
            def work(*args):entered.set();release.wait(10);return {}
            with patch.object(service,'execute',side_effect=work):
                launch(service,run['id']);self.assertTrue(entered.wait(10))
                try:
                    self.assertTrue(worker_status(service)['active'])
                    self.assertEqual(snapshot(service,run_id=run['id'])['schedule'][0]['state'],'RUNNING')
                    with self.assertRaises(ValueError):launch(service,run['id'])
                finally:release.set()
                deadline=time.monotonic()+10
                while worker_status(service)['active'] and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(service.db.summary(run['id'])['state'],'COMPLETED')

    def test_failed_task_does_not_mark_unexecuted_tasks_as_failed(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);spec=minimal()
            spec['steps'].append({'id':'later','operation':'knowledge.search','parameters':{'query':'Co'},'depends_on':['first']})
            run=service.start(service.db.save(spec)['id'])
            def fail(*args):raise ValueError('Injected task error')
            service.db.advance(run['id'],fail)
            rows={r['step_id']:r for r in snapshot(service,run_id=run['id'])['schedule']}
            self.assertEqual(rows['first']['state'],'FAILED');self.assertEqual(rows['later']['state'],'BLOCKED')
            self.assertIsNone(rows['later']['estimated_start_seconds'])

    def test_manual_run_pause_blocks_independent_ready_tasks_in_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            spec={'name':'Manual gate','steps':[{'id':'manual','operation':'manual.checkpoint','parameters':{'instruction':'Review evidence'},'schedule':{'priority':100}},
                {'id':'local','operation':'knowledge.search','parameters':{'query':'Fe'}}]}
            run=service.start(service.db.save(spec)['id']);service.db.advance(run['id'],service.execute)
            rows={r['step_id']:r for r in snapshot(service,run_id=run['id'])['schedule']}
            self.assertEqual(rows['manual']['state'],'WAITING');self.assertEqual(rows['local']['state'],'BLOCKED')


if __name__=='__main__':unittest.main()
