"""Template copies, shared discovery records and operator resource reservations."""
from contextlib import closing
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from sacfactory.storage import Store
from sacfactory.workflow_engine import validate_spec
from sacfactory.workflow_service import WorkflowService,local_workflow_message
from sacfactory.workflow_templates import TEMPLATES,build,catalog
from sacfactory.workflow_graph import snapshot
from sacfactory.device_catalog import profile_device
from test_pulseicp_bridge import specimens


class TemplateTests(unittest.TestCase):
    def service(self,folder,origin='DEMO',count=4):
        store=Store(folder,origin);rows=specimens(count,origin)
        for i,row in enumerate(rows):row['record_uid']='template_fixture_'+str(i)
        store.import_bundle(rows,[])
        return WorkflowService(store)

    def test_every_pattern_builds_in_its_declared_data_spaces(self):
        self.assertEqual(len(TEMPLATES),14)
        for template in TEMPLATES:
            for origin in template['origins']:
                options={'elements':['Fe','Co'],'artifact_ids':['registered_artifact']}
                spec=build(template['id'],origin,options,['sample_1'])
                validate_spec(spec)
                self.assertTrue(spec['steps'])
                if origin=='REAL':self.assertFalse(any(n['operation'].startswith('device.') for n in spec['steps']))
        self.assertEqual(len(catalog('REAL')),12)
        self.assertEqual(len(build('simulation_loop','DEMO')['steps']),16)

    def test_template_preview_has_no_writes_or_fits(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);before=service.store.all('sample')
            with patch('sacfactory.discovery.workflows.train_candidate') as fit:
                plan=service.template_plan('simulation_loop',{'register_resources':True})
                view=snapshot(service,draft_spec=plan['spec'],planned_devices=plan['resources_to_register'])
                fit.assert_not_called()
            self.assertEqual(len(view['source_spec']['steps']),16)
            self.assertTrue(all(math.isfinite(n['mind_x']) and math.isfinite(n['mind_y']) for n in view['nodes']))
            self.assertEqual(service.store.all('sample'),before)
            for kind in ['devices','jobs','runs','definitions']:self.assertFalse(service.db.all(kind))

    def test_copy_creates_resources_and_definition_without_running(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,'REAL');before=service.store.all('sample')
            result=service.instantiate_template('precursor_preparation',{'register_resources':True})
            self.assertEqual(result['steps'],4)
            self.assertEqual(len(service.db.all('devices')),4)
            self.assertTrue(all(d['mode']=='INVENTORY_ONLY' for d in service.db.all('devices')))
            self.assertEqual(service.store.all('sample'),before)
            self.assertEqual(service.db.all('jobs'),[]);self.assertEqual(service.db.all('runs'),[])
            first=service.db.get('definitions',result['workflow_id'])
            service.instantiate_template('precursor_preparation',{'register_resources':True,'name':'A second editable copy'})
            self.assertEqual(service.db.get('definitions',result['workflow_id']),first)

    def test_manual_resources_reserve_until_acknowledgement_and_cancel(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,'REAL')
            definition=service.instantiate_template('precursor_preparation',{'register_resources':True})
            first=service.start(definition['workflow_id']);second=service.start(definition['workflow_id'])
            service.db.run_ready(first['id'],service.execute)
            self.assertEqual(service.db.summary(first['id'])['state'],'WAITING')
            self.assertEqual(service.db.run_ready(second['id'],service.execute)['state'],'QUEUED')
            with closing(service.db.connect()) as db:self.assertEqual(db.execute('SELECT device_id FROM locks').fetchone()[0],'inventory_balance')
            service.db.acknowledge(first['id'],'weigh','Weighed according to the approved method.')
            self.assertEqual(service.db.run_ready(second['id'],service.execute)['state'],'WAITING')
            service.db.control(second['id'],'cancel')
            with closing(service.db.connect()) as db:self.assertFalse(db.execute('SELECT * FROM locks').fetchall())
            self.assertEqual(service.db.all('jobs'),[])

    def test_failed_manual_claim_releases_instrument(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,'REAL');service.devices.register(profile_device('balance'))
            spec={'name':'Invalid sample scope','steps':[{'id':'weigh','operation':'manual.checkpoint',
                'parameters':{'instruction':'Record weighing','device_id':'inventory_balance','capability':'prepare','sample_ids':['outside_selection']}}]}
            run=service.start(service.db.save(spec)['id']);service.db.run_ready(run['id'],service.execute)
            self.assertEqual(service.db.summary(run['id'])['state'],'FAILED')
            with closing(service.db.connect()) as db:self.assertFalse(db.execute('SELECT * FROM locks').fetchall())

    def test_operator_notes_never_generate_assays_or_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,'REAL');before=service.store.all('sample')
            definition=service.instantiate_template('electrochemical_performance',{'register_resources':True})
            run=service.start(definition['workflow_id'])
            for node in run['spec']['steps']:
                service.db.run_ready(run['id'],service.execute)
                self.assertEqual(service.db.summary(run['id'])['state'],'WAITING')
                service.db.acknowledge(run['id'],node['id'],'Approved protocol completed; measurement files retained separately.')
            self.assertEqual(service.db.summary(run['id'])['state'],'COMPLETED')
            self.assertEqual(service.db.run_ready(run['id'],service.execute)['state'],'COMPLETED')
            self.assertEqual(service.store.all('sample'),before)
            self.assertFalse(service.db.all('jobs'))

    def test_template_scope_binding_and_parameter_errors_have_no_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder,'REAL')
            for template,options in [('simulation_loop',{}),('data_benchmark',{'sample_ids':['other']}),
                                     ('multielement_bridge',{'elements':['Fe']}),('inverse_screening',{'bounds_mmol':{'Fe':[2.,1.]}}),
                                     ('precursor_preparation',{'bindings':{'balance':'invented'},'register_resources':True})]:
                with self.assertRaises(ValueError):service.instantiate_template(template,options)
            self.assertFalse(service.db.all('definitions'));self.assertFalse(service.db.all('devices'))

    def test_template_commands_and_capabilities_share_the_service(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            result=local_workflow_message('/workflow templates',service)
            self.assertEqual(len(result['templates']),14)
            saved=service.conversation('create_template',spec_json=json.dumps({'template_id':'inverse_screening','options':{}}))
            self.assertEqual(saved['template_id'],'inverse_screening')
            local=local_workflow_message('/workflow template precursor_preparation',service)
            self.assertEqual(local['steps'],4)
            self.assertFalse(service.db.all('runs'))
            self.assertEqual(len(service.capabilities()['templates']),14)

    def test_workflow_training_and_validation_enter_the_shared_history_once(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            definition=service.db.save({'name':'History','steps':[{'id':'memory','operation':'knowledge.search','parameters':{'query':'Fe'}}]})
            run=service.start(definition['id']);result={'name':'SAC-Gen-Set','model_id':'test_candidate'}
            with patch('sacfactory.discovery.workflows.train_candidate',return_value=result):
                for _ in range(2):service.execute('model.train',{'family':'SAC-Gen-Set','epochs':5},run,run['id']+':train')
            report={'metrics':[],'folds':[],'note':'Protected test fixture'}
            with patch('sacfactory.discovery.validation.validate',return_value=report):
                for _ in range(2):service.execute('validation.run',{},run,run['id']+':validate')
            self.assertEqual(len(service.ledger.records('training')),1)
            self.assertEqual(len(service.ledger.records('validation')),1)
            self.assertEqual(service.ledger.records('training')[0]['workflow_run_id'],run['id'])

    def test_unified_workspace_renders_lazily_and_creates_a_template_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            app=AppTest.from_string('''
import streamlit as st
from sacfactory.storage import Store
from sacfactory.workflow_service import WorkflowService
from sacfactory.ui_workflows import render_workflows
store=Store(st.session_state['root'],'DEMO');rows=store.all('sample')
render_workflows(store,rows,WorkflowService(store).context,{},unified=True)
''',default_timeout=45)
            app.session_state['root']=folder
            with patch('sacfactory.ui_discovery.train_candidate') as train,patch('sacfactory.ui_discovery.cached_model') as load:
                for tab in ['Live board','Workflow library','Scientific tools','Runs','Devices']:
                    app.session_state['workflow_DEMO_tabs']=tab;app.run()
                    self.assertFalse(app.exception,[e.value for e in app.exception])
                train.assert_not_called();load.assert_not_called()
                app.session_state['workflow_DEMO_tabs']='Workflow library';app.run()
                app.button(key='workflow_DEMO_template_simulation_loop_create').click().run()
                self.assertFalse(app.exception,[e.value for e in app.exception])
                self.assertEqual(app.session_state['workflow_DEMO_tabs'],'Live board')
            self.assertEqual(len(service.db.all('definitions')),1)
            self.assertEqual(service.db.all('jobs'),[])
            self.assertEqual(service.db.all('runs'),[])
            app.button(key='workflow_DEMO_board_start').click().run()
            self.assertFalse(app.exception,[e.value for e in app.exception])
            created_run=service.db.all('runs')[0]
            self.assertEqual(app.session_state['workflow_DEMO_board_run_selection'],created_run['id'])
            self.assertEqual(app.session_state['workflow_DEMO_graph_scope'],'Run')
            self.assertEqual(created_run['state'],'READY')

    def test_old_app_navigation_resolves_to_one_combined_entry(self):
        app_root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'SACFACTORY_DATA_DIR':folder}):
            app=AppTest.from_file(str(app_root/'app.py'),default_timeout=60)
            app.session_state['page']='Discovery lab';app.run()
            self.assertFalse(app.exception,[e.value for e in app.exception])
            nav=next(r for r in app.radio if r.label=='Workspace')
            self.assertIn('Discovery & Workflows',nav.options)
            self.assertNotIn('Discovery lab',nav.options);self.assertNotIn('Devices & Workflows',nav.options)
            self.assertEqual(app.session_state['workflow_REAL_tabs'],'Scientific tools')

    def test_import_validation_rejects_executable_or_cross_reference_definitions(self):
        for spec in [{'name':'Unsafe','steps':[{'id':'run','operation':'exec','parameters':{'code':'ignored'}}]},
                     {'name':'Forward','steps':[{'id':'run','operation':'manual.checkpoint','parameters':{'instruction':{'$ref':'later.text'}}}]}]:
            with self.assertRaises(ValueError):validate_spec(spec)


if __name__=='__main__':unittest.main()
