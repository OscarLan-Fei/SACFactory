"""Role provenance, reference handoffs, board/chat revisions and source permissions."""
from contextlib import closing
import json
import tempfile
import unittest
from sacfactory.storage import Store
from sacfactory.workflow_engine import OPERATIONS
from sacfactory.workflow_service import WorkflowService,simulation_template
from sacfactory.workflow_coordination import coordination,specialist_for,SPECIALISTS
from sacfactory.workflow_graph import snapshot
from sacfactory.workflow_chat import WorkflowChat,scoped_state
from sacfactory.ui_chat_workflows import apply_conversation_request
from test_pulseicp_bridge import specimens


class CoordinationTests(unittest.TestCase):
    def test_scientific_links_include_produced_records_and_select_the_matching_report(self):
        from unittest.mock import patch
        from sacfactory.ui_workflow_coordination import science_target
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder,'DEMO');rows=specimens(3,'DEMO');store.import_bundle(rows,[])
            service=WorkflowService(store,selected_ids=[rows[0]['sample_id']])
            definition=service.db.save({'name':'Sources','steps':[{'id':'memory','operation':'knowledge.search','parameters':{'query':'Fe'}}]})
            run=service.start(definition['id'])
            old=service.ledger.add('validation',{'origin':'DEMO','workflow_run_id':'another_run','result':{}})
            matching=service.ledger.add('validation',{'origin':'DEMO','workflow_run_id':run['id'],'result':{}})
            with patch.object(service.db,'steps',return_value=[{'operation':'campaign.plan','state':'COMPLETED','output':{'sample_ids':[rows[1]['sample_id']]}}]):
                target=science_target(service,'Validation',run['id'])
            self.assertEqual(target['selected_ids'],[rows[0]['sample_id'],rows[1]['sample_id']])
            self.assertEqual(target['report_id'],matching['id']);self.assertNotEqual(target['report_id'],old['id'])
            self.assertNotIn(rows[2]['sample_id'],target['selected_ids'])

    def test_every_operation_has_one_declared_specialist(self):
        self.assertEqual({specialist_for(op) for op in OPERATIONS},set(SPECIALISTS))
        with self.assertRaises(ValueError):specialist_for('shell.execute')

    def test_handoffs_require_completion_and_describe_referenced_fields(self):
        spec=simulation_template();steps=[dict(id=n['id'],state='PENDING',output=None) for n in spec['steps']]
        handoff=next(h for h in coordination(spec,steps)['handoffs'] if h['source_step']=='design' and h['target_step']=='pending')
        self.assertEqual(handoff['contract'],'design');self.assertEqual(handoff['state'],'WAITING')
        design=next(s for s in steps if s['id']=='design');design.update(state='RUNNING',output={'design':{}})
        self.assertEqual(next(h for h in coordination(spec,steps)['handoffs'] if h['source_step']=='design')['state'],'WAITING')
        design['state']='COMPLETED'
        self.assertEqual(next(h for h in coordination(spec,steps)['handoffs'] if h['source_step']=='design')['state'],'AVAILABLE')
        self.assertEqual(next(t for t in coordination(spec,steps)['tasks'] if t['step_id']=='pending')['state'],'PENDING')

    def test_operator_note_is_not_reported_as_an_assay(self):
        spec={'steps':[{'id':'review','operation':'manual.checkpoint','parameters':{'instruction':'Review'}}]}
        step={'id':'review','state':'COMPLETED','output':{'awaiting_manual_action':True}}
        task=coordination(spec,[step])['tasks'][0]
        self.assertIn('measurements remain separate',task['summary']);self.assertIn('completion recorded',task['summary'])

    def test_execution_records_role_once_and_observation_is_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            service=WorkflowService(Store(folder,'DEMO'))
            definition=service.db.save({'name':'Read evidence','steps':[{'id':'memory','operation':'knowledge.search','parameters':{'query':'mass balance'}}]})
            run=service.start(definition['id']);service.db.run_ready(run['id'],service.execute)
            self.assertEqual(service.db.get('receipts',run['id']+':memory')['specialist_id'],'knowledge')
            with closing(service.db.connect()) as db:
                before=db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
                count=db.execute("SELECT COUNT(*) FROM events WHERE json_extract(payload,'$.action')='SPECIALIST_RESULT'").fetchone()[0]
            self.assertEqual(count,1)
            self.assertEqual(service.conversation('status',run_id=run['id'])['coordination']['specialists'][0]['state'],'COMPLETED')
            self.assertEqual(snapshot(service,run_id=run['id'])['coordination']['recorded_results'],1)
            with closing(service.db.connect()) as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM events').fetchone()[0],before)

    def test_board_revision_retains_template_bindings_schedule_and_chat_editing(self):
        with tempfile.TemporaryDirectory() as folder:
            service=WorkflowService(Store(folder,'REAL'))
            original=service.instantiate_template('precursor_preparation',{'register_resources':True,'pan_mass_g':3})
            run=service.start(original['workflow_id'])
            from sacfactory.device_catalog import profile_device
            service.devices.register(profile_device('liquid_handler','backup'))
            revised=service.revise_board_workflow(original['workflow_id'],{'dispense':'backup'},{'weigh':{'priority':80,'estimated_seconds':90}})
            self.assertEqual(revised['template']['options']['bindings']['dispenser'],'backup')
            state=scoped_state(None,'REAL',service.selected_ids);controller=WorkflowChat(service,state)
            controller.template('inspect',workflow_id=revised['id'])
            edited=controller.template('revise',options_json=json.dumps({'solvent_volume_mL':40}))
            saved=service.db.get('definitions',edited['workflow_id'])
            self.assertEqual(next(n for n in saved['spec']['steps'] if n['id']=='weigh')['schedule']['priority'],80)
            self.assertEqual(next(n for n in saved['spec']['steps'] if n['id']=='dispense')['parameters']['device_id'],'backup')
            self.assertEqual(service.db.get('runs',run['id'])['context']['pan_mass_g'],3)
            self.assertEqual(service.db.get('runs',run['id'])['context']['solvent_volume_mL'],20)
            self.assertEqual(saved['template']['options']['solvent_volume_mL'],40)
            with self.assertRaises(ValueError):service.revise_board_workflow(saved['id'],{'dispense':'inventory_balance'})
            with self.assertRaises(ValueError):service.template_plan('precursor_preparation',{'scheduling':{'unknown':{'priority':50}}})

    def test_board_to_chat_never_widens_record_permissions(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder,'REAL');store.import_bundle(specimens(4,'REAL'),[])
            service=WorkflowService(store);definition=service.instantiate_template('data_benchmark')
            run=service.start(definition['workflow_id'])
            request={'origin':'REAL','workflow_id':definition['workflow_id'],'run_id':run['id']}
            narrow=WorkflowService(store,selected_ids=service.selected_ids[:1]);state=scoped_state(None,'REAL',narrow.selected_ids)
            session={'workflow_conversation_request':request}
            with self.assertRaises(ValueError):apply_conversation_request(session,narrow,state)
            self.assertNotIn('run_id',state);self.assertNotIn('workflow_conversation_request',session)
            session={'workflow_conversation_request':request};state=scoped_state(None,'REAL',service.selected_ids)
            self.assertTrue(apply_conversation_request(session,service,state));self.assertEqual(state['run_id'],run['id'])


if __name__=='__main__':unittest.main()
