"""Approval, structured returns, equimolar feeds and cumulative learning."""
import copy
import tempfile
import unittest
from sacfactory.workflow_service import WorkflowService
from sacfactory.workflow_templates import build
from sacfactory.workflow_graph import snapshot
from sacfactory.workflow_returns import validate_return
from sacfactory.storage import Store
from sacfactory.discovery.inverse import candidates
from test_pulseicp_bridge import specimens


class ConfirmedWorkflowTests(unittest.TestCase):
    def test_characterization_refresh_preserves_conversation_but_drops_stale_results(self):
        from sacfactory.ui_navigation import refresh_data_epoch
        session={'seen_data_epoch_REAL':'old','research_samples_REAL':['one'],
            'research_permission_REAL':True,'chat_workflow_state_REAL':{'run_id':'run_one'},
            'weighted_result_REAL':'stale','api_permission_REAL':True,'weighted_result_DEMO':'other'}
        refresh_data_epoch(session,'REAL','new')
        self.assertTrue(session['research_permission_REAL']);self.assertEqual(session['research_samples_REAL'],['one'])
        self.assertEqual(session['chat_workflow_state_REAL']['run_id'],'run_one')
        self.assertNotIn('weighted_result_REAL',session);self.assertEqual(session['weighted_result_DEMO'],'other')

    def service(self,folder):
        store=Store(folder,'DEMO');rows=specimens(20,'DEMO')
        for i,s in enumerate(rows):s['record_uid']='confirmed_'+str(i)
        store.import_bundle(rows,[])
        return WorkflowService(store)

    def start_plan(self,service):
        result=service.instantiate_template('confirmed_synthesis_loop',{'register_resources':True,'elements':['Fe','Co','Ni','Cu','Zn'],
            'composition_mode':'equimolar','batch_size':3,'allow_extrapolation':True,'performance_protocol':'Approved protocol P1'})
        run=service.start(result['workflow_id']);service.db.run_ready(run['id'],service.execute)
        return run

    def test_confirmation_blocks_pending_and_device_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);before=service.store.all('sample');run=self.start_plan(service)
            self.assertEqual(service.db.summary(run['id'])['state'],'WAITING')
            self.assertEqual(service.store.all('sample'),before);self.assertFalse(service.db.all('jobs'))
            with self.assertRaises(ValueError):service.acknowledge(run['id'],'confirm_plan','Looks fine')
            with self.assertRaises(ValueError):service.acknowledge(run['id'],'confirm_plan','Approved',{'approved':True})
            service.acknowledge(run['id'],'confirm_plan','Approved',{'approved':True,'performance_protocol':'Protocol P1'})
            service.db.run_ready(run['id'],service.execute)
            self.assertEqual(len(service.store.all('sample')),len(before)+3);self.assertFalse(service.db.all('jobs'))
            design=next(n for n in snapshot(service,run_id=run['id'])['nodes'] if n.get('step_id')=='design')
            self.assertTrue(design['pipeline']);self.assertIn('CPU',design['compute']);self.assertIn('ranked recipes',design['result'])
            nodes=[n for n in snapshot(service,run_id=run['id'])['nodes'] if n.get('step_id')]
            for i,a in enumerate(nodes):
                for b in nodes[i+1:]:self.assertTrue(abs(a['x']-b['x'])>=196 or abs(a['y']-b['y'])>=74,(a['step_id'],b['step_id']))

    def test_source_edits_invalidate_review(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);run=self.start_plan(service)
            sample=copy.deepcopy(service.store.all('sample')[0]);sample['notes']='Edited during review'
            service.store.import_bundle([sample],[],overwrite=True)
            with self.assertRaisesRegex(ValueError,'Source records changed'):service.acknowledge(run['id'],'confirm_plan','Approved',{'approved':True,'performance_protocol':'P1'})

    def test_structured_returns_require_finite_values(self):
        checkpoint={'return_fields':['pan_mass_g','precursor_amounts_mmol']}
        for data in [{},{'pan_mass_g':-2,'precursor_amounts_mmol':{'Fe':1}},{'pan_mass_g':2,'precursor_amounts_mmol':{'Fe':float('nan')}}]:
            with self.assertRaises(ValueError):validate_return(checkpoint,data)
        self.assertEqual(validate_return(checkpoint,{'pan_mass_g':2,'precursor_amounts_mmol':{'Fe':1}})['pan_mass_g'],2)

    def test_equimolar_units_and_disjoint_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            rows=candidates(service.context,['Fe','Co'],{'Fe':[.1,.5],'Co':[.2,.8]},count=8,composition_mode='equimolar')
            for r in rows:
                self.assertEqual(r['concentrations_mM']['Fe'],r['concentrations_mM']['Co'])
                self.assertTrue(.2<=r['concentrations_mM']['Fe']*r['solvent_volume_mL']/1000<=.5)
            with self.assertRaises(ValueError):candidates(service.context,['Fe','Co'],{'Fe':[.1,.2],'Co':[.3,.4]},composition_mode='equimolar')

    def test_scoped_cumulative_pool_and_real_manual_equipment(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);run=self.start_plan(service)
            result=service.execute('records.training_pool',{'sample_ids':run['selected_ids'][:3],'include_history':True},run,'pool-test')
            self.assertEqual(result['source_count'],20);self.assertEqual(result['historical_count'],20)
            with self.assertRaises(ValueError):service.execute('records.training_pool',{'sample_ids':['outside'],'include_history':True},run,'bad-pool')
            spec=build('confirmed_synthesis_loop','REAL',{'elements':['Fe']})
            self.assertFalse(any(n['operation'].startswith('device.') for n in spec['steps']))
            self.assertIn('next_design',[n['id'] for n in spec['steps']])


if __name__=='__main__':unittest.main()
