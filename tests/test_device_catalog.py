"""Categorized inventory, independent acquisitions and modality-safe scheduling."""
import copy
import tempfile
import unittest

from streamlit.testing.v1 import AppTest
from sacfactory.device_catalog import CATALOG,CATEGORIES,profile_device
from sacfactory.storage import Store
from sacfactory.workflow_service import WorkflowService,simulation_template
from sacfactory.workflow_graph import snapshot
from sacfactory.workflow_schedule import bind_devices,resource_check
from test_pulseicp_bridge import specimens
from test_workflows import minimal


class EquipmentTests(unittest.TestCase):
    def service(self,folder):
        store=Store(folder,'DEMO');rows=specimens(1,'DEMO');rows[0]['record_uid']='equipment_fixture_0001';store.import_bundle(rows,[])
        service=WorkflowService(store);service.devices.defaults()
        return service

    def test_catalog_covers_all_categories_and_registers_as_inventory(self):
        self.assertGreaterEqual(len(CATALOG),100)
        self.assertEqual(len(CATALOG),len({p['id'] for p in CATALOG}))
        self.assertEqual(set(CATEGORIES),{p['category'] for p in CATALOG})
        with tempfile.TemporaryDirectory() as folder:
            service=WorkflowService(Store(folder,'DEMO'))
            for profile in CATALOG:
                registered=service.devices.register(profile_device(profile['id']))
                self.assertEqual(registered['mode'],'INVENTORY_ONLY')
                self.assertEqual(registered['connection_state'],'INVENTORY')
            self.assertEqual(len(service.db.all('devices')),len(CATALOG))
            self.assertFalse(service.db.all('jobs'))

    def test_defaults_and_template_use_independent_instruments(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            self.assertEqual({d['id'] for d in service.db.all('devices')},{'demo_heater','demo_icp','demo_xrd','demo_stem','demo_sem'})
            steps=[s for s in simulation_template()['steps'] if s['operation']=='device.characterize']
            self.assertEqual(len(steps),4)
            self.assertEqual({s['parameters']['modalities'][0] for s in steps},{'ICP','XRD','STEM','SEM'})
            self.assertEqual(len({s['parameters']['device_id'] for s in steps}),4)

    def test_independent_acquisitions_join_without_fabricating_other_modalities(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);samples=service.store.all('sample')
            run=service.start(service.db.save(minimal())['id'])
            synthesis=service.devices.submit('demo_heater','synthesize',samples,'synthesis')
            receipts=[]
            for modality in ['ICP','XRD','STEM','SEM']:
                receipt=service.devices.submit('demo_'+modality.lower(),'characterize',samples,modality,synthesis,[modality])
                receipts.append(receipt)
                self.assertEqual(receipt['modalities'],[modality])
                row=receipt['measurements'][0]
                self.assertEqual(bool(row['icp']),modality=='ICP')
                self.assertEqual({f['modality'] for f in row['files']},set() if modality=='ICP' else {modality})
            evidence=service._analyze({'acquisitions':receipts},run,'analysis')
            self.assertEqual(len(evidence['artifact_ids']),3)
            self.assertEqual(len(evidence['source_job_ids']),4)
            sample=service.store.all('sample')[0]
            self.assertTrue(sample['icp']);self.assertEqual(len(sample['simulation']['source_job_ids']),4)
            artifacts=service.ledger.records('characterization')
            self.assertEqual(len({a['metadata']['source_job_id'] for a in artifacts}),3)
            again=service._analyze({'acquisitions':receipts},run,'analysis')
            self.assertEqual(again['artifact_ids'],evidence['artifact_ids'])
            self.assertEqual(len(service.ledger.records('characterization')),3)

    def test_join_rejects_forged_and_duplicate_receipts_before_any_results_write(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);samples=service.store.all('sample')
            run=service.start(service.db.save(minimal())['id'])
            synthesis=service.devices.submit('demo_heater','synthesize',samples,'synthesis')
            receipt=service.devices.submit('demo_icp','characterize',samples,'icp',synthesis,['ICP'])
            forged=copy.deepcopy(receipt);forged['measurements'][0]['icp'][0]['value']='999'
            for bundle in [[receipt,receipt],[forged]]:
                with self.assertRaises(ValueError):service._analyze({'acquisitions':bundle},run,'bad')
            self.assertEqual(service.store.all('sample'),samples)
            self.assertEqual(service.ledger.records('characterization'),[])

    def test_modality_mismatch_cannot_bind_schedule_or_execute(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            node={'id':'xrd','operation':'device.characterize','parameters':{'device_id':'demo_icp','synthesis':{},'modalities':['XRD']}}
            self.assertFalse(resource_check(node,service.db.all('devices'),{},'DEMO')[0])
            with self.assertRaises(ValueError):bind_devices({'name':'XRD','steps':[node]},{'xrd':'demo_icp'},service.db.all('devices'))
            with self.assertRaises(ValueError):service.devices.submit('demo_icp','characterize',service.store.all('sample'),'wrong',{},['XRD'])

    def test_changed_raw_export_is_rejected_before_icp_feedback_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder);samples=service.store.all('sample')
            run=service.start(service.db.save(minimal())['id'])
            synthesis=service.devices.submit('demo_heater','synthesize',samples,'synthesis')
            icp=service.devices.submit('demo_icp','characterize',samples,'icp',synthesis,['ICP'])
            xrd=service.devices.submit('demo_xrd','characterize',samples,'xrd',synthesis,['XRD'])
            path=service.store.root/xrd['measurements'][0]['files'][0]['relative_path']
            path.write_bytes(b'changed export')
            with self.assertRaises(ValueError):service._analyze({'acquisitions':[icp,xrd]},run,'bad')
            self.assertEqual(service.store.all('sample'),samples)
            self.assertEqual(service.ledger.records('characterization'),[])

    def test_inventory_and_unsupported_simulators_cannot_execute(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            service.devices.register(profile_device('tube_furnace'))
            node={'id':'heat','operation':'device.synthesize','parameters':{'device_id':'inventory_tube_furnace','sample_ids':[]}}
            okay,_,reason=resource_check(node,service.db.all('devices'),{},'DEMO')
            self.assertFalse(okay);self.assertIn('inventory',reason)
            for profile in ['tube_furnace','potentiostat','xps']:
                with self.assertRaises(ValueError):service.devices.register(profile_device(profile,mode='SIMULATOR'))

    def test_catalog_graph_has_complete_collection_membership(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            view=snapshot(service,include_catalog=True)
            nodes=view['nodes'];groups=[n for n in nodes if n['kind']=='device_group']
            self.assertEqual({g['category'] for g in groups},set(CATEGORIES))
            self.assertEqual(len(nodes),len({n['id'] for n in nodes}))
            represented={n.get('equipment_type') for n in nodes}
            self.assertTrue({p['id'] for p in CATALOG}<=represented)
            common=next(g for g in groups if g['category']=='General equipment')
            self.assertFalse(common['default_expanded'])
            self.assertGreater(len(common['members']),20)
            self.assertTrue(all(any(e['source']==n['group_id'] and e['target']==n['id'] for e in view['edges']) for n in nodes if n.get('group_id')))

    def test_legacy_combined_acquisition_remains_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            service=self.service(folder)
            service.devices.register({'id':'legacy','name':'Legacy combined fixture','capabilities':['characterize'],'mode':'SIMULATOR'})
            samples=service.store.all('sample');synthesis=service.devices.submit('demo_heater','synthesize',samples,'s')
            receipt=service.devices.submit('legacy','characterize',samples,'legacy',synthesis)
            self.assertEqual(len(receipt['measurements'][0]['files']),3)
            run=service.start(service.db.save(minimal())['id'])
            self.assertEqual(len(service._analyze(receipt,run,'analysis')['artifact_ids']),3)

    def test_directory_renders_all_profiles_and_filters_without_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            app=AppTest.from_string('''
import streamlit as st
from sacfactory.storage import Store
from sacfactory.workflow_service import WorkflowService
from sacfactory.ui_devices import render_devices
service=WorkflowService(Store(st.session_state['root'],'DEMO'))
render_devices(service,'catalog_test')
''',default_timeout=30)
            app.session_state['root']=folder;app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe[0].value),len(CATALOG))
            app.text_input(key='catalog_test_search').set_value('ICP').run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe[0].value),2)
            self.assertFalse(WorkflowService(Store(folder,'DEMO')).db.all('jobs'))


if __name__=='__main__':unittest.main()
