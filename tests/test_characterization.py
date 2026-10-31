"""Synthetic feature extraction, evidence review and feedback-loop contracts."""
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from sacfactory.discovery.characterization import (read_pattern,xrd_analysis,microscopy_analysis,
    register_artifact,review_artifact,evidence_summary,commit_assessment)
from sacfactory.discovery.campaigns import CampaignLedger
from sacfactory.discovery.common import identity,outcomes,query
from sacfactory.discovery.workflows import pending_from_proposal
from sacfactory.storage import Store
from test_discovery import forecast
from test_pulseicp_bridge import specimens


def pattern():
    x=np.linspace(10.,80.,1401)
    y=12+.04*x+100*np.exp(-.5*((x-30)/.15)**2)+70*np.exp(-.5*((x-55)/.25)**2)
    return ('two_theta_deg,intensity\n'+'\n'.join(f'{a:.6f},{b:.8f}' for a,b in zip(x,y))).encode()


def image_bytes():
    y,x=np.mgrid[:128,:128];image=np.full((128,128),100.)
    for a,b in [(32,32),(64,64),(90,95)]:image+=2000*np.exp(-((x-a)**2+(y-b)**2)/(2*1.5**2))
    buffer=io.BytesIO();Image.fromarray(image.astype('uint16')).save(buffer,format='TIFF')
    return buffer.getvalue()


class CharacterizationTests(unittest.TestCase):
    def setup_store(self,folder):
        store=Store(folder,'REAL');sample=specimens(1)[0]
        sample.update(record_uid='test-physical-record',xrd={'mode':'MANUAL','state':'PENDING'},stem={'state':'NOT_TESTED'},icp=[],result_reported_at=None)
        store.import_bundle([sample],[])
        return store,CampaignLedger(folder,'REAL'),store.all('sample')[0]

    def artifact(self,store,ledger,sample,modality='XRD'):
        if modality=='XRD':data=pattern();analysis,_=xrd_analysis(data);name='pattern.csv'
        else:data=image_bytes();analysis=microscopy_analysis(data,modality,nm_per_pixel=.02);name='image.tif'
        return register_artifact(store,ledger,sample['sample_id'],modality,name,data,analysis)

    def test_xrd_peaks_and_sourced_matches_remain_unreviewed(self):
        reference={'phase':'Synthetic phase','two_theta_deg':30.,'source':'Synthetic unit test','wavelength_angstrom':1.5406}
        result,_=xrd_analysis(pattern(),references=[reference],wavelength_angstrom=1.5406)
        self.assertTrue(result['reference_matches'][0]['matched'])
        self.assertTrue(any(abs(p['two_theta_deg']-55)<.2 for p in result['peaks']))
        self.assertEqual(result['suggested_state'],'PENDING');self.assertIsNone(result['probability'])
        with self.assertRaises(ValueError):xrd_analysis(pattern(),references=[{'phase':'Unknown','two_theta_deg':30,'source':''}])
        with self.assertRaises(ValueError):xrd_analysis(pattern(),references=[reference])
        with self.assertRaises(ValueError):xrd_analysis(pattern(),references=[reference],wavelength_angstrom=.71)

    def test_pattern_format_and_finite_values(self):
        self.assertEqual(read_pattern(pattern()).shape,(1401,2))
        with self.assertRaises(ValueError):read_pattern(b'10,1\n11,nan\n12,3')
        with self.assertRaises(ValueError):read_pattern(pattern()+b'\n81,10,3')
        flat=b'\n'.join(f'{i},1'.encode() for i in range(10,30))
        self.assertEqual(xrd_analysis(flat)[0]['peaks'],[])

    def test_stem_scale_roi_and_raw_spots_are_not_element_identities(self):
        result=microscopy_analysis(image_bytes(),'STEM',nm_per_pixel=.02,roi=[16,16,112,112])
        self.assertEqual(result['candidate_count'],3)
        self.assertEqual(result['sampled_area_nm2'],96*96*.02**2)
        self.assertTrue(all(r['nearest_neighbor_nm']>0 for r in result['features']))
        self.assertNotIn('element',result['features'][0]);self.assertIsNone(result['probability'])
        with self.assertRaises(ValueError):microscopy_analysis(image_bytes(),'STEM',neighbor_distance_nm=.3)
        with self.assertRaises(ValueError):microscopy_analysis(image_bytes(),'STEM',roi=[0,0,140,140])

    def test_sem_descriptive_sizes_need_scale(self):
        result=microscopy_analysis(image_bytes(),'SEM',nm_per_pixel=2.)
        self.assertGreater(result['candidate_count'],0)
        for row in result['features']:self.assertAlmostEqual(row['equivalent_diameter_nm'],2*row['equivalent_diameter_px'])
        unscaled=microscopy_analysis(image_bytes(),'SEM')
        self.assertIsNone(unscaled['features'][0]['equivalent_diameter_nm'])

    def test_upload_and_observation_do_not_change_experimental_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            store,ledger,sample=self.setup_store(folder);before=store.all('sample')
            artifact=self.artifact(store,ledger,sample)
            review_artifact(store,ledger,artifact['id'],'NO_RESOLVED_TARGET_CRYSTALLINE_PEAKS','Analyst')
            self.assertEqual(before,store.all('sample'))
            self.assertEqual(evidence_summary(sample,ledger)['recommended_screening_state'],'PASS')
            self.assertEqual(outcomes(sample),({},None))

    def test_sem_alone_cannot_resolve_structural_label(self):
        with tempfile.TemporaryDirectory() as folder:
            store,ledger,sample=self.setup_store(folder);artifact=self.artifact(store,ledger,sample,'SEM')
            review_artifact(store,ledger,artifact['id'],'AGGREGATES_OBSERVED','Analyst')
            with self.assertRaises(ValueError):commit_assessment(store,ledger,sample['sample_id'],'FAIL','Analyst','Visible objects')
            self.assertEqual(outcomes(store.get('sample',sample['sample_id'])),({},None))

    def test_source_tampering_is_detected_at_review_and_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            store,ledger,sample=self.setup_store(folder);artifact=self.artifact(store,ledger,sample)
            review_artifact(store,ledger,artifact['id'],'NO_RESOLVED_TARGET_CRYSTALLINE_PEAKS','Analyst')
            (Path(folder)/artifact['relative_path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):review_artifact(store,ledger,artifact['id'],'INCONCLUSIVE','Analyst')
            with self.assertRaises(ValueError):commit_assessment(store,ledger,sample['sample_id'],'PASS','Analyst','Reviewed screening')

    def test_changed_recipe_and_cross_namespace_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            store,ledger,sample=self.setup_store(folder);artifact=self.artifact(store,ledger,sample)
            sample['pulse_limit_C']+=10;store.import_bundle([sample],[],overwrite=True)
            with self.assertRaises(ValueError):review_artifact(store,ledger,artifact['id'],'INCONCLUSIVE','Analyst')
            with self.assertRaises(ValueError):evidence_summary(sample,CampaignLedger(folder,'DEMO'))

    def test_conflict_requires_explicit_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            store,ledger,sample=self.setup_store(folder)
            for modality,observation in [('XRD','TARGET_CRYSTALLINE_PHASE_DETECTED'),('STEM','ISOLATED_SITE_CANDIDATES_CONFIRMED')]:
                artifact=self.artifact(store,ledger,sample,modality)
                review_artifact(store,ledger,artifact['id'],observation,'Analyst')
            self.assertTrue(evidence_summary(sample,ledger)['conflict'])
            with self.assertRaises(ValueError):commit_assessment(store,ledger,sample['sample_id'],'PASS','Analyst','Needs interpretation')
            event=commit_assessment(store,ledger,sample['sample_id'],'FAIL','Analyst','Isolated sites coexist with crystalline aggregates; fail screening.',override=True)
            self.assertEqual(event['state'],'FAIL');self.assertEqual(outcomes(store.all('sample')[0])[1],0)

    def test_reviewed_feedback_closes_the_frozen_forecast_loop(self):
        with tempfile.TemporaryDirectory() as folder:
            store,ledger,sample=self.setup_store(folder);frozen=ledger.freeze(store.all('sample'),{'model_id':'test-model'})
            recipe=query(sample);recipe['pulse_limit_C']+=10
            proposal=ledger.propose(frozen['id'],[recipe],forecast([recipe]))
            ids=pending_from_proposal(store,ledger,proposal['id']);pending=store.get('sample',ids[0])
            artifact=self.artifact(store,ledger,pending)
            review_artifact(store,ledger,artifact['id'],'NO_RESOLVED_TARGET_CRYSTALLINE_PEAKS','Analyst')
            commit_assessment(store,ledger,ids[0],'PASS','Analyst','Reviewed target peaks absent within measured scan and sensitivity.')
            feedback=ledger.evaluate(store.all('sample'))[0]
            self.assertEqual(feedback['structural_specimens'],1)
            self.assertEqual(feedback['quantified_specimens'],0)
            self.assertAlmostEqual(feedback['structural_Brier'],.09)
            self.assertEqual(ledger.records('proposal')[0]['predictions'][0]['p_xrd_pass'],.7)
            self.assertNotIn('characterization_assessment_id',query(store.get('sample',ids[0])))

    def test_cross_database_failure_rolls_back_sample_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            store,ledger,sample=self.setup_store(folder);artifact=self.artifact(store,ledger,sample)
            review=review_artifact(store,ledger,artifact['id'],'NO_RESOLVED_TARGET_CRYSTALLINE_PEAKS','Analyst')
            before=store.all('sample')
            class Collision:hex=review['id']
            import sqlite3
            with patch('sacfactory.discovery.characterization.uuid4',return_value=Collision()):
                with self.assertRaises(sqlite3.IntegrityError):commit_assessment(store,ledger,sample['sample_id'],'PASS','Analyst','Reviewed')
            self.assertEqual(before,store.all('sample'));self.assertEqual(ledger.records('characterization_assessment'),[])

    def test_review_and_commit_interface(self):
        with tempfile.TemporaryDirectory() as folder:
            store,ledger,sample=self.setup_store(folder);self.artifact(store,ledger,sample)
            app=AppTest.from_string('''
import streamlit as st
from sacfactory.storage import Store
from sacfactory.discovery.campaigns import CampaignLedger
from sacfactory.ui_characterization import render_characterization
store=Store(st.session_state['root'],'REAL')
rows=store.all('sample')
render_characterization(store,rows,CampaignLedger(store.root,'REAL'),rows[0])
''',default_timeout=30)
            app.session_state['root']=folder;app.run()
            key='discovery_characterization_REAL'
            app.text_input(key=key+'_reviewer').set_value('Test analyst')
            app.selectbox(key=key+'_observation_XRD').set_value('NO_RESOLVED_TARGET_CRYSTALLINE_PEAKS')
            app.button(key=key+'_review').click().run();self.assertEqual(len(app.exception),0)
            app.selectbox(key=key+'_state').set_value('PASS')
            app.text_area(key=key+'_criteria').set_value('Reviewed XRD screening observation in synthetic test.')
            app.button(key=key+'_commit').click().run()
            self.assertEqual(len(app.exception),0);self.assertEqual(outcomes(store.all('sample')[0])[1],1)
            self.assertEqual(len(ledger.records('characterization_assessment')),1)

    def test_legacy_identity_binding_survives_display_id_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            store,ledger,sample=self.setup_store(folder)
            with store.connect() as db:
                sample.pop('record_uid');db.execute("UPDATE items SET payload=? WHERE kind='sample'",(__import__('json').dumps(sample),))
            artifact=self.artifact(store,ledger,sample)
            current=store.get('sample',sample['sample_id'])
            self.assertTrue(current['record_uid']);self.assertEqual(artifact['record_key'],current['record_uid'])
            self.assertEqual(ledger.records('identity_binding')[0]['previous_key'],sample['sample_id'])
            review_artifact(store,ledger,artifact['id'],'NO_RESOLVED_TARGET_CRYSTALLINE_PEAKS','Analyst')
            store.realign_sample_ids()
            self.assertEqual(evidence_summary(store.all('sample')[0],ledger)['recommended_screening_state'],'PASS')

    def test_committing_same_reviewed_assessment_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            store,ledger,sample=self.setup_store(folder);artifact=self.artifact(store,ledger,sample)
            review_artifact(store,ledger,artifact['id'],'NO_RESOLVED_TARGET_CRYSTALLINE_PEAKS','Analyst')
            first=commit_assessment(store,ledger,sample['sample_id'],'PASS','Analyst','Reviewed screening')
            second=commit_assessment(store,ledger,sample['sample_id'],'PASS','Analyst','Reviewed screening')
            self.assertEqual(first['id'],second['id']);self.assertEqual(len(ledger.records('characterization_assessment')),1)


if __name__=='__main__':unittest.main()
