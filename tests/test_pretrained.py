"""Inference releases exclude observations and work without a local database."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest
from sacfactory.compute_runtime import execution
from sacfactory.pretrained import available,bundle_root,load

RECIPE={'origin':'REAL','elements':['Fe'],'concentrations_mM':{'Fe':50.},
        'pulse_count':1,'pulse_limit_C':1000.,'pan_mass_g':2.,'solvent_volume_mL':20.}


class PretrainedTests(unittest.TestCase):
    def test_bundles_predict_without_observations_and_ignore_changed_results(self):
        self.assertEqual(set(available()),{'SAC-Gen-Diffusion','SAC-Gen-GB'})
        for name in available():
            with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
                model=load(name,'cpu')
                baseline=model.predict([RECIPE])[0]
                changed={**copy.deepcopy(RECIPE),'icp':[{'element':'Fe','value':99.}],
                         'xrd':{'state':'FAIL'},'stem':{'state':'NOT_CONFIRMED'}}
                updated=model.predict([changed])[0]
            self.assertAlmostEqual(baseline['predicted_total_wt_percent'],updated['predicted_total_wt_percent'])
            self.assertAlmostEqual(baseline['p_xrd_pass'],updated['p_xrd_pass'])
            self.assertTrue(0<=baseline['predicted_total_wt_percent']<100)
            self.assertTrue(0<=baseline['p_xrd_pass']<=1)
            self.assertTrue(baseline['outside_training_support'])
            self.assertEqual(model.training_samples,[])
            self.assertEqual(model.training_ids,[])
            with self.assertRaises(ValueError):model.fit([], 'REAL')
            with self.assertRaises(ValueError):model.predict([{**RECIPE,'origin':'DEMO'}])
            with self.assertRaises(ValueError):model.predict([{**RECIPE,'concentrations_mM':{'Fe':-1.}}])

    def test_no_experimental_snapshot_fields_or_pickle_assets(self):
        forbidden={'training_samples','training_ids','training_keys','history','bootstrap','inner_folds',
                   'sample_id','batch_id','record_uid','training_fingerprint'}
        def inspect(value):
            if isinstance(value,dict):
                self.assertFalse(set(value)&forbidden)
                for child in value.values():inspect(child)
            elif isinstance(value,list):
                for child in value:inspect(child)
        for path in bundle_root().glob('*.json'):inspect(json.loads(path.read_text()))
        for path in bundle_root().glob('*.npz'):
            with np.load(path,allow_pickle=False) as data:
                self.assertTrue(all(data[key].dtype.kind in 'fiu' for key in data.files))
        self.assertFalse(list(bundle_root().glob('*.joblib')))

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'registry.json').write_text(json.dumps({'models':{'SAC-Gen-GB':{'file':'bad.json','sha256':'bad'}}}))
            (root/'bad.json').write_text('{}')
            with patch('sacfactory.pretrained.bundle_root',return_value=root):
                with self.assertRaisesRegex(ValueError,'integrity'):load('SAC-Gen-GB')

    def test_discovery_lists_public_models_only_in_real(self):
        from sacfactory.discovery.workflows import available_models,load_model
        with tempfile.TemporaryDirectory() as folder:
            self.assertIn('Pretrained SAC-Gen-GB',available_models(folder,'REAL'))
            self.assertNotIn('Pretrained SAC-Gen-GB',available_models(folder,'DEMO'))
            self.assertTrue(load_model(folder,'REAL','Pretrained SAC-Gen-GB',[]).is_public_release)

    def test_predict_ui_works_with_zero_experimental_records(self):
        from sacfactory.storage import Store
        with tempfile.TemporaryDirectory() as folder:
            script='''
import streamlit as st
from sacfactory.storage import Store
from sacfactory.ui_discovery import render_discovery
store=Store(st.session_state['root'],'REAL')
render_discovery(store,[],{}, {'elements':['Fe']})
'''
            app=AppTest.from_string(script,default_timeout=30)
            app.session_state['root']=folder
            app.run()
            self.assertFalse(app.exception)
            app.selectbox(key='pretrained_variant').set_value('SAC-Gen-GB')
            app.button(key='pretrained_predict').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertGreater(app.session_state['pretrained_prediction'][1]['predicted_total_wt_percent'],0.)
            self.assertEqual(Store(folder,'REAL').all('sample'),[])


if __name__=='__main__':unittest.main()
