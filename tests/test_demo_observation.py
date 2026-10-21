from sacfactory.bootstrap import load_demo_seed
"""Physical demo consistency, reported endpoints and protected forecast inputs."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
from demo_campaign import prediction_features
from sacfactory.chemical_descriptors import reference
from sacfactory.demo_physics import mass_balance
from sacfactory.demo_observation import aggregation_parameters,apply_batch_retention,observe_assays
from sacfactory.domain import screen,recipe_key


class DemoObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content=load_demo_seed(ROOT)
        cls.rows=cls.content['samples']

    def test_preserved_design_and_all_result_states_match_generated_observations(self):
        design=json.loads((ROOT/'examples/demo_campaign_design.json').read_text(encoding='utf-8'))['rows']
        self.assertEqual(len(self.rows),len(design))
        for s,d in zip(self.rows,design):
            for key in ('sample_id','batch_id','elements','concentrations_mM','pulse_count','pulse_limit_C','temperature_C','voltage_V','current_A','precursors'):
                self.assertEqual(s[key],d[key])
            self.assertEqual(s['demo_mass_balance']['used_temperature_C'],s['temperature_C'])
            self.assertFalse(screen(s,{})['review_conflicts'])
            obs=s['demo_observation']
            if obs['workflow_case']!='RECOVERED':
                self.assertEqual(s['icp'],[])
                continue
            expected='PASS' if all(a['detected'] for a in obs['assays'].values()) else 'FAIL'
            self.assertEqual(s['icp_review']['state'],expected)
            latent=obs['screening_latent']
            self.assertEqual(s['xrd']['state'],'FAIL' if latent['simulated_xrd_draw']<latent['xrd_fail_probability'] else 'PASS')
            for a in s['icp']:
                detail=obs['assays'][a['element']]
                if a['flag']=='MEASURED':
                    self.assertGreaterEqual(float(a['value']),detail['loq_wt_percent']-1e-8)
                else:
                    self.assertTrue(a['value'].startswith('<'))
                    self.assertLess(detail['simulated_uncensored_reading_wt_percent'],detail['loq_wt_percent'])

    def test_aggregation_uses_retained_atoms_and_carrier_not_input_id(self):
        anchor=mass_balance({'Fe':120},1000,1)
        result=aggregation_parameters(anchor,1000,1)
        self.assertAlmostEqual(result['relative_site_burden'],1.)
        decreased=copy.deepcopy(anchor);apply_batch_retention(decreased,.5)
        reduced=aggregation_parameters(decreased,1000,1)
        self.assertLess(reduced['aggregation_probability'],result['aggregation_probability'])
        self.assertLess(reduced['retained_effective_mmol'],result['retained_effective_mmol'])

    def test_detection_and_quantification_are_separate_and_not_zero_imputed(self):
        balance=mass_balance({'Fe':50},1000,1)
        settings={'simulated_icp_lod_wt_percent':.0015,'simulated_icp_loq_wt_percent':.005,'simulated_icp_log_sd':0.}
        balance['elements']['Fe']['true_w_wt_percent']=.003
        rows,state,_=observe_assays(balance,np.random.default_rng(1),settings)
        self.assertEqual(state,'PASS');self.assertEqual(rows[0]['flag'],'BLQ')
        self.assertEqual(rows[0]['value'],'<0.005')
        balance['elements']['Fe']['true_w_wt_percent']=.0005
        rows,state,_=observe_assays(balance,np.random.default_rng(1),settings)
        self.assertEqual(state,'FAIL');self.assertEqual(rows[0]['flag'],'ND')
        self.assertEqual(rows[0]['value'],'<0.0015')

    def test_future_features_exclude_measured_and_latent_outcomes(self):
        atoms={r['symbol']:r for r in reference()['elements']}
        s=self.rows[0];changed=copy.deepcopy(s)
        changed.update(temperature_C=9999.,voltage_V=9999.,current_A=9999.,icp=[],xrd={'state':'FAIL'},demo_mass_balance={},demo_observation={})
        np.testing.assert_array_equal(prediction_features(s,atoms),prediction_features(changed,atoms))

    def test_all_captured_forecasts_use_strictly_earlier_other_stages(self):
        mapping={s['sample_id']:s for s in self.rows}
        for s in self.rows:
            if 'proposal' not in s:
                continue
            p=s['proposal'];capture=p['evaluation_capture']
            self.assertTrue(0<=p['p_screen_pass']<=1)
            self.assertLessEqual(len(capture['training_records']),144)
            for row in capture['training_records']:
                source=mapping[row['sample_id']]
                self.assertLess(source['demo_sequence'],s['demo_sequence'])
                self.assertNotEqual(tuple(source['elements']),tuple(s['elements']))
                self.assertLess(source['result_reported_at'],p['created'])

    def test_unknown_electronegativity_is_explicitly_masked_not_a_known_zero(self):
        atoms={r['symbol']:r for r in reference()['elements']}
        eu=next(s for s in self.rows if s['elements']==['Eu'])
        fe=next(s for s in self.rows if s['elements']==['Fe'])
        self.assertIsNone(atoms['Eu']['en_pauling'])
        self.assertEqual(prediction_features(eu,atoms)[7],1.)
        self.assertEqual(prediction_features(fe,atoms)[7],0.)


if __name__=='__main__':
    unittest.main()
