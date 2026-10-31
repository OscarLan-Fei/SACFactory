"""Empirical anchors, mass conservation, saturation and campaign ordering."""
import json
from pathlib import Path
import unittest

from sacfactory.demo_physics import mmol_to_mM,carrier_loss,mass_balance,dose_factor
from sacfactory.domain import TARGETS,screen
from sacfactory.thermal_loss import loss_settings


class DemoPhysicsTests(unittest.TestCase):
    def test_amount_conversions_and_fe_anchor(self):
        self.assertEqual(mmol_to_mM(1),50)
        self.assertEqual(mmol_to_mM(2.4),120)
        self.assertEqual(mmol_to_mM(1.8),90)
        balance=mass_balance({'Fe':50},1000,1)
        self.assertAlmostEqual(balance['carrier_before_pulse_g'],1.88)
        self.assertAlmostEqual(balance['carrier_after_pulse_g'],1.6356)
        self.assertAlmostEqual(balance['elements']['Fe']['true_w_wt_percent'],2.5)
        self.assertLess(balance['elements']['Fe']['retained_metal_g'],balance['elements']['Fe']['feed_metal_g'])

    def test_one_pulse_anchors_and_saturating_repeats(self):
        for temperature,expected in [(1000,.13),(1500,.15),(2000,.18)]:
            self.assertAlmostEqual(carrier_loss(temperature,1),expected)
            self.assertGreater(carrier_loss(temperature,10),expected)
            self.assertLess(carrier_loss(temperature,10)-carrier_loss(temperature,9),carrier_loss(temperature,2)-expected)
            self.assertLessEqual(carrier_loss(temperature,100),loss_settings()['carrier_loss_cap'])
        self.assertLess(carrier_loss(1200,16),carrier_loss(1600,16))

    def test_shared_denominator_and_light_element_retention_cap(self):
        balance=mass_balance({'Fe':50,'Au':50,'Mg':90},1500,4)
        masses=balance['elements']
        total=sum(r['retained_metal_g'] for r in masses.values())
        self.assertAlmostEqual(balance['recovered_sample_g'],balance['carrier_after_pulse_g']+total)
        for row in masses.values():
            self.assertAlmostEqual(row['feed_metal_g'],row['retained_metal_g']+row['lost_metal_g'])
            self.assertLessEqual(row['retained_metal_g'],row['feed_metal_g'])
        light=mass_balance({'Li':90},1000,1)['elements']['Li']
        self.assertLessEqual(light['retention_fraction'],1.)
        self.assertLess(light['true_w_wt_percent'],2.4)

    def test_new_seed_follows_single_element_first_and_every_mass_ledger(self):
        from sacfactory.bootstrap import load_demo_seed
        content=load_demo_seed(Path(__file__).resolve().parents[1])
        rows=content['samples']; self.assertEqual(len(rows),1476)
        self.assertGreater(content['unique_recipes'],1000)
        singles=[s for s in rows if len(s['elements'])==1]; mixtures=[s for s in rows if len(s['elements'])>1]
        self.assertEqual({s['elements'][0] for s in singles},set(TARGETS))
        self.assertLess(max(s['demo_sequence'] for s in singles),min(s['demo_sequence'] for s in mixtures))
        fe=[s for s in singles if s['elements']==['Fe']]
        self.assertEqual(len(fe),60)
        self.assertLess(max(s['demo_sequence'] for s in fe),min(s['demo_sequence'] for s in singles if s['elements']!=['Fe']))
        for e in TARGETS:
            self.assertTrue(any(screen(s,{})['label']==1 for s in singles if s['elements']==[e]))
        for s in rows:
            b=s['demo_mass_balance']
            self.assertAlmostEqual(b['recovered_sample_g'],b['carrier_after_pulse_g']+sum(r['retained_metal_g'] for r in b['elements'].values()))
            for e,r in b['elements'].items():
                self.assertLessEqual(r['retained_metal_g'],r['feed_metal_g']+1e-12)
                self.assertAlmostEqual(r['feed_metal_g'],r['retained_metal_g']+r['lost_metal_g'])
                if r['true_w_wt_percent'] is not None:
                    self.assertAlmostEqual(r['true_w_wt_percent'],100*r['retained_metal_g']/b['recovered_sample_g'])
