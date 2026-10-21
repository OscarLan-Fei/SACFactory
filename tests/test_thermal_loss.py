"""Thermal prior continuity, sourced missingness, monotonicity and mass bounds."""
import unittest
import numpy as np
from sacfactory.demo_physics import mass_balance
from sacfactory.domain import TARGETS
from sacfactory.thermal_loss import elemental_loss_reference,metal_loss_parameters,metal_retention,thermal_risk,loss_settings

ANCHOR = .750979703437399


class ThermalLossTests(unittest.TestCase):
    def test_fe_anchor_and_temperature_effect_have_no_dead_zone(self):
        self.assertAlmostEqual(metal_retention('Fe',1000,1,ANCHOR)[0],ANCHOR)
        temperatures=[1000,1000.001,1200,1600]
        retained=[metal_retention('Fe',t,1,ANCHOR)[0] for t in temperatures]
        self.assertTrue(np.all(np.diff(retained)<0))
        parameters=[metal_loss_parameters('Fe',t,ANCHOR) for t in temperatures]
        self.assertTrue(np.all(np.diff([p['loss_plateau_fraction'] for p in parameters])>0))
        self.assertTrue(np.all(np.diff([p['tau_pulses'] for p in parameters])<0))

    def test_all_fifty_have_coverage_and_saturate_with_bounded_mass(self):
        for element in TARGETS:
            parameters=metal_loss_parameters(element,1000,ANCHOR)
            self.assertEqual(parameters['reference_weight_coverage'],1.)
            self.assertTrue(parameters['escape_proxy_used'])
            self.assertFalse(parameters['sublimation_temperature_available'])
            self.assertFalse(parameters['sublimation_temperature_used'])
            retained=[metal_retention(element,1600,n,ANCHOR)[0] for n in range(1,17)]
            self.assertTrue(np.all(np.diff(retained)<0))
            self.assertLess(abs(retained[-1]-retained[-2]),abs(retained[1]-retained[0]))
            plateau=1-metal_loss_parameters(element,1600,ANCHOR)['loss_plateau_fraction']
            self.assertAlmostEqual(metal_retention(element,1600,10000,ANCHOR)[0],plateau)
            for temperature in (1000,1200,1600):
                balance=mass_balance({element:50},temperature,16)
                row=balance['elements'][element]
                self.assertGreaterEqual(row['retained_metal_g'],0)
                self.assertLessEqual(row['retained_metal_g'],row['feed_metal_g'])
                self.assertAlmostEqual(row['retained_metal_g']+row['lost_metal_g'],row['feed_metal_g'])
                self.assertAlmostEqual(balance['recovered_sample_g'],balance['carrier_after_pulse_g']+row['retained_metal_g'])

    def test_increasing_each_resistance_descriptor_improves_retention(self):
        baseline=elemental_loss_reference('Fe')
        reference=metal_retention('Fe',1600,16,ANCHOR,baseline)[0]
        for field in ('melting_point_K','boiling_point_K','heat_of_formation_kJ_mol'):
            modified={**baseline,field:baseline[field]*1.5}
            self.assertGreater(metal_retention('Fe',1600,16,ANCHOR,modified)[0],reference)
        self.assertGreater(metal_retention('W',1600,16,ANCHOR)[0],metal_retention('Fe',1600,16,ANCHOR)[0])
        self.assertGreater(metal_retention('Fe',1600,16,ANCHOR)[0],metal_retention('Zn',1600,16,ANCHOR)[0])

    def test_sublimation_requires_source_and_matching_pressure_and_is_monotone(self):
        row={**elemental_loss_reference('Fe'),'reported_sublimation_temperature_K':2500,
             'sublimation_pressure_Pa':101325.,'sublimation_source':'TEST_DOCUMENTED_REFERENCE'}
        self.assertTrue(thermal_risk('Fe',1200,row)['sublimation_temperature_used'])
        self.assertFalse(thermal_risk('Fe',1200,row)['escape_proxy_used'])
        higher={**row,'reported_sublimation_temperature_K':3500}
        self.assertGreater(metal_retention('Fe',1600,16,ANCHOR,higher)[0],metal_retention('Fe',1600,16,ANCHOR,row)[0])
        for patch in ({'sublimation_pressure_Pa':1.},{'sublimation_pressure_Pa':None},{'sublimation_source':''}):
            result=thermal_risk('Fe',1200,{**row,**patch})
            self.assertFalse(result['sublimation_temperature_used'])
            self.assertTrue(result['escape_proxy_used'])

    def test_missing_values_remain_absent_and_partial_weights_are_audited(self):
        row={**elemental_loss_reference('Fe'),'melting_point_K':None,'heat_of_formation_kJ_mol':None}
        result=thermal_risk('Fe',1000,row)
        self.assertAlmostEqual(result['reference_weight_coverage'],.35)
        self.assertEqual(set(result['missing_channels']),{'melting','escape'})
        self.assertIsNone(row['melting_point_K'])
        with self.assertRaises(ValueError):
            thermal_risk('Fe',1000,{'element':'Fe'})
        with self.assertRaises(ValueError):
            metal_retention('Fe',1000,1.5,ANCHOR)
        with self.assertRaises(ValueError):
            thermal_risk('Fe',float('nan'))

    def test_invalid_parameters_and_nonfinite_mass_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            thermal_risk('Fe',1000,settings={**loss_settings(),'temperature_exponent':-1})
        with self.assertRaises(ValueError):
            metal_loss_parameters('Fe',1000,ANCHOR,settings={**loss_settings(),'tau_risk_exponent':-1})
        for concentration in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                mass_balance({'Fe':concentration},1000,1)
        with self.assertRaises(ValueError):
            mass_balance({'Fe':50},1000,1,pan_mass_g=float('nan'))
        self.assertGreater(metal_retention('Fe',1000,np.int64(2),ANCHOR)[0],0)


if __name__=='__main__':
    unittest.main()
