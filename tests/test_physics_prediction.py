"""Physics-assisted transfer, protected residual learning and prior boundaries."""
import copy
import json
import unittest

import numpy as np

from sacfactory.adaptive import HierarchicalKernel
from sacfactory.chemical_descriptors import descriptor_matrix
from sacfactory.composition import transfer_feature
from sacfactory.domain import ELEMENTS, TARGETS
from sacfactory.element_loading import cross_element_loading, combined_element_loading
from sacfactory.loading import loading_validation
from sacfactory.physics_features import (thermal_matrix, loading_prior, material_assumptions,
    physics_manifest, loss_features)
from test_loading import campaign

PHYSICS = {'pan_mass_g': 2., 'solvent_volume_mL': 20., 'carbonization_loss': .06,
           'confirmed': True, 'source': 'Synthetic comparable-run fixture; user anchors are unfitted'}


class PhysicsPredictionTests(unittest.TestCase):
    def test_sourced_thermal_values_masks_and_unknown_elements(self):
        x = thermal_matrix()
        self.assertEqual(x.shape, (118, 6))
        self.assertTrue(np.isfinite(x).all())
        for e in TARGETS:
            self.assertTrue((x[ELEMENTS.index(e), :3] > 0).all())
            self.assertTrue((x[ELEMENTS.index(e), 3:] == 0).all())
        unknown = x[ELEMENTS.index('Og')]
        np.testing.assert_array_equal(unknown[:3], 0.)
        np.testing.assert_array_equal(unknown[3:], 1.)
        matrix = descriptor_matrix('chemistry_thermal')
        baseline = descriptor_matrix('chemistry_state')
        np.testing.assert_array_equal(matrix[:, :baseline.shape[1]], baseline)
        np.testing.assert_array_equal(matrix[:, baseline.shape[1]:], x)

    def test_features_are_permutation_invariant_and_exclude_results(self):
        sample = campaign()[0]
        sample.update(elements=['Fe', 'Zn'], concentrations_mM={'Fe': 50., 'Zn': 30.})
        x = transfer_feature(sample, 'chemistry_loss', True)
        changed = copy.deepcopy(sample)
        changed.update(elements=['Zn', 'Fe'], temperature_C=1900, voltage_V=99, current_A=500,
                       latent_mass_balance={'true_w_wt_percent': 99.})
        changed['icp'][0]['value'] = '90'
        changed['xrd']['state'] = 'FAIL'
        np.testing.assert_allclose(x, transfer_feature(changed, 'chemistry_loss', True))
        self.assertEqual(len(x) - len(transfer_feature(sample, 'chemistry_thermal', True)), 4)
        unknown = copy.deepcopy(sample)
        unknown.update(elements=['Og'], concentrations_mM={'Og': 50.})
        features = loss_features(unknown)
        self.assertEqual(features[:3], [0., 0., 0.])

    def test_new_kernel_remains_psd_and_cutoff_changes_prior(self):
        rows = campaign()[:8]
        for profile in ['chemistry_thermal', 'chemistry_loss']:
            x = np.array([transfer_feature(s, profile, True) for s in rows])
            kernel = HierarchicalKernel(positions=descriptor_matrix(profile))
            self.assertGreaterEqual(np.linalg.eigvalsh(kernel(x)).min(), -1e-9)
        s = rows[0]
        low = loss_features({**s, 'pulse_limit_C': 1000., 'pulse_count': 1})
        high = loss_features({**s, 'pulse_limit_C': 1600., 'pulse_count': 16})
        self.assertLess(high[0], low[0])
        self.assertGreater(high[-1], low[-1])
        self.assertTrue(physics_manifest('chemistry_thermal')['thermal_descriptors_enabled'])
        self.assertFalse(physics_manifest('chemistry_thermal')['empirical_loss_features_enabled'])

    def test_explicit_material_assumptions_and_mass_conserving_anchor(self):
        row = campaign()[0]
        row.update(concentrations_mM={'Fe': 50.}, pulse_limit_C=1000., pulse_count=1)
        self.assertAlmostEqual(loading_prior(row, 'Fe', PHYSICS), 2.5)
        mixture = {**row, 'elements': ['Fe', 'Au'], 'concentrations_mM': {'Fe': 50., 'Au': 50.}}
        self.assertLess(loading_prior(mixture, 'Fe', PHYSICS), 2.5)
        for settings in [None, {}, {**PHYSICS, 'confirmed': False}, {**PHYSICS, 'pan_mass_g': None},
                         {**PHYSICS, 'solvent_volume_mL': float('nan')}, {**PHYSICS, 'source': ''}]:
            with self.assertRaises(ValueError):
                material_assumptions(settings)
        unknown = {**mixture, 'elements': ['Fe', 'Og'], 'concentrations_mM': {'Fe': 50., 'Og': 1.}}
        with self.assertRaisesRegex(ValueError, 'Og'):
            loading_prior(unknown, 'Fe', PHYSICS)

    def test_residual_and_direct_comparisons_use_identical_protected_folds(self):
        rows = campaign()
        for row in rows:
            row['concentrations_mM']['Fe'] *= 1000.
            prior = loading_prior(row, 'Fe', PHYSICS)
            row['icp'][0]['value'] = str(np.expm1(np.log1p(prior) + .15))
        before = copy.deepcopy(rows)
        first, _, info = loading_validation(rows, 'REAL', 'Fe', rows[0], physics=PHYSICS)
        self.assertEqual(len(first), len(rows))
        self.assertEqual({v['n'] for v in info['paired_comparison']}, {len(rows)})
        self.assertLess(info['paired_comparison'][2]['mae_wt_percent'], 1e-6)
        for fold in info['folds']:
            self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))
        changed = copy.deepcopy(rows)
        for s in changed:
            s.update(temperature_C=2000, voltage_V=90, current_A=800)
            if s['sample_id'] in info['folds'][0]['test_sample_ids']:
                s['icp'][0]['value'] = '95'
        second, _, _ = loading_validation(changed, 'REAL', 'Fe', changed[0], physics=PHYSICS)
        np.testing.assert_allclose(first.loc[first.fold == 1, 'predicted_w_wt_percent'],
                                   second.loc[second.fold == 1, 'predicted_w_wt_percent'])
        self.assertEqual(rows, before)
        json.dumps(info, allow_nan=False)

    def test_cross_element_residual_purges_target_and_shared_batches(self):
        rows = campaign()
        for i, row in enumerate(rows):
            e = 'Fe' if i < 16 else 'Co'
            c = row['concentrations_mM']['Fe'] * 1000
            row.update(elements=[e], concentrations_mM={e: c})
            row['icp'][0]['element'] = e
            row['icp'][0]['value'] = str(loading_prior(row, e, PHYSICS))
        first, _, info = cross_element_loading(rows, 'REAL', 'Co', rows[-1], physics=PHYSICS,
                                             profile='chemistry_loss', family='Ridge baseline')
        self.assertEqual(len(first), 8)
        self.assertEqual(info['train_elements'], ['Fe'])
        self.assertFalse(set(info['train_batch_ids']) & set(info['test_batch_ids']))
        changed = copy.deepcopy(rows)
        for row in changed[16:]:
            row['icp'][0]['value'] = '90'
        second, _, _ = cross_element_loading(changed, 'REAL', 'Co', changed[-1], physics=PHYSICS,
                                              profile='chemistry_loss', family='Ridge baseline')
        np.testing.assert_allclose(first.predicted_w_wt_percent, second.predicted_w_wt_percent)
        frame, _, combined = combined_element_loading(rows, 'REAL', ['Co'], physics=PHYSICS)
        self.assertEqual(len(frame), 8)
        json.dumps(combined, allow_nan=False)
