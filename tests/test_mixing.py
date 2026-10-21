"""Nominal-feed thermodynamics, source coverage and actual prediction integration."""
import copy
import json
import math
import unittest

import numpy as np

from sacfactory.adaptive import HierarchicalKernel, fit_adaptive
from sacfactory.chemical_descriptors import descriptor_matrix
from sacfactory.chemistry_validation import chemistry_ablation
from sacfactory.composition import transfer_feature
from sacfactory.descriptor_analysis import descriptor_report
from sacfactory.domain import TARGETS
from sacfactory.loading import controls, loading_validation
from sacfactory.mixing import (R, descriptors, feature_vector, component_features, mixing_table,
    reference_identity, pair_coverage, summary)
from sacfactory.outcome_analysis import enriched_frame, variable_groups
from sacfactory.planning import recommend, recommendation_table
from test_chemistry import campaign
from test_loading import campaign as loading_campaign


class MixingTests(unittest.TestCase):
    def test_source_integrity_original_scope_and_primary_spot_checks(self):
        table = mixing_table()
        self.assertEqual(len(table), 2628)
        pairs = pair_coverage(TARGETS)
        self.assertEqual(len(pairs), 1225)
        self.assertTrue(pairs.available.all())
        for a, b, expected in [('Fe', 'Co', -1.), ('Fe', 'Ni', -2.), ('Co', 'Ni', 0.),
                               ('Fe', 'Cu', 13.), ('Fe', 'Mg', 18.)]:
            self.assertEqual(table[tuple(sorted((a, b)))], expected)
        self.assertIn(tuple(sorted(('Fe', 'Tc'))), table)
        self.assertEqual(reference_identity()['primary_doi'], '10.2320/matertrans.46.2817')

    def test_single_binary_unequal_and_equimolar_fifty(self):
        single = descriptors({'Fe': 50.}, 1000.)
        self.assertEqual(single['h_mix_kJ_mol_atoms'], 0.)
        self.assertEqual(single['s_mix_J_mol_K'], 0.)
        self.assertIsNone(single['mix_pair_weight_coverage'])
        self.assertFalse(single['mix_unlike_pairs_applicable'])
        binary = descriptors({'Fe': 50., 'Co': 50.}, 1000.)
        self.assertEqual(binary['h_mix_kJ_mol_atoms'], -1.)
        self.assertAlmostEqual(binary['s_mix_J_mol_K'], R * math.log(2.))
        self.assertAlmostEqual(binary['g_mix_proxy_kJ_mol_atoms'], -1. - 1273.15 * R * math.log(2.) / 1000.)
        unequal = descriptors({'Fe': 75., 'Co': 25.}, 1200.)
        self.assertAlmostEqual(unequal['h_mix_kJ_mol_atoms'], -.75)
        many = descriptors({e: 50. for e in TARGETS}, 1200.)
        self.assertAlmostEqual(many['s_mix_J_mol_K'], R * math.log(50.))

    def test_partial_pairs_are_not_zero_filled_or_renormalized(self):
        d = descriptors({'Fe': 1., 'Co': 1., 'Og': 1.}, 1000., table={('Co', 'Fe'): -1.})
        self.assertIsNone(d['h_mix_kJ_mol_atoms'])
        self.assertIsNone(d['g_mix_proxy_kJ_mol_atoms'])
        self.assertAlmostEqual(d['h_mix_known_contribution_kJ_mol_atoms'], -4. / 9.)
        self.assertAlmostEqual(d['mix_pair_weight_coverage'], 1. / 3.)
        self.assertEqual(len(d['missing_pairs']), 2)
        zero = descriptors({'Co': 1., 'Ni': 1.}, 1000.)
        self.assertTrue(zero['mix_complete'])
        self.assertEqual(zero['h_mix_kJ_mol_atoms'], 0.)

    def test_invalid_inputs_and_missing_cutoff_are_explicit(self):
        for values in [{}, {'Fe': None}, {'Fe': 0}, {'Fe': True}, {'Fe': float('nan')}, {'Unknown': 1.}]:
            with self.assertRaises((ValueError, TypeError)):
                descriptors(values, 1000.)
        d = descriptors({'Fe': 1., 'Co': 1.})
        self.assertEqual(d['h_mix_kJ_mol_atoms'], -1.)
        self.assertIsNone(d['g_mix_proxy_kJ_mol_atoms'])
        row = campaign()[-1]
        row['concentrations_mM']['Fe'] = None
        self.assertIsNone(summary(row)['s_mix_J_mol_K'])

    def test_permutation_scale_invariance_and_no_post_synthesis_features(self):
        row = campaign()[-1]
        original = copy.deepcopy(row)
        first = feature_vector(row)
        changed = copy.deepcopy(row)
        changed.update(elements=list(reversed(row['elements'])), temperature_C=2000., voltage_V=99., current_A=500.)
        changed['icp'] = [{'element': 'Fe', 'value': '88'}]
        changed['xrd']['state'] = 'FAIL'
        np.testing.assert_allclose(first, feature_vector(changed))
        changed['concentrations_mM'] = {e: c * 10. for e, c in changed['concentrations_mM'].items()}
        np.testing.assert_allclose(first, feature_vector(changed))
        self.assertEqual(row, original)
        changed['pulse_limit_C'] += 200.
        self.assertNotEqual(first[2], feature_vector(changed)[2])

    def test_new_profile_preserves_baseline_prefix_and_psd_kernel(self):
        rows = campaign()
        first = np.array([transfer_feature(s, 'chemistry_thermal', True) for s in rows])
        extended = np.array([transfer_feature(s, 'chemistry_mixing', True) for s in rows])
        np.testing.assert_array_equal(first, extended[:, :first.shape[1]])
        self.assertEqual(extended.shape[1] - first.shape[1], 8)
        self.assertGreater(np.ptp(extended[:, -7]), 0.)
        kernel = HierarchicalKernel(positions=descriptor_matrix('chemistry_mixing'))
        self.assertGreater(np.linalg.eigvalsh(kernel(extended)).min(), -1e-9)
        x = controls(rows[-1], 'chemistry_mixing', 'Fe')
        base = controls(rows[-1], 'chemistry_mixing')
        self.assertEqual(len(x) - len(base), 2)
        np.testing.assert_allclose(x[-2:], component_features(rows[-1], 'Fe'))

    def test_fitted_mixing_gp_and_acquisition_export_mark_actual_usage(self):
        rows = campaign()
        target = rows[-1]
        state = fit_adaptive(rows, {}, target, 'REAL', True,
            {'descriptor_profile': 'chemistry_mixing', 'learning_iterations': 2, 'skip_validation': True})
        self.assertIsNotNone(state.model)
        self.assertTrue(state.info['physics']['mixing_descriptors_enabled'])
        self.assertEqual(state.info['mixing_training_n'], 9)
        self.assertEqual(state.model.gp.x.shape[1], len(transfer_feature(target, 'chemistry_mixing')))
        space = {'concentration_bounds_mM': {e: {'min_mM': .01, 'max_mM': .15} for e in target['elements']},
            'pulses': [1, 2], 'pulse_limit_min_C': 900, 'pulse_limit_max_C': 1100, 'max_total_mM': .3,
            'verified': True, 'baseline_repeat': False, 'repeat_strategy': 'none', 'candidate_budget': 256}
        proposals, _ = recommend(state, rows, target, space, 'REAL', 2)
        table = recommendation_table(proposals)
        self.assertTrue(table)
        self.assertTrue(all(r['mixing_descriptors_used_by_model'] for r in table))
        self.assertTrue(all(r['h_mix_kJ_mol_atoms'] is not None for r in table))
        json.dumps(state.info, allow_nan=False)

    def test_analysis_descriptors_and_outcome_variables_are_recipe_level(self):
        rows = campaign()
        before = copy.deepcopy(rows)
        report = descriptor_report(rows, {}, 'REAL', 'chemistry_mixing')
        self.assertEqual(len(report['recipes']), len(rows))
        self.assertIn('s_mix_J_mol_K', report['recipes'])
        frame = enriched_frame(rows, {}, 'REAL', ['Fe', 'Co'])
        self.assertIn('h_mix_kJ_mol_atoms', variable_groups(frame, ['Fe', 'Co'])['Mixing thermodynamics'])
        self.assertEqual(rows, before)

    def test_ablation_uses_identical_outer_rows_and_thermal_baseline_for_mixing(self):
        rows = campaign()
        frame, info = chemistry_ablation(rows, {}, rows[-1], 'REAL', iterations=2)
        thermal = frame.loc[frame.profile == 'chemistry_thermal']
        mixed = frame.loc[frame.profile == 'chemistry_mixing']
        self.assertEqual(set(thermal.sample_id), set(mixed.sample_id))
        effect = next(e for e in info['paired_effects'] if e['added_information'] == 'chemistry_mixing')
        self.assertEqual(effect['baseline'], 'chemistry_thermal')
        for fold in info['folds']:
            self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))

    def test_loading_mixing_profile_masks_heldout_outcomes(self):
        rows = loading_campaign()
        for row in rows:
            row['elements'] = ['Fe', 'Co']
            row['concentrations_mM']['Co'] = .03
        frame, _, info = loading_validation(rows, 'REAL', 'Fe', rows[0], profile='chemistry_mixing', family='Ridge baseline')
        self.assertFalse(frame.empty)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row['sample_id'] in info['folds'][0]['test_sample_ids']:
                row['icp'][0]['value'] = '88'
        second, _, _ = loading_validation(changed, 'REAL', 'Fe', changed[0], profile='chemistry_mixing', family='Ridge baseline')
        np.testing.assert_allclose(frame.loc[frame.fold == 1, 'predicted_w_wt_percent'],
                                   second.loc[second.fold == 1, 'predicted_w_wt_percent'])
