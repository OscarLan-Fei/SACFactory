"""Scientific coverage boundaries and categorical chart regression checks."""
import copy
import unittest

import numpy as np
import pandas as pd

from sacfactory.descriptor_analysis import descriptor_report, weighted_summary, profile_manifest
from sacfactory.precursor_catalog import planning_draft
from sacfactory.ui_charts import correlation_grid, matrix_chart, row_height, interaction_chart
from test_chemistry import campaign


class DescriptorAnalysisTests(unittest.TestCase):
    def test_known_only_weighted_moments_and_missing_concentrations(self):
        mean, sd, fraction = weighted_summary([2., None, 4.], [.25, .5, .25])
        self.assertEqual((mean, sd, fraction), (3., 1., .5))
        self.assertEqual(weighted_summary([None], [1.]), (None, None, 0.))
        self.assertEqual(weighted_summary([2.], None), (None, None, None))

    def test_historical_masks_namespace_selection_and_no_backfill(self):
        rows = campaign()[:2]
        rows[0]['precursors'] = planning_draft(['Fe'])
        rows[1]['concentrations_mM']['Fe'] = None
        demo = copy.deepcopy(rows[0])
        demo.update(sample_id='DEMO_ONLY', origin='DEMO')
        original = copy.deepcopy(rows + [demo])
        report = descriptor_report(rows + [demo], {}, 'REAL')
        self.assertEqual(report['provenance']['sample_ids'], [rows[0]['sample_id'], rows[1]['sample_id']])
        recipes = report['recipes'].set_index('sample_id')
        self.assertTrue(pd.isna(recipes.loc[rows[0]['sample_id'], 'formal_oxidation_state_mean']))
        self.assertEqual(recipes.loc[rows[0]['sample_id'], 'formal_oxidation_state_coverage'], 0.)
        self.assertTrue(pd.isna(recipes.loc[rows[1]['sample_id'], 'en_pauling_mean']))
        self.assertEqual(report['members'].declaration_status.iloc[0], 'Catalog assumption')
        self.assertEqual(rows + [demo], original)

    def test_mixture_descriptors_are_permutation_invariant_and_state_specific(self):
        row = campaign()[-1]
        row['precursors']['Fe'].update(coordination='VI', spin='HS', state_source='Explicit synthetic reference state')
        reordered = copy.deepcopy(row)
        reordered['elements'] = list(reversed(row['elements']))
        first = descriptor_report([row], {}, 'REAL')['recipes']
        second = descriptor_report([reordered], {}, 'REAL')['recipes']
        numeric = [c for c in first if c.endswith(('_mean', '_sd', '_coverage', '_fraction'))]
        np.testing.assert_allclose(first[numeric].to_numpy(dtype=float), second[numeric].to_numpy(dtype=float), equal_nan=True)
        self.assertEqual(first.ionic_radius_pm_mean.iloc[0], 78.)
        self.assertAlmostEqual(first.ionic_radius_pm_coverage.iloc[0], row['concentrations_mM']['Fe'] / sum(row['concentrations_mM'].values()))

    def test_manifest_distinguishes_enabled_salt_and_ionic_features(self):
        ion = profile_manifest('chemistry_ion')
        self.assertFalse(ion.loc[ion.descriptor.str.contains('Salt family'), 'enabled'].iloc[0])
        self.assertFalse(ion.loc[ion.descriptor.str.contains('Matched ionic'), 'enabled'].iloc[0])
        from sacfactory.mixing import LABELS as MIXING_LABELS
        state = profile_manifest('chemistry_state')
        self.assertFalse(state.loc[state.descriptor.str.startswith('PulseICP physical-control'), 'enabled'].iloc[0])
        state = state.loc[~state.descriptor.str.startswith('PulseICP physical-control')]
        self.assertTrue(state.loc[~state.descriptor.isin(['melting_point_K', 'boiling_point_K',
            'heat_of_formation_kJ_mol', 'Empirical metal-retention moments and carrier loss', *MIXING_LABELS.values()]), 'enabled'].all())
        self.assertFalse(state.loc[state.descriptor == 'melting_point_K', 'enabled'].iloc[0])
        loss = profile_manifest('chemistry_loss')
        loss = loss.loc[~loss.descriptor.str.startswith('PulseICP physical-control')]
        self.assertTrue(loss.loc[~loss.descriptor.isin(MIXING_LABELS.values()), 'enabled'].all())
        migrated = profile_manifest('chemistry_pulseicp')
        self.assertTrue(migrated.loc[migrated.descriptor.str.startswith('PulseICP physical-control'), 'enabled'].iloc[0])

    def test_matrix_preserves_all_axes_order_missing_values_and_large_inline_data(self):
        variables = [f'variable_{i}' for i in range(8)]
        pairs = pd.DataFrame([{'parameter_1':variables[0], 'parameter_2':variables[1], 'paired_n':4, 'spearman_rho':.5}])
        grid = correlation_grid(pairs, variables)
        self.assertEqual(len(grid), 64)
        self.assertEqual(grid.spearman_rho.notna().sum(), 2)
        spec = matrix_chart(grid, 'parameter_1', 'parameter_2', 'spearman_rho', variables, variables, 'rho').to_dict()
        self.assertEqual(spec['encoding']['y']['scale']['domain'], variables)
        self.assertFalse(spec['encoding']['y']['axis']['labelOverlap'])
        self.assertEqual(spec['height'], {'step':32})
        large = pd.DataFrame([{'x':str(i), 'y':str(j), 'value':0.} for i in range(118) for j in range(118)])
        large_spec = matrix_chart(large, 'x', 'y', 'value', list(map(str,range(118))), list(map(str,range(118))), 'Count').to_dict()
        self.assertEqual(len(large_spec['data']['values']), 13924)
        self.assertEqual(large_spec['encoding']['color']['condition']['scale']['domain'], [0, 1.])
        self.assertGreater(row_height(118), row_height(12))

    def test_interaction_axes_reserve_bin_space_and_sort_numerically(self):
        frame = pd.DataFrame([
            {'x_range': '0.04 to 0.08', 'y_range': '1000 to 1105', 'x_order': .06, 'y_order': 1052.5, 'pass_rate': 1., 'n': 1},
            {'x_range': '0.015 to 0.03', 'y_range': '829 to 911.8', 'x_order': .0225, 'y_order': 870.4, 'pass_rate': 0., 'n': 1}])
        spec = interaction_chart(frame, 'Fe precursor / mM', 'Controlled cutoff / C').to_dict()
        self.assertEqual(spec['height'], {'step': 52})
        self.assertEqual(spec['width'], {'step': 180})
        self.assertEqual(spec['encoding']['y']['scale']['domain'], ['829 to 911.8', '1000 to 1105'])
        self.assertEqual(spec['encoding']['x']['scale']['domain'], ['0.015 to 0.03', '0.04 to 0.08'])
        self.assertFalse(spec['encoding']['y']['axis']['labelOverlap'])
        self.assertEqual(spec['encoding']['x']['axis']['labelAngle'], 0)
        self.assertEqual(len(spec['data']['values']), 2)


if __name__ == '__main__':
    unittest.main()
