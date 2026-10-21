"""Cross-element leakage, source growth and stage-boundary regression checks."""
import copy
import unittest
import numpy as np

from sacfactory.element_campaign import campaign_progress, campaign_training_samples, enforce_campaign, element_transfer_validation
from sacfactory.element_loading import cross_element_loading, single_element_features, combined_element_loading
from sacfactory.loading import LoadingRegressor
from test_loading import campaign


def singles():
    rows = campaign()
    for i, s in enumerate(rows):
        s['origin'] = 'DEMO'
        s['icp_review'] = {'state': 'PASS'}
        s['xrd'] = {'state': 'PASS' if i % 4 < 2 else 'FAIL', 'mode': 'MANUAL'}
    target = copy.deepcopy(rows)
    for i, s in enumerate(target):
        s['sample_id'] = 'CO_' + s['sample_id']
        s['batch_id'] = 'CO_' + s['batch_id']
        s['elements'] = ['Co']
        s['concentrations_mM'] = {'Co': s['concentrations_mM']['Fe']}
        s['icp'][0]['element'] = 'Co'
        s['result_reported_at'] = f'2026-02-{i//4+1:02d}T12:00:00Z'
    return rows + target


class ElementCampaignTests(unittest.TestCase):
    def test_combined_plot_preserves_sample_element_points_and_reports_unavailable_targets(self):
        rows = singles()
        before = copy.deepcopy(rows)
        frame, report, info = combined_element_loading(rows, 'DEMO', ['Fe', 'Co', 'Ni', 'Fe'])
        self.assertEqual(len(frame), 48)
        self.assertEqual(set(frame.element), {'Fe', 'Co'})
        self.assertFalse(frame.duplicated(['sample_id', 'element']).any())
        self.assertEqual(info['metrics']['n'], 48)
        self.assertEqual(report.loc[report.element == 'Ni', 'plotted_samples'].iloc[0], 0)
        for fit in info['fits']:
            self.assertNotIn(fit['target'], fit['train_elements'])
            self.assertFalse(set(fit['train_batch_ids']) & set(fit['test_batch_ids']))
        self.assertEqual(rows, before)

    def test_combined_plot_keeps_contexts_separate_and_does_not_replace_missing_data(self):
        rows = singles()
        for row in rows[24:]:
            row['support_batch'] = 'OTHER_SUPPORT'
        frame, report, info = combined_element_loading(rows, 'DEMO', ['Fe', 'Co', 'Ni'])
        self.assertTrue(frame.empty)
        self.assertTrue((report.plotted_samples == 0).all())
        self.assertEqual(info['metrics']['n'], 0)

    def test_progress_requires_singles_passes_and_named_batches(self):
        rows = singles()
        mixture = copy.deepcopy(rows[0]); mixture['elements'] = ['Fe', 'Ni']
        mixture['concentrations_mM']['Ni'] = .05
        rows.append(mixture)
        frame = campaign_progress(rows, {}, 'DEMO', ['Fe', 'Co', 'Ni'])
        self.assertEqual(frame.candidate_covered.tolist(), [True, True, False])
        self.assertEqual(frame.single_element_records.tolist(), [24, 24, 0])
        self.assertFalse(campaign_progress(rows, {}, 'REAL', ['Fe']).candidate_covered.iloc[0])
        self.assertEqual(len(campaign_training_samples(rows, True, rows[0])), 48)
        self.assertEqual(campaign_training_samples(rows, True, mixture), rows)
        with self.assertRaisesRegex(ValueError, 'Single-element-first'):
            enforce_campaign(mixture, {'_single_element_campaign': {'enabled': True, 'ready': False}})
        enforce_campaign(mixture, {'_single_element_campaign': {'enabled': True, 'ready': True}})
        enforce_campaign(rows[0], {'_single_element_campaign': {'enabled': True, 'ready': False}})

    def test_strict_screening_transfer_excludes_target_mixtures_and_target_outcomes(self):
        rows = singles()
        target = rows[24]
        frame, info = element_transfer_validation(rows, {}, target, 'DEMO', 'Co', ['Fe'], iterations=2)
        self.assertEqual(len(frame), 24)
        self.assertEqual(info['folds'][0]['train_elements'], ['Fe'])
        self.assertFalse(frame.target_seen_before.any())
        changed = copy.deepcopy(rows)
        for s in changed[24:]:
            s['xrd']['state'] = 'FAIL' if s['xrd']['state'] == 'PASS' else 'PASS'
        second, _ = element_transfer_validation(changed, {}, changed[24], 'DEMO', 'Co', ['Fe'], iterations=2)
        np.testing.assert_allclose(frame.p_screen_pass, second.p_screen_pass)

    def test_sequential_screening_replay_only_uses_completed_earlier_batches(self):
        rows = singles()
        frame, info = element_transfer_validation(rows, {}, rows[24], 'DEMO', 'Co', ['Fe'],
            protocol='Sequential element replay', iterations=2, max_folds=2)
        self.assertFalse(frame.empty)
        self.assertFalse(frame.target_seen_before.iloc[0])
        self.assertTrue(frame.target_seen_before.iloc[-1])
        for fold in info['folds']:
            self.assertLess(fold['latest_train_result'], fold['first_test_result'])
            self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))

    def test_cross_element_loading_retains_unseen_descriptors_and_purges_shared_batches(self):
        rows = singles()
        rows[24]['batch_id'] = rows[0]['batch_id']
        before = copy.deepcopy(rows)
        frame, _, info = cross_element_loading(rows, 'DEMO', 'Co', rows[24], ['Fe'])
        self.assertEqual(len(frame), 24)
        self.assertEqual(info['train_elements'], ['Fe'])
        self.assertFalse(set(info['train_batch_ids']) & set(info['test_batch_ids']))
        self.assertTrue(frame.outside_training_range.all())
        self.assertEqual(rows, before)
        x = np.array([single_element_features(s, 'chemistry_ion') for s in rows[:24]])
        model = LoadingRegressor(preserve_constant=True).fit(x, [float(s['icp'][0]['value']) for s in rows[:24]])
        self.assertTrue(model.active.all())
        self.assertFalse(np.array_equal(single_element_features(rows[0], 'chemistry_ion'), single_element_features(rows[24], 'chemistry_ion')))

    def test_cross_element_loading_target_values_do_not_affect_forecasts(self):
        rows = singles()
        first, _, _ = cross_element_loading(rows, 'DEMO', 'Co', rows[24], ['Fe'])
        changed = copy.deepcopy(rows)
        for s in changed[24:]:
            s['icp'][0]['value'] = '70'
        second, _, _ = cross_element_loading(changed, 'DEMO', 'Co', changed[24], ['Fe'])
        np.testing.assert_allclose(first.predicted_w_wt_percent, second.predicted_w_wt_percent)

