"""Tests for batch effects, protected transfer/replay, repeat design and figures."""
import copy
from io import BytesIO
import json
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd

from sacfactory.adaptive import LearnedGP, fit_adaptive
from sacfactory.adaptive_acquisition import critical_repeat
from sacfactory.composition import transfer_feature
from sacfactory.domain import recipe_key, screen
from sacfactory.planning import recommend
from sacfactory.prediction_plot import agreement_table, descriptive_line, figure_bundle
from sacfactory.validation_research import forward_validation, unseen_element_validation, negative_transfer_validation, paired_effect
from test_transfer import source, context, space


def campaign(element='Fe', prefix='F', batches=6):
    rows = []
    for batch in range(batches):
        for recipe in range(6):
            s = source(batch*6+recipe, element, passed=recipe >= 3, concentration=.025+recipe*.02)
            s.update(sample_id=f'{prefix}_{batch}_{recipe}', batch_id=f'{prefix}_B{batch}',
                pulse_limit_C=950+recipe*20, pulse_count=1,
                result_reported_at=f'2026-01-{batch+1:02d}T12:00:00Z')
            rows.append(s)
    return rows


def audit_frame():
    return pd.DataFrame([{'sample_id': f'S{i}', 'eligible': i != 25, 'label': int(i%5 >= 2),
        'p_screen_pass': .1 + (i%5)*.18, 'batch_id': f'B{i//5}', 'composition': 'Fe', 'element_count': 1,
        'recipe_identity': f'R{i%5}', 'process_identity': 'P'} for i in range(26)])


class ResearchValidationTests(unittest.TestCase):
    def test_batch_covariance_is_supported_by_repeats_and_new_batch_variance_is_integrated(self):
        records = campaign()
        x = np.array([transfer_feature(s) for s in records])
        y = np.array([screen(s, {})['label'] for s in records])
        model = LearnedGP(3).fit(x, y, [s['batch_id'] for s in records])
        self.assertTrue(model.batch_enabled)
        self.assertGreater(model.batch_variance, 0)
        mean, covariance = model.latent(x[:4], True)
        _, variance = model.latent(x[:4])
        self.assertTrue(np.allclose(covariance.diagonal(), variance))
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-8)
        scale = model.batch_variance
        model.batch_variance = 0
        _, process_covariance = model.latent(x[:4], True)
        self.assertTrue(np.allclose(covariance-process_covariance, scale))
        self.assertEqual(model.diagnostics['batch_effect']['cross_batch_repeated_recipes'], 6)
        self.assertEqual(len(model.diagnostics['batch_effect']['offsets']), 6)
        self.assertTrue(all(np.isfinite(row['latent_mean']) and row['conditional_sd'] >= 0
                            for row in model.diagnostics['batch_effect']['offsets']))

    def test_batch_effect_disabled_without_repeat_support_or_by_user(self):
        records = [source(i, passed=i >= 12) for i in range(24)]
        x = np.array([transfer_feature(s) for s in records])
        y = np.array([screen(s, {})['label'] for s in records])
        model = LearnedGP(2).fit(x, y, [s['batch_id'] for s in records])
        self.assertFalse(model.batch_enabled)
        records = campaign()
        x = np.array([transfer_feature(s) for s in records]); y = np.array([screen(s, {})['label'] for s in records])
        model = LearnedGP(2).fit(x, y, [s['batch_id'] for s in records], batch_effect=False)
        self.assertFalse(model.batch_enabled)

    def test_chronological_replay_uses_strictly_earlier_complete_batches(self):
        records = campaign()
        frame, info = forward_validation(records[::-1], {}, records[0], 'REAL', iterations=2, max_folds=2)
        self.assertEqual(info['evaluated_folds'], 2)
        self.assertFalse(frame.empty)
        for fold in info['folds']:
            self.assertLess(pd.Timestamp(fold['latest_train_result']), pd.Timestamp(fold['test_batch_start']))
            self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))
        self.assertEqual(info['protocol'], 'CHRONOLOGICAL_CORRECTED_DATA_REPLAY')

    def test_replay_does_not_infer_dates_or_leak_a_partly_timestamped_batch(self):
        records = campaign()
        records[0].pop('result_reported_at')
        frame, info = forward_validation(records, {}, records[0], 'REAL', iterations=2, max_folds=2)
        self.assertIn('F_B0', info['batches_missing_timestamps'])
        self.assertTrue(all('F_B0' not in fold['train_batch_ids'] for fold in info['folds']))
        frame, info = forward_validation(campaign(), {}, records[0], 'REAL', iterations=2, novel_recipes=True)
        self.assertTrue(frame.empty)

    def test_unseen_element_removes_all_containing_mixtures_and_shared_batches(self):
        records = campaign('Fe', 'F') + campaign('Co', 'C')
        mixture = copy.deepcopy(records[0])
        mixture.update(sample_id='MIX', elements=['Fe', 'Co'], concentrations_mM={'Fe': .03, 'Co': .03}, batch_id='F_B0')
        records.append(mixture)
        frame, info = unseen_element_validation(records, {}, records[0], 'REAL', 'Co', iterations=2)
        self.assertEqual(len(frame), 37)
        fold = info['folds'][0]
        self.assertNotIn('Co', fold['train_elements'])
        self.assertNotIn('F_B0', fold['train_batch_ids'])
        self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))
        missing, info = unseen_element_validation(records, {}, records[0], 'REAL', 'Og', iterations=2)
        self.assertTrue(missing.empty)

    def test_adaptive_transfer_comparison_is_paired_and_protected(self):
        records = campaign('Fe', 'F') + campaign('Co', 'C')
        frame, info = negative_transfer_validation(records, {}, records[-1], 'REAL', iterations=2)
        self.assertEqual(len(frame), 36)
        self.assertFalse(frame[['p_screen_pass', 'p_target_only']].isna().any().any())
        for fold in info['folds']:
            for arm in ('transfer', 'target_only'):
                self.assertFalse(set(fold[arm]['train_batch_ids']) & set(fold[arm]['test_batch_ids']))
                self.assertFalse(set(fold[arm]['kernel_fit_batch_ids']) & set(fold[arm]['calibration_batch_ids']))
        self.assertEqual(info['effect']['independent_batches'], 6)

    def test_negative_transfer_signal_and_inconclusive_are_not_hidden(self):
        frame = pd.DataFrame({'label': [0, 1]*9, 'p_target_only': [.1, .9]*9,
                              'p_screen_pass': [.9, .1]*9, 'batch_id': np.repeat(['B1', 'B2', 'B3'], 6)})
        effect = paired_effect(frame)
        self.assertEqual(effect['status'], 'NEGATIVE_TRANSFER_SIGNAL')
        self.assertLess(effect['upper_95'], 0)
        frame['p_screen_pass'] = frame['p_target_only']
        self.assertEqual(paired_effect(frame)['status'], 'INCONCLUSIVE')
        self.assertEqual(paired_effect(frame.iloc[:6])['status'], 'INSUFFICIENT_BATCHES')

    def test_critical_repeat_is_an_observed_bounded_recipe_with_provenance(self):
        records = campaign()
        ctx = context('Fe')
        state = fit_adaptive(records, {}, ctx, 'REAL', settings={'learning_iterations': 2})
        operating = {**space('Fe'), 'baseline_repeat': True, 'repeat_strategy': 'critical'}
        rows, _ = recommend(state, records, ctx, operating, 'REAL', 3)
        repeat = [row for row in rows if row.get('is_repeat')]
        self.assertEqual(len(repeat), 1)
        self.assertIn(recipe_key(repeat[0]['recipe']), {recipe_key(s) for s in records})
        self.assertIn('repeat_priority', repeat[0]['repeat_rationale'])
        reserved = {recipe_key(s) for s in records}
        result, rationale = critical_repeat(state, ctx, operating['concentration_bounds_mM'], [1], 900, 1050, .2, reserved)
        self.assertIsNone(result)

    def test_agreement_excludes_ineligible_and_preserves_denominators_and_batch_intervals(self):
        frame = audit_frame()
        table = agreement_table(frame)
        self.assertEqual(table.n.sum(), 25)
        self.assertTrue(table.interval_method.str.contains('cluster bootstrap').all())
        self.assertTrue(table['mean_prediction'].between(0, 1).all())
        self.assertTrue(table['observed_pass_rate'].between(0, 1).all())
        recipe = agreement_table(frame, recipe_level=True)
        self.assertEqual(recipe.n.sum(), 25)
        single = frame.iloc[:5].copy()
        self.assertTrue(agreement_table(single, recipe_level=True).empty)

    def test_descriptive_fit_requires_three_distinct_points_and_handles_constant_outcomes(self):
        table = agreement_table(audit_frame())
        self.assertIsNotNone(descriptive_line(table))
        self.assertIsNone(descriptive_line(table.iloc[:2]))
        table['observed_pass_rate'] = 1
        self.assertIsNone(descriptive_line(table)['aggregated_r_squared'])

    def test_vector_export_contains_editable_text_source_records_and_simulation_label(self):
        frame = audit_frame()
        bundle = figure_bundle(agreement_table(frame), frame, 'DEMO', True)
        with ZipFile(BytesIO(bundle)) as archive:
            self.assertEqual(archive.testzip(), None)
            self.assertIn('<text', archive.read('prediction_agreement.svg').decode())
            self.assertTrue(archive.read('prediction_agreement.pdf').startswith(b'%PDF'))
            notes = json.loads(archive.read('figure_notes.json'))
            self.assertEqual(notes['plotted_observations'], 25)
            self.assertEqual(notes['origin'], 'DEMO')
            self.assertIsNotNone(notes['fit'])

    def test_vector_export_preserves_intervals_not_centered_on_point_estimates(self):
        frame = audit_frame()
        table = agreement_table(frame)
        table.loc[table.index[0], ['observed_pass_rate', 'lower_95', 'upper_95']] = [1., .8, 1. - 1e-15]
        table.loc[table.index[1], ['observed_pass_rate', 'lower_95', 'upper_95']] = [.2, .3, .5]
        with ZipFile(BytesIO(figure_bundle(table, frame, 'DEMO'))) as archive:
            exported = pd.read_csv(BytesIO(archive.read('agreement_points.csv')))
            self.assertAlmostEqual(exported.loc[1, 'lower_95'], .3)
            self.assertAlmostEqual(exported.loc[1, 'observed_pass_rate'], .2)
            self.assertTrue(archive.read('prediction_agreement.pdf').startswith(b'%PDF'))


if __name__ == '__main__':
    unittest.main()
