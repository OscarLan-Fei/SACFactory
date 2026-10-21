"""Recipe-count, adaptive-transfer and repetition-allocation tests."""
import copy
import json
import unittest

import numpy as np

from sacfactory.domain import recipe_key
from sacfactory.planning import fit_model, recommend
from sacfactory.recipe_reliability import recipe_evidence, CountReliabilityModel
from sacfactory.reliable_transfer import optimize_weights
from sacfactory.repetition_allocation import future_repeat_budget
from test_research_validation import campaign
from test_transfer import context, space


class ReliabilityTests(unittest.TestCase):
    def state(self, records=None, ctx=None):
        return fit_model(records or campaign(), {}, ctx or context('Fe'), 'REAL', True,
                         engine='reliability', settings={'learning_iterations': 2})

    def test_same_batch_duplicates_do_not_count_as_independent_evidence(self):
        records = campaign()
        before = recipe_evidence(records, {})[0]
        clone = copy.deepcopy(records[0]); clone['sample_id'] = 'EXTRA'
        after = recipe_evidence(records+[clone], {})[0]
        self.assertEqual(after['independent_batches'], before['independent_batches'])
        self.assertEqual(after['n'], before['n']+1)
        self.assertEqual(after['a'], before['a'])
        self.assertEqual(after['b'], before['b'])

    def test_count_challenger_uses_beta_binomial_only_with_supported_groups(self):
        records = campaign()
        model = CountReliabilityModel().fit(records, {})
        self.assertEqual(model.diagnostics['family'], 'BINOMIAL')
        repeated = records+[dict(copy.deepcopy(s), sample_id=s['sample_id']+'_REP') for s in records]
        model = CountReliabilityModel().fit(repeated, {})
        self.assertEqual(model.diagnostics['family'], 'BETA_BINOMIAL')
        self.assertEqual(model.diagnostics['n_samples'], 72)
        self.assertTrue(2 <= model.precision <= 1000)

    def test_harmful_source_weights_fall_under_paired_target_development_loss(self):
        y = np.array([0, 1]*12)
        p = np.column_stack([np.full(len(y), .5), np.where(y, .9, .1), np.where(y, .1, .9)])
        weights, info = optimize_weights(p, y, np.repeat(['B1', 'B2', 'B3'], 8), np.array([1., 0., 0.]))
        self.assertGreater(weights[1], .9)
        self.assertLess(weights[2], .01)
        self.assertLessEqual(info['objective_after'], info['objective_before'])
        self.assertAlmostEqual(weights.sum(), 1)

    def test_oof_weights_purge_actual_test_batches_for_every_expert(self):
        state = self.state(campaign()+campaign('Co', 'C'))
        self.assertEqual(state.info['transfer_weights']['status'], 'OOF_TARGET_WEIGHT_FIT')
        for fold in state.info['transfer_weights']['folds']:
            for expert in fold['experts']:
                self.assertFalse(set(expert['train_batch_ids']) & set(fold['test_batch_ids']))
        self.assertEqual(state.model.calibration_status, 'UNCALIBRATED_ENSEMBLE')
        json.dumps(state.info, allow_nan=False)

    def test_unseen_target_retains_declared_prior_and_never_fake_calibration(self):
        state = self.state(campaign(), context('Co'))
        self.assertEqual(state.info['transfer_weights']['status'], 'POOLED_PRIOR_ONLY')
        self.assertEqual(state.model.weights[0], 1)
        from sacfactory.composition import transfer_feature
        x = np.array([transfer_feature(context('Co'))])
        self.assertFalse(state.model.calibration_mask(x).any())
        a = state.model.probability_draws(x, joint=True)
        self.assertTrue(np.array_equal(a, state.model.probability_draws(x, joint=True)))
        self.assertTrue(np.isfinite(a).all())

    def test_repeat_budget_decreases_as_independent_evidence_increases(self):
        first, _, _ = future_repeat_budget(2.5, 2.5, 4, .2, 50)
        later, _, _ = future_repeat_budget(15.5, 15.5, 30, .2, 50)
        self.assertGreater(first, later)
        self.assertEqual(later, 0)
        m, _, capped = future_repeat_budget(.5, .5, 0, .05, 2)
        self.assertEqual(m, 2); self.assertTrue(capped)

    def test_repeat_and_new_recipe_slots_are_unique_bounded_and_budgeted(self):
        records, ctx = campaign(), context('Fe')
        state = self.state(records, ctx)
        operating = {**space('Fe'), 'baseline_repeat': True, 'repeat_strategy': 'allocation',
                     'repeat_budget_fraction': .5, 'repeat_goal_half_width': .1, 'repeat_cap': 20}
        rows, _ = recommend(state, records, ctx, operating, 'REAL', 6)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({recipe_key(r['recipe']) for r in rows}), 6)
        repeats = [r for r in rows if r.get('is_repeat')]
        self.assertGreater(len(repeats), 0); self.assertLessEqual(len(repeats), 3)
        self.assertTrue(all(r['repeat_rationale']['planned_additional_repeats'] >= 1 for r in repeats))
        self.assertTrue(all(r['policy'] == 'RELIABILITY_INFORMATION_AND_REPEAT_BUDGET' for r in rows))

    def test_design_only_preserves_blank_predictions(self):
        state = self.state(campaign(batches=1))
        self.assertIsNone(state.model)
        rows, _ = recommend(state, [], context('Fe'), space('Fe'), 'REAL', 2)
        self.assertTrue(all(row['p_screen_pass'] is None for row in rows))

    def test_chemical_kernel_has_real_coverage_explicit_missing_masks_and_psd(self):
        from sacfactory.chemical_descriptors import descriptor_matrix, reference, coverage, model_features
        from sacfactory.adaptive import HierarchicalKernel
        from sacfactory.composition import transfer_feature
        from sacfactory.domain import ELEMENTS
        properties = reference()
        self.assertEqual(len(properties['elements']), 118)
        self.assertEqual(len(properties['snapshot_sha256']), 64)
        self.assertEqual(coverage(['Fe'])[0]['en_pauling'], 1.83)
        matrix = descriptor_matrix('chemistry')
        self.assertTrue(np.isfinite(matrix).all())
        self.assertGreater(matrix[:, -5:].sum(), 0)
        x = np.array([transfer_feature(context(e)) for e in ['Fe', 'Co', 'Og']])
        kernel = HierarchicalKernel(positions=matrix)
        self.assertGreaterEqual(np.linalg.eigvalsh(kernel(x)).min(), -1e-9)
        self.assertTrue(np.isfinite(model_features(x)).all())
        self.assertTrue(np.allclose(np.diag(kernel(x)), kernel.diag(x)))

    def test_chemistry_profile_and_reference_hash_enter_model_provenance(self):
        state = fit_model(campaign(), {}, context('Fe'), 'REAL', True, engine='reliability',
            settings={'learning_iterations': 2, 'descriptor_profile': 'chemistry'})
        self.assertEqual(state.model.gp.diagnostics['descriptor_profile'], 'chemistry')
        self.assertIn('descriptor_snapshot_sha256', state.info)

    def test_beta_calibration_is_monotone_and_uses_protected_batches(self):
        from sacfactory.extra_models import BetaCalibrator, TabularCandidate
        from sacfactory.composition import transfer_feature
        from sacfactory.domain import screen
        calibrator = BetaCalibrator().fit(np.linspace(.01, .99, 40), np.array([0]*20+[1]*20))
        self.assertTrue(np.all(np.diff(calibrator.transform(np.linspace(.01, .99, 50))) >= 0))
        records = campaign(batches=20)
        model = TabularCandidate('boosting', calibration=True).fit(np.array([transfer_feature(s) for s in records]),
            np.array([screen(s, {})['label'] for s in records]), [s['batch_id'] for s in records])
        self.assertEqual(model.provenance['calibration_status'], 'BETA_HOLDOUT_FIT')
        self.assertFalse(set(model.provenance['fit_batch_ids']) & set(model.provenance['calibration_batch_ids']))

    def test_model_and_descriptor_benchmark_uses_identical_outer_test_rows(self):
        from sacfactory.extra_models import reliability_benchmark
        records = campaign()+campaign('Co', 'C')
        frame, info = reliability_benchmark(records, {}, records[0], 'REAL', iterations=2, max_folds=2)
        self.assertEqual(frame.model.nunique(), 6)
        groups = [set(part.sample_id) for _, part in frame.groupby('model')]
        self.assertTrue(all(ids == groups[0] for ids in groups))
        for fold in info['folds']:
            self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))
            for fit in fold['weight_fits']:
                for inner in fit['folds']:
                    self.assertFalse(set(inner['test_batch_ids']) & set(fold['test_batch_ids']))
        self.assertEqual(len(info['comparison']), 6)

    def test_benchmark_strict_element_and_forward_modes_do_not_use_forbidden_rows(self):
        from sacfactory.extra_models import reliability_benchmark
        records = campaign()+campaign('Co', 'C')
        frame, info = reliability_benchmark(records, {}, records[0], 'REAL', iterations=2, protocol='unseen_element', element='Co')
        self.assertTrue(set(info['folds'][0]['train_sample_ids']).issubset({s['sample_id'] for s in records if s['elements'] == ['Fe']}))
        frame, info = reliability_benchmark(records, {}, records[0], 'REAL', iterations=2, protocol='forward', max_folds=2)
        for fold in info['folds']:
            self.assertLess(fold['latest_train_result'], fold['first_test_result'])

    def test_missing_batch_or_controls_never_create_recipe_reliability_counts(self):
        record = campaign()[0]
        record['batch_id'] = 'UNKNOWN'
        self.assertEqual(recipe_evidence([record], {}), [])
        record['batch_id'] = 'B'; record['pulse_limit_C'] = None
        self.assertEqual(recipe_evidence([record], {}), [])

    def test_robustness_uses_declared_tolerances_and_rejects_out_of_bound_scenarios(self):
        from sacfactory.robustness import robustness_summary
        ctx = context('Fe'); state = self.state(ctx=ctx)
        operating = space('Fe')
        self.assertIsNone(robustness_summary(state.model, ctx, operating))
        operating.update(concentration_tolerance_fraction=.1, cutoff_tolerance_C=5)
        result = robustness_summary(state.model, ctx, operating)
        self.assertEqual(result['valid_scenarios'], 16)
        self.assertLessEqual(result['scenario_min_probability'], result['nominal_probability'])
        edge = copy.deepcopy(ctx); edge['concentrations_mM']['Fe'] = .01; edge['pulse_limit_C'] = 900
        result = robustness_summary(state.model, edge, operating)
        self.assertLess(result['valid_scenarios'], result['requested_scenarios'])
        operating['concentration_tolerance_fraction'] = .8
        with self.assertRaises(ValueError):
            robustness_summary(state.model, ctx, operating)

    def test_future_independent_batches_have_separate_random_intercepts(self):
        from sacfactory.composition import transfer_feature
        state = self.state()
        x = np.array([transfer_feature(context('Fe'))]*2)
        _, shared = state.model.gp.latent(x, True)
        _, independent = state.model.gp.latent(x, True, ['NEXT_A', 'NEXT_B'])
        self.assertTrue(np.allclose(shared.diagonal(), independent.diagonal()))
        self.assertAlmostEqual(shared[0, 1]-independent[0, 1], state.model.gp.batch_variance)


if __name__ == '__main__':
    unittest.main()
