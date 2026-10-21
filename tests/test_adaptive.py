"""Numerical, leakage, adaptation and workflow tests for the learned engine."""
import copy
import json
import tempfile
import unittest

import numpy as np

from sacfactory.adaptive import HierarchicalKernel, LearnedGP, AdaptivePredictor, fit_adaptive, entropy
from sacfactory.adaptive_acquisition import greedy_batch, joint_information, dynamic_policy
from sacfactory.composition import transfer_feature
from sacfactory.domain import ELEMENTS, TARGETS, RULE_VERSION, recipe, screen
from sacfactory.planning import fit_model, recommend, recommendation_table
from sacfactory.storage import Store
from test_transfer import source, context, space


class AdaptiveTests(unittest.TestCase):
    def test_large_gp_budget_excludes_calibration_and_preserves_full_challenger_pool(self):
        rows=[source(i,passed=i%4>=2) for i in range(400)]
        x=np.array([transfer_feature(s) for s in rows]); y=np.array([int(i%4>=2) for i in range(400)])
        groups=np.array([s['batch_id'] for s in rows])
        model=AdaptivePredictor(2).fit(x,y,groups)
        self.assertEqual(len(model.gp_fit_indices),256)
        self.assertGreater(len(model.fit_indices),256)
        self.assertTrue(set(model.gp_fit_indices)<=set(model.fit_indices))
        self.assertFalse(set(model.gp_fit_indices)&set(model.calibration_indices))
        self.assertEqual(len(model.gp.x),256)
        self.assertTrue(np.isfinite(model.challenger.predict_proba(x)).all())

    def sources(self, n=24):
        return [source(i, passed=i % 4 >= 2) for i in range(n)]

    def state(self, samples=None, ctx=None):
        return fit_model(samples or self.sources(), {}, ctx or context(), 'REAL', True,
                         engine='adaptive', settings={'learning_iterations': 4})

    def test_learned_kernel_psd_and_permutation_invariance_across_cardinalities(self):
        points = []
        for members in (['Fe'], ['Co'], ['Fe', 'Co'], ['Co', 'Fe'], ['Fe', 'Ni', 'Co'], TARGETS, ELEMENTS):
            s = source(1)
            s.update(elements=members, concentrations_mM=recipe(members))
            points.append(transfer_feature(s))
        x = np.array(points)
        for perturbation in (np.zeros(9), np.arange(9) * .15 - .5):
            from sacfactory.adaptive import INITIAL
            kernel = HierarchicalKernel(INITIAL + perturbation)
            k = kernel(x)
            self.assertGreaterEqual(np.linalg.eigvalsh(k).min(), -1e-9)
            self.assertTrue(np.allclose(k.diagonal(), kernel.diag(x)))
            self.assertTrue(np.allclose(k[2], k[3]))

    def test_evidence_learning_changes_parameters_without_worsening_regularized_loss(self):
        state = self.state()
        diagnostics = state.info['learning']
        self.assertLessEqual(diagnostics['final_regularized_loss'], diagnostics['initial_regularized_loss'] + 1e-7)
        self.assertNotEqual(diagnostics['parameters'], diagnostics['initial_parameters'])
        self.assertEqual(state.info['engine'], 'adaptive')
        self.assertEqual(state.info['n_labeled'], 24)
        json.dumps(state.info, allow_nan=False)

    def test_joint_covariance_matches_marginal_and_draws_are_reproducible(self):
        state = self.state()
        x = np.array([transfer_feature(source(i)) for i in range(6)])
        mean, covariance = state.model.gp.latent(x, True)
        marginal_mean, variance = state.model.gp.latent(x)
        self.assertTrue(np.allclose(mean, marginal_mean))
        self.assertTrue(np.allclose(covariance.diagonal(), variance))
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-8)
        a = state.model.probability_draws(x, joint=True)
        b = state.model.probability_draws(x, joint=True)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.all((a >= 0) & (a <= 1)))

    def test_joint_information_avoids_redundant_batch_points(self):
        patterns = np.tile([[.01, .01, .01], [.01, .01, .99], [.99, .99, .01], [.99, .99, .99]], (32, 1))
        selected, trace = greedy_batch(patterns, np.array([.5, .5, .5]), 2, 2.)
        self.assertEqual(selected, [0, 2])
        self.assertGreater(joint_information(patterns, [0, 2]), joint_information(patterns, [0, 1]))
        self.assertAlmostEqual(trace[-1]['joint_information_bits'], joint_information(patterns, selected), places=10)
        constant = np.full((64, 3), .7)
        self.assertAlmostEqual(joint_information(constant, [0, 1, 2]), 0, places=10)

    def test_calibration_and_outer_validation_batches_are_disjoint(self):
        samples = self.sources(96)
        x = np.array([transfer_feature(s) for s in samples])
        y = np.array([screen(s, {})['label'] for s in samples])
        groups = np.array([s['batch_id'] for s in samples])
        model = AdaptivePredictor(3).fit(x, y, groups)
        self.assertGreater(len(model.calibration_indices), 0)
        self.assertFalse(set(groups[model.fit_indices]) & set(groups[model.calibration_indices]))
        state = self.state(samples)
        for fold in state.info['validation_provenance']:
            self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))
            self.assertFalse(set(fold['calibration_batch_ids']) & set(fold['kernel_fit_batch_ids']))

    def test_new_target_is_uncalibrated_and_mixture_holdouts_never_train_higher_stages(self):
        samples = self.sources(24)
        for i in range(24, 48):
            s = source(i, passed=i % 4 >= 2)
            s.update(elements=['Fe', 'Co'], concentrations_mM={'Fe': .04 + i * .001, 'Co': .06})
            samples.append(s)
        state = self.state(samples)
        self.assertTrue(state.info['unseen_target'])
        self.assertFalse(state.model.calibration_mask(np.array([transfer_feature(context())]))[0])
        checks = state.info['expansion_validation']
        self.assertIsNotNone(checks)
        for fold in checks['folds']:
            self.assertLess(max(fold['train_stages']), min(fold['test_stages']))
            self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))

    def test_dynamic_policy_responds_to_new_target_and_requested_strategy(self):
        state = self.state()
        summary = {'bald': np.array([.1, .2])}
        novel = dynamic_policy(state, [], None, {}, summary, [])
        state.info['unseen_target'] = False
        state.info['unseen_elements'] = []
        observed = dynamic_policy(state, [], None, {}, summary, [])
        self.assertGreater(novel['information_weight'], observed['information_weight'])
        success = dynamic_policy(state, [], None, {'acquisition_policy': 'success'}, summary, [])
        self.assertLess(success['information_weight'], observed['information_weight'])

    def test_recommendations_respect_bounds_exclude_duplicates_and_preserve_pending_metadata(self):
        samples, ctx, operating = self.sources(), context(), space()
        state = self.state(samples, ctx)
        rows, note = recommend(state, samples, ctx, operating, 'REAL', 6)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({str(r['recipe']) for r in rows}), 6)
        self.assertIn('joint posterior', note)
        for row in rows:
            self.assertEqual(row['recipe']['elements'], ['Co'])
            self.assertIsNone(row['recipe']['temperature_C'])
            self.assertLessEqual(row['recipe']['concentrations_mM']['Co'], .2)
            self.assertGreaterEqual(row['recipe']['concentrations_mM']['Co'], .01)
            self.assertEqual(row['calibration_state'], 'UNCALIBRATED_TARGET')
            self.assertGreaterEqual(row['p_screen_high'], row['p_screen_low'])
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, 'REAL')
            from sacfactory.planning import fingerprint
            snapshot = fingerprint({'context': ctx, 'space': operating, 'samples': [], 'methods': {}, 'rule_version': RULE_VERSION})
            for row in rows:
                row['snapshot'] = snapshot
            ids = store.enqueue_pending(rows, ctx, operating)
            saved = store.get('sample', ids[0])
            self.assertEqual(screen(saved, {})['status'], 'PENDING')
            self.assertEqual(saved['proposal']['policy'], 'ADAPTIVE_JOINT_INFORMATION')
            self.assertEqual(saved['proposal']['acquisition'], rows[0]['acquisition'])
            self.assertEqual(saved['proposal']['p_screen_low'], rows[0]['p_screen_low'])
        self.assertIn('information_weight', recommendation_table(rows)[0])

    def test_new_data_changes_model_and_design_only_never_invents_probability(self):
        a = self.state()
        changed = self.sources()
        changed[0]['icp_review']['state'] = 'PASS'
        b = self.state(changed)
        self.assertNotEqual(a.info['model_id'], b.info['model_id'])
        insufficient = self.state([source(i) for i in range(8)])
        self.assertIsNone(insufficient.model)
        rows, _ = recommend(insufficient, [], context(), space(), 'REAL', 2)
        self.assertTrue(all(r['p_screen_pass'] is None for r in rows))

    def test_burnout_namespace_and_missing_controls_do_not_create_false_labels(self):
        records = self.sources()
        burnout = source(100)
        burnout.update(sample_condition='BURNED_OUT', icp_review={'state': 'PENDING'}, xrd={'state': 'PENDING'})
        records += [burnout, source(101)]
        records[-1]['origin'] = 'DEMO'
        missing = source(102)
        missing['pulse_limit_C'] = None
        records.append(missing)
        state = self.state(records)
        self.assertEqual(state.info['n_labeled'], 25)
        self.assertIn('Different data namespace', state.info['exclusions'])
        self.assertIn('Different composition or process cohort', state.info['exclusions'])

    def test_resolved_prior_forecast_errors_increase_information_weight(self):
        from unittest.mock import patch
        import pandas as pd
        state = self.state()
        summary = {'bald': np.array([.1])}
        base = dynamic_policy(state, [], {}, {}, summary, [])
        forecasts = pd.DataFrame({'eligible': [True]*6, 'composition': ['Co']*6,
            'result_reported_at': ['2026-10-10T00:00:00Z']*6, 'label': [1]*6,
            'p_screen_pass': [.1]*6, 'batch_id': ['B1']*3+['B2']*3})
        with patch('sacfactory.analysis_extended.prediction_records', return_value=forecasts):
            updated = dynamic_policy(state, [], {}, {}, summary, [])
        self.assertGreater(updated['information_weight'], base['information_weight'])
        self.assertAlmostEqual(updated['recent_forecast_brier'], .81)
        self.assertEqual(updated['recent_forecast_n'], 6)


if __name__ == '__main__':
    unittest.main()
