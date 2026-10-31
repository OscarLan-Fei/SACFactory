"""Scientific safeguards for synthesis dashboard and forward evaluation."""
import copy
import json
from pathlib import Path
import unittest
import tempfile

import numpy as np

from sacfactory.analysis_extended import (analysis_records, calibration_table, composition_coverage,
    conditional_surface, failure_breakdown, gate_summary, interaction_rates, learning_progress,
    prediction_records, probability_metrics, repeatability, strategy_comparison, transfer_comparison)
from sacfactory.domain import RULE_VERSION, recipe_key, screen
from test_science import sample, method


class ExtendedAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = json.loads((Path(__file__).resolve().parents[1] / 'examples/demo_seed.json').read_text())['samples']

    def forecast(self):
        s = sample()
        s['result_reported_at'] = '2026-01-02T12:00:00Z'
        s['proposal'] = {'recipe': copy.deepcopy(s), 'p_screen_pass': .8, 'created': '2026-01-02T00:00:00Z', 'model_id': 'M_TEST', 'evaluation_capture': {'rule_version': RULE_VERSION, 'training_batch_ids': ['BEFORE'], 'training_records': [{'sample_id': 'OTHER'}]}}
        return s

    def test_prediction_matching_scores_only_unseen_batches_and_unchanged_recipes(self):
        good = self.forecast()
        data = prediction_records([good], method())
        self.assertTrue(data.iloc[0]['eligible'])
        self.assertAlmostEqual(probability_metrics(data)['brier'], .04)
        for change in ('stale', 'recipe', 'batch', 'sample', 'time', 'rule', 'namespace', 'no_capture', 'conflict'):
            bad = copy.deepcopy(good)
            if change == 'stale': bad['proposal']['prediction_stale'] = True
            if change == 'recipe': bad['concentrations_mM']['Fe'] = .1
            if change == 'batch': bad['proposal']['evaluation_capture']['training_batch_ids'] = [bad['batch_id']]
            if change == 'sample': bad['proposal']['evaluation_capture']['training_records'] = [{'sample_id': bad['sample_id']}]
            if change == 'time': bad['result_reported_at'] = '2026-01-01T12:00:00Z'
            if change == 'rule': bad['proposal']['evaluation_capture']['rule_version'] = 'OLD'
            if change == 'namespace': bad['proposal']['recipe']['origin'] = 'DEMO'
            if change == 'no_capture': bad['proposal'].pop('evaluation_capture')
            if change == 'conflict': bad['icp_review']['state'] = 'FAIL'
            self.assertFalse(prediction_records([bad], method()).iloc[0]['eligible'], change)

    def test_stable_record_identity_survives_display_id_changes(self):
        s = self.forecast()
        s['record_uid'] = 'PHYSICAL_ONE'
        s['proposal']['evaluation_capture']['training_records'] = [{'sample_id': 'OLDER_ID', 'record_uid': 'PHYSICAL_ONE'}]
        self.assertFalse(prediction_records([s], method()).iloc[0]['eligible'])

    def test_unresolved_and_missing_forecasts_are_visible_but_not_scored(self):
        s = self.forecast()
        s['xrd']['state'] = 'PENDING'
        data = prediction_records([s], method())
        self.assertFalse(data.iloc[0]['eligible'])
        self.assertTrue(calibration_table(data).empty)
        self.assertTrue(prediction_records([sample()], method()).empty)

    def test_calibration_uses_saved_probabilities_and_explicit_denominators(self):
        a, b = self.forecast(), self.forecast()
        b.update(sample_id='S1', batch_id='B1')
        b['icp'][0]['value'] = '1'
        b['proposal']['recipe'] = copy.deepcopy(b['proposal']['recipe'])
        table = calibration_table(prediction_records([a, b], method()))
        self.assertEqual(table['n'].sum(), 2)
        self.assertAlmostEqual(table.iloc[0]['observed_pass_rate'], .5)
        self.assertEqual(table.iloc[0]['independent_batches'], 2)
        self.assertEqual(probability_metrics(prediction_records([a], method()))['balanced_accuracy'], None)

    def test_repeatability_separates_same_batch_from_independent_batches(self):
        samples = [sample(i) for i in range(3)]
        samples[1]['batch_id'] = samples[0]['batch_id']
        samples[2]['sample_condition'] = 'BURNED_OUT'
        table = repeatability(analysis_records(samples, method()))
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]['independent_batches'], 2)
        self.assertEqual(table.iloc[0]['same_batch_extra_records'], 1)
        self.assertAlmostEqual(table.iloc[0]['pass_rate'], 2 / 3)
        self.assertAlmostEqual(table.iloc[0]['equal_batch_pass_rate'], .5)
        self.assertEqual(table.iloc[0]['burned_out'], 1)

    def test_failure_categories_never_infer_burnout_from_both_fail(self):
        a, b, c = [sample(i) for i in range(3)]
        a['icp'][0]['value'] = '1'; a['xrd']['state'] = 'FAIL'
        b['sample_condition'] = 'BURNED_OUT'
        c['sample_condition'] = 'ACCIDENTAL_LOSS'
        table = failure_breakdown(analysis_records([a, b, c], method()))
        self.assertEqual(set(table['category']), {'BOTH_FAIL', 'BURNED_OUT', 'ACCIDENTAL_LOSS'})

    def test_readiness_mirrors_existing_model_gate_without_imputation(self):
        context = self.demo[0]
        detail, info = gate_summary(self.demo, {}, context, 'DEMO')
        self.assertGreater(info['eligible'], 12)
        self.assertEqual(detail['eligible'].sum(), info['eligible'])
        wrong_origin, stats = gate_summary(self.demo, {}, context, 'REAL')
        self.assertEqual(stats['eligible'], 0)
        incomplete = sample()
        incomplete['concentrations_mM']['Fe'] = None
        detail, stats = gate_summary([incomplete], method(), incomplete, 'REAL')
        self.assertFalse(stats['ready'])
        self.assertIn('Incomplete synthesis', detail.iloc[0]['reason'])
        from sacfactory.planning import fit_model
        from sacfactory.manual import actual_batch_recorded
        unassigned = sample()
        unassigned['batch_id'] = 'UNASSIGNED'
        self.assertFalse(actual_batch_recorded(unassigned))
        for transfer in (False, True):
            self.assertEqual(fit_model([unassigned], method(), unassigned, 'REAL', transfer).info['n_labeled'], 0)

    def test_surface_is_gated_conditional_and_has_bounded_probabilities(self):
        frame, info, anchor = conditional_surface([sample()], method(), sample(), 'REAL', 'Fe', 1)
        self.assertTrue(frame.empty)
        target = self.demo[0]
        frame, info, anchor = conditional_surface(self.demo, {}, target, 'DEMO', 'Fe', 1, resolution=5)
        self.assertFalse(frame.empty)
        self.assertTrue(frame['p_screen_pass'].between(0, 1).all())
        self.assertTrue(frame['predictive_entropy_bits'].between(0, 1).all())
        self.assertTrue((frame['nearest_training_distance'] >= 0).all())
        self.assertEqual(set(frame['pulse_count']), {1})

    def test_transfer_comparison_pairs_identical_test_rows_and_purges_groups(self):
        context = next(s for s in self.demo if len(s['elements']) == 2)
        paired_pool=[s for s in self.demo if s['elements'] in [['Fe'],['Co'],context['elements']]]
        frame, info = transfer_comparison(paired_pool, {}, context, 'DEMO')
        self.assertFalse(frame.empty)
        lookup = {s['sample_id']: s for s in self.demo}
        for row in frame.to_dict('records'):
            test_rows = frame.loc[frame['fold'] == row['fold'], 'sample_id']
            test_batches = {lookup[i]['batch_id'] for i in test_rows}
            test_recipes = {recipe_key(lookup[i]) for i in test_rows}
            for column in ('target_train_ids', 'transfer_train_ids'):
                train = [lookup[i] for i in row[column].split(';')]
                self.assertFalse(test_batches & {s['batch_id'] for s in train})
                self.assertFalse(test_recipes & {recipe_key(s) for s in train})
        self.assertEqual(frame['p_target_only'].count(), frame['p_transfer'].count())

    def test_periodic_coverage_and_pairs_do_not_create_element_success_labels(self):
        elements, pairs, stages = composition_coverage(self.demo, {})
        self.assertEqual(len(elements), 118)
        self.assertEqual(int(elements.loc[elements['element'] == 'Fe', 'single_element_records'].iloc[0]), 60)
        self.assertEqual(stages['records'].sum(), 1476)
        self.assertIn(50, stages['element_count'].tolist())
        self.assertTrue((pairs['resolved_records'] > 0).all())

    def test_progress_requires_timestamps_and_never_sorts_sample_ids_as_time(self):
        self.assertTrue(learning_progress([sample()], method()).empty)
        first, last = sample(9), sample(0)
        first['result_reported_at'] = '2026-01-01T12:00:00Z'
        last['result_reported_at'] = '2026-01-03T12:00:00Z'
        frame = learning_progress([last, first], method())
        self.assertEqual(frame['batch_id'].tolist(), ['B9', 'B0'])
        self.assertEqual(frame['cumulative_resolved_experiments'].tolist(), [1, 2])

    def test_interaction_bins_exclude_unresolved_and_do_not_fill_empty_cells(self):
        a, b = sample(), sample(1, alpha=2)
        a['pulse_limit_C'] = 900; b['pulse_limit_C'] = 1100
        b['xrd']['state'] = 'PENDING'
        frame = interaction_rates(analysis_records([a, b], method()), 'Fe_mM', 'pulse_limit_C')
        self.assertEqual(frame['n'].sum(), 1)
        self.assertEqual(len(frame), 1)

    def test_demo_is_explicitly_synthetic_with_audited_forward_predictions(self):
        data = prediction_records(self.demo, {})
        valid = data.loc[data['eligible']]
        self.assertGreater(len(valid), 80)
        lookup = {s['sample_id']: s for s in self.demo}
        for ident in valid['sample_id']:
            s = lookup[ident]
            self.assertTrue(all(lookup[ref['sample_id']]['demo_sequence'] < s['demo_sequence'] for ref in s['proposal']['evaluation_capture']['training_records']))
        self.assertEqual({s['origin'] for s in self.demo}, {'DEMO'})
        self.assertTrue(all('SIMULATED' in s['notes'] for s in self.demo))

    def test_demo_reload_cannot_write_real_or_reseed_on_normal_startup(self):
        from sacfactory.bootstrap import replace_demo, seed
        from sacfactory.storage import Store
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            real = Store(folder, 'REAL')
            real.import_bundle([sample()], list(method().values()))
            before = real.export()
            with self.assertRaises(ValueError):
                replace_demo(real, root)
            self.assertEqual(before, real.export())
            demo = Store(folder, 'DEMO')
            self.assertEqual(replace_demo(demo, root), 1476)
            seed(demo, root)
            self.assertEqual(len(demo.all('sample')), 1476)
            self.assertEqual(before, real.export())

    def test_strategy_comparison_uses_disjoint_batches_and_resolved_denominators(self):
        a, b, pending = sample(), sample(1), sample(2)
        b['sample_condition'] = 'BURNED_OUT'
        pending['batch_id'] = b['batch_id']; pending['xrd']['state'] = 'PENDING'
        frame = analysis_records([a, b, pending], method())
        result = strategy_comparison(frame, ['B0'], ['B1'], 'Random design')
        self.assertEqual(result['resolved'].tolist(), [1, 1])
        self.assertEqual(result['pass_rate'].tolist(), [1., 0.])
        self.assertEqual(result['recorded_samples'].tolist(), [1, 2])
        with self.assertRaises(ValueError):
            strategy_comparison(frame, ['B0'], ['B0'], 'Random design')
        frame.loc[0, 'cohort'] = 'OTHER_PROCESS'
        with self.assertRaises(ValueError):
            strategy_comparison(frame, ['B0'], ['B1'], 'Random design')

    def test_recommend_pending_feedback_captures_prediction_for_evaluation(self):
        from sacfactory.storage import Store
        from sacfactory.planning import fit_model, recommend, fingerprint
        from sacfactory.manual import save_manual_revisions
        training = [copy.deepcopy(s) for s in self.demo if s['elements']==['Fe']]
        context = copy.deepcopy(training[0])
        operating = {'concentration_bounds_mM': {'Fe': {'min_mM': 15., 'max_mM': 150.}}, 'pulses': [1, 2, 3], 'max_total_mM': 150., 'pulse_limit_min_C': 830, 'pulse_limit_max_C': 2000, 'verified': True, 'range_source': 'Simulated device range', 'baseline_repeat': False}
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder, 'DEMO')
            store.import_bundle(training, [])
            saved = store.all('sample')
            state = fit_model(saved, {}, context, 'DEMO')
            rows, note = recommend(state, saved, context, operating, 'DEMO', 1)
            self.assertIsNotNone(rows[0]['p_screen_pass'])
            rows[0]['snapshot'] = fingerprint({'context': context, 'space': operating, 'samples': saved, 'methods': {}, 'rule_version': RULE_VERSION})
            ident = store.enqueue_pending(rows, context, operating)[0]
            pending = store.get('sample', ident)
            self.assertEqual(prediction_records([pending], {}).iloc[0]['exclusion_reason'].count('Unresolved'), 1)
            report = copy.deepcopy(pending)
            report.update(batch_id='NEW_SYNTHESIS_BATCH', sample_condition='BURNED_OUT')
            save_manual_revisions(store, [report], {}, 'feedback', {ident: fingerprint(pending)})
            completed = store.get('sample', ident)
            self.assertTrue(prediction_records([completed], {}).iloc[0]['eligible'])
            self.assertEqual(prediction_records([completed], {}).iloc[0]['label'], 0)


if __name__ == '__main__':
    unittest.main()
