"""Auxiliary regression target integrity and protected validation tests."""
import copy
import unittest

import numpy as np
import pandas as pd

from sacfactory.loading import loading_records, loading_validation, MODELS, regression_metrics
from sacfactory.chemical_linking import link_declared_chlorides
from sacfactory.descriptor_analysis import association_availability, descriptor_report
from sacfactory.precursors import precursor_coverage
from sacfactory.probability_evaluation import probability_evaluation
from test_science import sample


def campaign():
    rows = []
    for batch in range(6):
        for i in range(4):
            row = sample(batch * 4 + i, alpha=.5 + i * .6 + batch * .025)
            row.update(batch_id=f'B{batch}', pulse_limit_C=850 + i * 80,
                       result_reported_at=f'2026-01-{batch+1:02d}T12:00:00Z')
            row['icp'][0]['value'] = str(.02 + .15 * row['concentrations_mM']['Fe'] + i * .03)
            rows.append(row)
    return rows


class LoadingTests(unittest.TestCase):
    def test_quantified_targets_exclude_censoring_burnout_ambiguity_and_other_namespace(self):
        rows = campaign()[:7]
        rows[0]['icp'][0]['value'] = '0'
        rows[1]['icp'][0]['value'] = '<0.05'
        rows[2]['icp'][0]['value'] = 'ND'
        rows[3]['sample_condition'] = 'BURNED_OUT'
        rows[4]['icp'].append(copy.deepcopy(rows[4]['icp'][0]))
        rows[5]['origin'] = 'DEMO'
        rows[6]['icp'][0]['flag'] = 'BLQ'
        before = copy.deepcopy(rows)
        eligible, audit = loading_records(rows, 'REAL', 'Fe')
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0][1], 0)
        self.assertEqual(len(audit), 6)
        self.assertEqual(audit.loc[audit.sample_id == rows[1]['sample_id'], 'censor_upper_w_wt_percent'].iloc[0], .05)
        self.assertEqual(rows, before)

    def test_batch_splits_all_models_nonnegative_predictions_and_units(self):
        rows = campaign()
        for model in MODELS:
            predictions, audit, info = loading_validation(rows, 'REAL', 'Fe', rows[0], family=model)
            self.assertEqual(len(predictions), 24)
            self.assertTrue(predictions.predicted_w_wt_percent.between(0, 100).all())
            self.assertEqual(info['metrics']['n'], 24)
            for fold in info['folds']:
                self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))
            if model != 'Gaussian process':
                self.assertTrue(predictions.lower_95_w_wt_percent.isna().all())
            self.assertTrue(audit.eligible.all())

    def test_heldout_target_and_retrospective_readings_cannot_influence_its_prediction(self):
        rows = campaign()
        first, _, info = loading_validation(rows, 'REAL', 'Fe', rows[0])
        fold = info['folds'][0]
        changed = copy.deepcopy(rows)
        for row in changed:
            row['temperature_C'], row['voltage_V'], row['current_A'] = 1500, 99, 300
            if row['sample_id'] in fold['test_sample_ids']:
                row['icp'][0]['value'] = '88'
                row['xrd']['state'] = 'FAIL'
        second, _, _ = loading_validation(changed, 'REAL', 'Fe', changed[0])
        np.testing.assert_allclose(first.loc[first.fold == 1, 'predicted_w_wt_percent'], second.loc[second.fold == 1, 'predicted_w_wt_percent'])

    def test_temporal_and_single_to_mixture_protocols_exclude_test_batches(self):
        rows = campaign()
        predictions, _, info = loading_validation(rows, 'REAL', 'Fe', rows[0], protocol='Chronological replay', max_folds=6)
        self.assertFalse(predictions.empty)
        lookup = {r['sample_id']: r for r in rows}
        for fold in info['folds']:
            self.assertLess(max(lookup[i]['result_reported_at'] for i in fold['train_sample_ids']), min(lookup[i]['result_reported_at'] for i in fold['test_sample_ids']))
        for row in rows[12:]:
            row['elements'] = ['Fe', 'Co']
            row['concentrations_mM']['Co'] = .05
        predictions, _, info = loading_validation(rows, 'REAL', 'Fe', rows[0], protocol='Single to multiple elements')
        self.assertEqual(len(predictions), 12)
        self.assertTrue((predictions.element_count == 2).all())
        self.assertTrue(all(len(lookup[i]['elements']) == 1 for i in info['folds'][0]['train_sample_ids']))

    def test_new_recipe_purge_and_missing_metadata_are_explicit(self):
        rows = campaign()
        for row in rows:
            row['concentrations_mM']['Fe'] = .05 + (int(row['sample_id'][1:]) % 4) * .01
        predictions, _, info = loading_validation(rows, 'REAL', 'Fe', rows[0], protocol='New recipe holdout')
        self.assertTrue(predictions.empty)
        self.assertEqual(len(info['skipped_folds']), 3)
        rows[0]['batch_id'] = 'UNKNOWN'
        rows[1]['pulse_limit_C'] = None
        _, audit = loading_records(rows, 'REAL', 'Fe')
        self.assertFalse(audit.eligible.iloc[0])
        self.assertFalse(audit.eligible.iloc[1])

    def test_metrics_do_not_invent_constant_target_r_squared_or_intervals(self):
        frame = pd.DataFrame({'observed_w_wt_percent': [.1, .1], 'predicted_w_wt_percent': [.1, .2],
                              'lower_95_w_wt_percent': [None, None], 'upper_95_w_wt_percent': [None, None]})
        score = regression_metrics(frame)
        self.assertAlmostEqual(score['mae_wt_percent'], .05)
        self.assertIsNone(score['r_squared'])
        self.assertIsNone(score['interval_coverage'])

    def test_chloride_linking_is_explicit_idempotent_and_source_gated(self):
        row = sample()
        row['elements'] = ['Fe', 'Cu', 'In', 'Au', 'Co']
        row['concentrations_mM'] = {e: .05 for e in row['elements']}
        library = {e: {'compound': formula, 'source': 'User declaration', 'type': 'CHLORIDE'}
                   for e, formula in [('Fe', 'FeCl2'), ('Cu', 'CuCl2'), ('In', 'InCl3'), ('Au', 'AuCl3')]}
        library['Co'] = {'compound': 'CoCl2', 'source': 'PLANNED_CATALOG_ASSUMPTION: Sigma'}
        before = copy.deepcopy(row)
        updated, _ = link_declared_chlorides([row], library, 'REAL', square_planar=True)
        self.assertEqual(row, before)
        self.assertNotIn('Co', updated[0]['precursors'])
        coverage = {r['element']: r for r in precursor_coverage(updated[0])}
        self.assertEqual(coverage['Cu']['oxidation_state'], 2)
        self.assertAlmostEqual(coverage['Cu']['ionic_radius_pm'], 57.)
        self.assertEqual(coverage['Au']['ionic_radius_pm'], 68.)
        self.assertIsNone(coverage['Fe']['ionic_radius_pm'])
        self.assertIsNone(coverage['In']['ionic_radius_pm'])
        self.assertEqual(link_declared_chlorides(updated, library, 'REAL', True)[0], [])
        self.assertEqual(link_declared_chlorides([row], library, 'DEMO', True)[0], [])
        states = descriptor_report(updated, {}, 'REAL')['members']
        self.assertEqual(states.formal_oxidation_state.notna().sum(), 4)
        library['Fe'].update(coordination='IVSQ', spin='HS', state_source='MODEL_ASSUMPTION: User selected HS reference')
        linked, _ = link_declared_chlorides(updated, library, 'REAL', True)
        fe = next(r for r in precursor_coverage(linked[0]) if r['element'] == 'Fe')
        self.assertEqual(fe['ionic_radius_pm'], 64.)
        self.assertTrue(fe['state_source'].startswith('MODEL_ASSUMPTION:'))

    def test_availability_explains_missing_constant_and_small_sample_variables(self):
        frame = pd.DataFrame({'formal_oxidation_state_mean': [None]*3, 'ionic_radius_pm_mean': [57.]*3,
                              'few': [1., 2., None], 'good': [1., 2., 3.]})
        audit = association_availability(frame, list(frame))
        self.assertEqual(audit.status.tolist(), ['No recorded values', 'Constant values', 'Fewer than three observations', 'Available'])

    def test_probability_metrics_use_eligible_rows_and_correct_false_pass_denominator(self):
        records = pd.DataFrame({'eligible': [True, True, True, False], 'label': [1, 0, 0, 1], 'p_screen_pass': [.8, .7, .1, .9]})
        scores, curve = probability_evaluation(records)
        self.assertEqual(scores['n'], 3)
        self.assertEqual(scores['false_pass_fraction'], .5)
        self.assertEqual(scores['pass_recall'], 1.)
        self.assertFalse(curve.empty)
        records.loc[records.eligible, 'label'] = 1
        scores, curve = probability_evaluation(records)
        self.assertIsNone(scores['roc_auc'])
        self.assertTrue(curve.empty)
