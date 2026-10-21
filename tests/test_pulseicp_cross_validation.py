"""Whole-batch rotation, endpoint coverage, leakage and report isolation tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sacfactory import pulseicp_bridge as bridge
from sacfactory.pulseicp_cross_validation import batch_groups, load_matching_report
from test_pulseicp_bridge import specimens


def budget():
    config = bridge.configuration()
    config['model'].update(members=1, epochs_baseline=2, epochs_diffusion=2,
        draws_per_member=4, diffusion_steps=10)
    return config


class RotatingICPTests(unittest.TestCase):
    def test_batch_groups_cover_every_batch_once_without_label_access(self):
        for per_fold in (1, 2, 4):
            groups = batch_groups(['A','B','C','D','E','F'], per_fold, 42)
            flat = [b for group in groups for b in group]
            self.assertEqual(len(flat), 6)
            self.assertEqual(len(set(flat)), 6)
            self.assertEqual(groups, batch_groups(list(reversed(flat)), per_fold, 42))
        for invalid in (True, 1.5, 0, 5):
            with self.assertRaises(ValueError):
                batch_groups(['A','B','C','D','E','F'], invalid, 42)
        with self.assertRaises(ValueError):
            batch_groups(['A','B'], 1, 42)

    def test_two_batch_rotation_has_unique_complete_oof_points_and_pooled_metrics(self):
        from sacfactory._pulseicp.uncertainty import metrics
        rows = specimens(); rows[0]['icp'] = []
        before = copy.deepcopy(rows); config = budget(); progress = []
        with tempfile.TemporaryDirectory() as root:
            pointer = Path(root)/'latest.json'
            pointer.write_text('{"run_id":"unchanged-production-checkpoint"}', encoding='utf-8')
            original_pointer = pointer.read_bytes()
            destination, points, comparison, info = bridge.validate(rows,'REAL',config['model'],config['physics'],root,
                protocol=bridge.ROTATING_CV,batches_per_fold=2,progress=lambda a,b:progress.append((a,b)))
            self.assertEqual(info['status'],'COMPLETE')
            self.assertEqual(info['recorded_n'],60)
            self.assertEqual(info['eligible_n'],59)
            self.assertEqual(info['evaluated_specimens_n'],59)
            self.assertEqual(info['coverage_fraction'],1.)
            self.assertEqual(len(points),59)
            self.assertFalse(points.duplicated(['sample_id','element']).any())
            self.assertEqual(set(points.model),{'Gaussian residual'})
            self.assertEqual(progress,[(1,3),(2,3),(3,3)])
            seen = []
            for part in info['partitions']:
                self.assertFalse(set(part['train_sample_ids']) & set(part['test_sample_ids']))
                self.assertFalse(set(part['train_batch_ids']) & set(part['test_batch_ids']))
                seen += part['test_sample_ids']
                for member in part['bootstrap']:
                    self.assertTrue(set(member['sample_ids_with_replacement']) <= set(part['train_sample_ids']))
                    self.assertFalse(set(member['batch_ids_with_replacement']) & set(part['test_batch_ids']))
            self.assertEqual(len(seen),59)
            self.assertEqual(len(set(seen)),59)
            expected = metrics(points)
            for key in ('mae_wt_percent','rmse_wt_percent','r_squared'):
                self.assertAlmostEqual(comparison.iloc[0][key],expected[key])
            self.assertEqual(pointer.read_bytes(),original_pointer)
            self.assertFalse(any(destination.rglob('*.pt')))
            audit = __import__('pandas').read_csv(destination/'eligibility.csv')
            self.assertEqual(int(audit.evaluated.sum()),59)
            context = list(__import__('sacfactory.composition',fromlist=['process_key']).process_key(rows[0]))
            loaded = load_matching_report(root,bridge.input_frame(rows,'REAL',config['physics']),'EXPERIMENTAL',
                config['physics'],config['model'],'Gaussian residual',2,context)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded[1]),59)
            changed = copy.deepcopy(rows);changed[1]['concentrations_mM']['Fe'] *= 2
            self.assertIsNone(load_matching_report(root,bridge.input_frame(changed,'REAL',config['physics']),'EXPERIMENTAL',
                config['physics'],config['model'],'Gaussian residual',2,context))
            self.assertIsNone(load_matching_report(root,bridge.input_frame(rows,'REAL',config['physics']),'EXPERIMENTAL',
                config['physics'],config['model'],'Conditional diffusion',2,context))
            with (destination/'test_predictions.csv').open('a',encoding='utf-8') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError,'integrity'):
                load_matching_report(root,bridge.input_frame(rows,'REAL',config['physics']),'EXPERIMENTAL',
                    config['physics'],config['model'],'Gaussian residual',2,context)
            json.dumps(info,allow_nan=False)
        self.assertEqual(rows,before)

    def test_withheld_outcome_changes_cannot_change_its_own_fold_predictions(self):
        rows = specimens(); config = budget()
        changed = copy.deepcopy(rows)
        for row in changed:
            if row['batch_id'] == 'BATCH_0006':
                row['icp'][0]['value'] = str(float(row['icp'][0]['value'])*.8)
        with tempfile.TemporaryDirectory() as root:
            first = bridge.validate(rows,'REAL',config['model'],config['physics'],Path(root)/'a',
                protocol=bridge.ROTATING_CV,batches_per_fold=2)[1]
            second = bridge.validate(changed,'REAL',config['model'],config['physics'],Path(root)/'b',
                protocol=bridge.ROTATING_CV,batches_per_fold=2)[1]
        a = first.loc[first.batch_id=='BATCH_0006'].sort_values('sample_id')
        b = second.loc[second.batch_id=='BATCH_0006'].sort_values('sample_id')
        np.testing.assert_array_equal(a.predicted_mean_w_wt_percent,b.predicted_mean_w_wt_percent)
        self.assertTrue(np.all(a.observed_w_wt_percent.to_numpy() != b.observed_w_wt_percent.to_numpy()))

    def test_single_batch_rotation_and_diffusion_cover_all_targets(self):
        rows = specimens(); config = budget()
        with tempfile.TemporaryDirectory() as root:
            _, points, comparison, info = bridge.validate(rows,'REAL',config['model'],config['physics'],root,
                protocol=bridge.ROTATING_CV,family='Conditional diffusion',batches_per_fold=1)
        self.assertEqual(len(info['partitions']),6)
        self.assertEqual(len(points),60)
        self.assertEqual(len(comparison),1)
        self.assertEqual(set(points.model),{'Conditional diffusion'})
        self.assertTrue(np.isfinite(points.predicted_mean_w_wt_percent).all())
        self.assertTrue((points.calibration_status=='PROVISIONAL_UNCALIBRATED').all())

    def test_unevaluable_folds_stay_explicit_without_imputing_predictions(self):
        rows = specimens(24); config = budget()
        with tempfile.TemporaryDirectory() as root:
            destination, points, _, info = bridge.validate(rows,'REAL',config['model'],config['physics'],root,
                protocol=bridge.ROTATING_CV,batches_per_fold=1)
            audit = __import__('pandas').read_csv(destination/'eligibility.csv')
        self.assertEqual(info['status'],'PARTIAL_COVERAGE')
        self.assertEqual(len(info['skipped_folds']),2)
        self.assertEqual(len(points),4)
        self.assertEqual(int(audit.evaluated.sum()),4)
        self.assertEqual(int(audit.eligible.sum()),24)

    def test_complete_mixtures_stay_together_and_receive_one_prediction_per_element(self):
        from sacfactory._pulseicp.composition import decode_composition
        from sacfactory.domain import validate_sample
        rows = specimens(); config = budget()
        for row in rows:
            row.update(elements=['Fe','Co'],concentrations_mM={'Fe':row['concentrations_mM']['Fe'],'Co':20.})
            values = decode_composition(bridge.composition(row),{'Fe':-.2,'Co':.1},config['physics'])
            row['icp'] = [{'element':e,'value':str(float(v)),'unit':'wt%','flag':'MEASURED'} for e,v in values.items()]
            validate_sample(row)
        with tempfile.TemporaryDirectory() as root:
            _, points, comparison, info = bridge.validate(rows,'REAL',config['model'],config['physics'],root,
                protocol=bridge.ROTATING_CV,batches_per_fold=2)
        self.assertEqual(len(points),120)
        self.assertEqual(info['evaluated_specimens_n'],60)
        self.assertEqual(int(comparison.iloc[0].specimens_n),60)
        self.assertTrue((points.groupby('sample_id').fold.nunique()==1).all())
        self.assertFalse(points.duplicated(['sample_id','element']).any())


if __name__ == '__main__':
    unittest.main()
