"""Manual intervention forecasts preserve history and reject result leakage."""
import copy
import tempfile
import unittest
from unittest.mock import patch

from sacfactory.analysis_extended import prediction_records
from sacfactory.manual import save_manual_revisions
from sacfactory.pending_prediction import refresh_pending_prediction
from sacfactory.planning import fingerprint
from sacfactory.storage import Store
from test_chemistry import campaign


class PendingPredictionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(self.tmp.name, 'REAL')
        self.rows = campaign()[:36]
        self.store.import_bundle(self.rows, [])
        self.sample = copy.deepcopy(self.rows[0])
        self.sample.update(sample_id='SAC_1000', batch_id='PLANNED_NAMED_BATCH', icp=[],
            icp_review={'state': 'PENDING'}, xrd={'state': 'PENDING'}, temperature_C=None,
            voltage_V=None, current_A=None, sample_condition='NOT_RECORDED')
        self.sample.pop('result_reported_at', None)
        self.store.import_bundle([self.sample], [])
        self.store.put_many('meta', [{'id': 'space', 'origin': 'REAL', 'space': {
            'model_engine': 'adaptive', 'descriptor_profile': 'chemistry_thermal',
            'learning_iterations': 2, 'element_transfer': True}}], 'id')

    def tearDown(self):
        self.tmp.cleanup()

    def test_named_unmeasured_pending_can_be_edited_and_repredicted_then_validated(self):
        original = self.store.get('sample', self.sample['sample_id'])
        edited = copy.deepcopy(original)
        edited.update(pulse_count=3, pulse_limit_C=1200.)
        edited['concentrations_mM']['Fe'] = .08
        save_manual_revisions(self.store, [edited], {}, 'pending', {original['sample_id']: fingerprint(original)})
        edited = self.store.get('sample', original['sample_id'])
        forecast = refresh_pending_prediction(self.store, original['sample_id'], fingerprint(edited))
        self.assertTrue(0 <= forecast['p_screen_pass'] <= 1)
        self.assertEqual(forecast['policy'], 'HUMAN_ADJUSTED_RECIPE')
        self.assertNotIn(original['sample_id'], {r['sample_id'] for r in forecast['evaluation_capture']['training_records']})
        stored = self.store.get('sample', original['sample_id'])
        self.assertEqual(len(stored['manual_prediction_history']), 1)
        returned = copy.deepcopy(stored)
        returned.update(icp_review={'state': 'PASS'}, xrd={'state': 'PASS', 'mode': 'MANUAL'})
        save_manual_revisions(self.store, [returned], {}, 'feedback', {stored['sample_id']: fingerprint(stored)})
        result = self.store.get('sample', original['sample_id'])
        evaluated = prediction_records([result], {})
        self.assertTrue(evaluated.eligible.iloc[0], evaluated.exclusion_reason.iloc[0])
        self.assertEqual(evaluated.policy.iloc[0], 'HUMAN_ADJUSTED_RECIPE')
        with self.assertRaisesRegex(ValueError, 'unmeasured'):
            refresh_pending_prediction(self.store, original['sample_id'], fingerprint(result))

    def test_later_edit_invalidates_new_prediction_but_history_is_immutable(self):
        original = self.store.get('sample', self.sample['sample_id'])
        refresh_pending_prediction(self.store, original['sample_id'], fingerprint(original))
        old = self.store.get('sample', original['sample_id'])
        edited = copy.deepcopy(old)
        edited['pulse_count'] += 1
        save_manual_revisions(self.store, [edited], {}, 'pending', {old['sample_id']: fingerprint(old)})
        current = self.store.get('sample', old['sample_id'])
        self.assertTrue(current['manual_prediction']['prediction_stale'])
        self.assertFalse(current['manual_prediction_history'][0]['prediction_stale'])
        refresh_pending_prediction(self.store, current['sample_id'], fingerprint(current))
        current = self.store.get('sample', old['sample_id'])
        self.assertFalse(current['manual_prediction']['prediction_stale'])
        self.assertEqual(len(current['manual_prediction_history']), 2)

    def test_concurrent_settings_change_rejects_forecast_without_partial_save(self):
        from sacfactory.pending_prediction import fit_model
        sample = self.store.get('sample', self.sample['sample_id'])
        def changed(*args, **kwargs):
            model = fit_model(*args, **kwargs)
            self.store.put_many('meta', [{'id': 'race_test', 'origin': 'REAL', 'note': 'Concurrent edit'}], 'id')
            return model
        with patch('sacfactory.pending_prediction.fit_model', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'changed while fitting'):
                refresh_pending_prediction(self.store, sample['sample_id'], fingerprint(sample))
        self.assertNotIn('manual_prediction', self.store.get('sample', sample['sample_id']))

    def test_insufficient_data_does_not_fabricate_probability(self):
        with self.store.connect() as connection:
            connection.execute("DELETE FROM items WHERE kind='sample' AND id<>?", (self.sample['sample_id'],))
        sample = self.store.get('sample', self.sample['sample_id'])
        with self.assertRaisesRegex(ValueError, 'No fitted probability'):
            refresh_pending_prediction(self.store, sample['sample_id'], fingerprint(sample))
        self.assertNotIn('manual_prediction', self.store.get('sample', sample['sample_id']))
