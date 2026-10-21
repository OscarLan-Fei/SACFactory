"""Demo preparation inputs survive packaging, upgrades and ICP fitting."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from sacfactory.bootstrap import load_demo_seed, seed
from sacfactory.demo_preparation import FIELDS, VERSION, migrate_store, restore_inputs
from sacfactory import pulseicp_bridge as bridge
from sacfactory.storage import Store

ROOT = Path(__file__).resolve().parents[1]


class DemoPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = load_demo_seed(ROOT)

    def legacy(self):
        sample = copy.deepcopy(self.content['samples'][0])
        for field in FIELDS:
            sample.pop(field, None)
        sample.pop('precursor_amounts_mmol', None)
        sample.pop('demo_preparation_provenance', None)
        return sample

    def test_restores_only_declared_simulation_inputs_and_converts_amounts(self):
        original = self.legacy()
        restored = restore_inputs(original)
        self.assertNotIn('pan_mass_g', original)
        for field in FIELDS:
            self.assertEqual(restored[field], original['demo_mass_balance'][field])
        self.assertEqual(restored['icp'], original['icp'])
        self.assertEqual(restored['xrd'], original['xrd'])
        self.assertEqual(restored['concentrations_mM'], original['concentrations_mM'])
        self.assertAlmostEqual(restored['precursor_amounts_mmol']['Fe'],
                               restored['concentrations_mM']['Fe'] * 20 / 1000)
        self.assertEqual(restore_inputs(restored), restored)

    def test_does_not_fill_real_unknown_or_edited_metadata(self):
        original = self.legacy()
        for patch in [{'origin': 'REAL'}, {'record_uid': 'manually-entered-demo'},
                      {'demo_mass_balance': {}}]:
            sample = {**original, **patch}
            self.assertEqual(restore_inputs(sample), sample)
        sample = {**original, 'pan_mass_g': 3.0, 'solvent_volume_mL': 40.0}
        restored = restore_inputs(sample)
        self.assertEqual(restored['pan_mass_g'], 3.0)
        self.assertEqual(restored['solvent_volume_mL'], 40.0)

    def test_saved_simulator_forecasts_keep_probabilities_and_original_inputs(self):
        sample = copy.deepcopy(next(s for s in self.content['samples'] if 'proposal' in s))
        for field in FIELDS:
            sample['proposal']['recipe'].pop(field, None)
        sample['pan_mass_g'] = 3.0
        probability = sample['proposal']['p_screen_pass']
        restored = restore_inputs(sample)
        self.assertEqual(restored['proposal']['p_screen_pass'], probability)
        self.assertEqual(restored['proposal']['recipe']['pan_mass_g'], 2.0)
        self.assertEqual(restored['pan_mass_g'], 3.0)
        self.assertEqual(restore_inputs(restored), restored)

    def test_compressed_fixture_and_planning_context_are_trainable(self):
        legacy = copy.deepcopy(self.content)
        legacy['samples'] = [self.legacy()]
        legacy['planning_context'] = self.legacy()
        with tempfile.TemporaryDirectory() as directory:
            examples = Path(directory) / 'examples'
            examples.mkdir()
            with gzip.open(examples / 'demo_seed.json.gz', 'wt', encoding='utf-8') as stream:
                json.dump(legacy, stream)
            restored = load_demo_seed(directory)
        self.assertEqual(restored['planning_context']['pan_mass_g'], 2.0)
        self.assertEqual(len(bridge.prepare(restored['samples'], 'DEMO').records), 1)

    def test_existing_demo_store_repairs_once_without_replacing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, 'DEMO')
            original = self.legacy()
            store.put_many('sample', [original], 'sample_id')
            store.put_many('meta', [{'id': 'seed-v1', 'origin': 'DEMO'},
                {'id': 'plan_context', 'origin': 'DEMO', 'context': original}], 'id')
            seed(store, ROOT)
            restored = store.get('sample', original['sample_id'])
            self.assertEqual(restored['icp'], original['icp'])
            self.assertEqual(restored['pan_mass_g'], 2.0)
            self.assertEqual(store.get('meta', 'plan_context')['context']['solvent_volume_mL'], 20.0)
            self.assertEqual(store.get('meta', VERSION)['restored_samples'], 1)
            self.assertEqual(migrate_store(store), 0)
            real = Store(directory, 'REAL')
            self.assertEqual(migrate_store(real), 0)
            self.assertIsNone(real.get('meta', VERSION))

    def test_bundled_campaign_has_many_eligible_icp_targets(self):
        samples = self.content['samples']
        self.assertEqual(len(samples), 1476)
        for sample in samples:
            self.assertEqual(sample['pan_mass_g'], 2.0)
            self.assertEqual(sample['solvent_volume_mL'], 20.0)
            self.assertEqual(sample['carbonization_loss_fraction'], .06)
        data = bridge.prepare(samples, 'DEMO')
        self.assertGreater(len(data.records), 3000)
        self.assertGreater(len({r['raw']['batch_id'] for r in data.records}), 6)
        self.assertFalse(data.audit.exclusion_reason.str.contains('Missing recipe inputs').any())


if __name__ == '__main__':
    unittest.main()
