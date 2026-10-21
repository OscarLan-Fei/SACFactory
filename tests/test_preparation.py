"""Declared preparation and unit conversion must survive the manual data loop."""
import copy
import unittest

import numpy as np

from sacfactory.batching import blank_batch, batch_rows_to_standard
from sacfactory.domain import validate_sample, recipe_key
from sacfactory.importers import samples_from_rows, sample_tables
from sacfactory.preparation import material_inputs, to_mM, from_mM, model_features, precursor_cell, precursor_summary, precursor_value
from sacfactory.composition import transfer_feature
from test_science import sample


class PreparationTests(unittest.TestCase):
    def test_explicit_salt_units_override_selector_and_preserve_each_element(self):
        self.assertEqual(precursor_cell('1 mmol', ['Fe'], 'mM', 20.), {'Fe': 50.})
        self.assertEqual(precursor_cell('50 mM', ['Fe'], 'mmol', 20.), {'Fe': 50.})
        values = precursor_cell('Fe:1 mmol;Co:25 mM', ['Fe', 'Co'], 'mM', 40.)
        self.assertEqual(values, {'Fe': 25., 'Co': 25.})
        self.assertEqual(precursor_summary({'Fe': 1., 'Co': .5}, ['Fe', 'Co']), 'Fe:1;Co:0.5')
        self.assertEqual(precursor_summary({'Fe': None}, ['Fe']), 'DETAIL')
        for invalid in ['1 mg', '1 M', '-1 mmol', '0 mM', 'inf', True]:
            with self.assertRaises(ValueError):
                precursor_value(invalid)
        with self.assertRaises(ValueError):
            precursor_cell('Fe:1 mmol', ['Fe', 'Co'])

    def test_mmol_mM_conversion_uses_volume_and_not_pan_mass(self):
        self.assertEqual(to_mM(1., 'mmol', 20.), 50.)
        self.assertEqual(to_mM(2.4, 'mmol', 20.), 120.)
        self.assertEqual(to_mM(1., 'mmol', 40.), 25.)
        self.assertEqual(from_mM(50., 'mmol', 20.), 1.)
        self.assertEqual(to_mM(50., 'mM', 40.), 50.)
        self.assertEqual(material_inputs(), {'pan_mass_g': 2., 'solvent_volume_mL': 20.})
        for value in [0., -1., float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError):
                material_inputs(value, 20.)

    def test_compact_batch_mmol_conversion_per_sample_volume_and_roundtrip(self):
        batch, details = blank_batch(['SAC_0001', 'SAC_0002'], ['Fe', 'Co'])
        batch[0].update(pan_mass_g=2., solvent_volume_mL=20., pulse_limit_C=1000.)
        batch[1].update(pan_mass_g=4., solvent_volume_mL=40., pulse_limit_C=1200.)
        for detail in details:
            detail['precursor_mM'] = 1. if detail['element'] == 'Fe' else .5
        settings = {'origin': 'REAL', 'batch_id': 'BATCH_1', 'target_elements': 'Fe;Co',
            'precursor_family': 'SALTS', 'support_batch': 'PAN', 'protocol_id': 'P1', 'precursor_input_unit': 'mmol'}
        ss, ii = batch_rows_to_standard(batch, details, settings)
        records = samples_from_rows(ss, ii)
        self.assertEqual(records[0]['concentrations_mM'], {'Fe': 50., 'Co': 25.})
        self.assertEqual(records[1]['concentrations_mM'], {'Fe': 25., 'Co': 12.5})
        self.assertEqual(records[1]['pan_mass_g'], 4.)
        self.assertEqual(records[1]['precursor_amounts_mmol'], {'Fe': 1., 'Co': .5})
        exported, assays = sample_tables(records)
        restored = samples_from_rows(exported, assays)
        for before, after in zip(records, restored):
            self.assertEqual(before['concentrations_mM'], after['concentrations_mM'])
            self.assertEqual(before['precursor_amounts_mmol'], after['precursor_amounts_mmol'])

    def test_new_entry_defaults_are_explicit_but_legacy_records_stay_unknown(self):
        batch, details = blank_batch(['SAC_0001'], ['Fe'])
        settings = {'origin': 'REAL', 'batch_id': '', 'target_elements': 'Fe', 'precursor_input_unit': 'mmol'}
        details[0]['precursor_mM'] = 1.
        ss, ii = batch_rows_to_standard(batch, details, settings)
        record = samples_from_rows(ss, ii)[0]
        self.assertEqual((record['pan_mass_g'], record['solvent_volume_mL']), (2., 20.))
        legacy = sample()
        self.assertNotIn('pan_mass_g', validate_sample(copy.deepcopy(legacy)))
        np.testing.assert_allclose(model_features(legacy), [0.] * 6)

    def test_explicit_mmol_compact_cell_records_the_conversion_volume(self):
        batch, details = blank_batch(['SAC_0001'], ['Fe'])
        batch[0]['concentration_mM'] = '1 mmol'
        ss, ii = batch_rows_to_standard(batch, details, {'origin': 'REAL', 'target_elements': 'Fe'})
        record = samples_from_rows(ss, ii)[0]
        self.assertEqual(record['concentrations_mM']['Fe'], 50.)
        self.assertEqual(record['solvent_volume_mL'], 20.)
        self.assertEqual(record['precursor_amounts_mmol']['Fe'], 1.)

    def test_preparation_changes_recipe_identity_and_enters_model_features(self):
        first = validate_sample({**sample(), 'pulse_limit_C': 1000., 'pan_mass_g': 2., 'solvent_volume_mL': 20.})
        second = validate_sample({**sample(), 'pulse_limit_C': 1000., 'pan_mass_g': 4., 'solvent_volume_mL': 20.})
        self.assertNotEqual(recipe_key(first), recipe_key(second))
        self.assertEqual(first['concentrations_mM'], second['concentrations_mM'])
        self.assertFalse(np.allclose(transfer_feature(first, 'chemistry_mixing'), transfer_feature(second, 'chemistry_mixing')))
