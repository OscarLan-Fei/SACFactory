import copy
import io
import json
import unittest

import openpyxl
import numpy as np

from sacfactory.domain import validate_sample, common_alpha
from sacfactory.planning import feature, fit_model, recommend, recommendation_table, validate_space, concentration_bounds
from sacfactory.importers import sample_tables, samples_from_rows, recommendation_workbook, read_table, SAMPLE_COLUMNS
from sacfactory.language import shared_context
from sacfactory.operating_space import restore_operating_space


class DirectConcentrationTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for i in range(18):
            self.rows.append(validate_sample({
                "sample_id": f"D{i}", "batch_id": f"B{i}", "origin": "REAL",
                "elements": ["Fe", "Mg"], "concentrations_mM": {"Fe": .02 + i * .004, "Mg": .06 + (i % 5) * .017},
                "pulse_count": i % 3 + 1, "pulse_limit_C": 950 + i * 5, "temperature_C": None,
                "precursor_family": "NITRATES", "support_batch": "CARBON_A", "protocol_id": "P1",
                "icp": [], "icp_review": {"state": "PASS" if i >= 8 else "FAIL"},
                "xrd": {"state": "PASS", "mode": "MANUAL"}}))
        self.context = self.rows[0]
        self.space = {"concentration_bounds_mM": {"Fe": {"min_mM": .01, "max_mM": .1}, "Mg": {"min_mM": .04, "max_mM": .2}},
            "pulses": [1, 2], "pulse_limit_min_C": 900, "pulse_limit_max_C": 1100,
            "max_total_mM": .22, "verified": True, "baseline_repeat": False}

    def test_independent_element_concentrations_train_and_do_not_leak_outcomes(self):
        self.assertIsNone(common_alpha(self.rows[3]))
        before = feature(self.context)
        changed = copy.deepcopy(self.context)
        changed["concentrations_mM"]["Mg"] *= 2
        after = feature(changed)
        self.assertEqual(len(before), 4)
        self.assertEqual(before[self.context["elements"].index("Fe")], after[self.context["elements"].index("Fe")])
        self.assertNotEqual(before[self.context["elements"].index("Mg")], after[self.context["elements"].index("Mg")])
        changed["icp_review"]["state"] = "PASS"
        changed["temperature_C"] = 1250
        self.assertEqual(after, feature(changed))
        state = fit_model(self.rows, {}, self.context)
        self.assertEqual(state.info["n_labeled"], 18)
        self.assertIsNotNone(state.model)
        self.assertEqual(state.model.n_features_in_, 4)
        self.assertIsNotNone(state.info["validation"])

    def test_recommendations_obey_each_direct_bound_and_export_actual_concentrations(self):
        state = fit_model(self.rows, {}, self.context)
        rows, _ = recommend(state, self.rows, self.context, self.space, "REAL", 6)
        self.assertEqual(len(rows), 6)
        ratios = []
        for r in rows:
            recipe = r["recipe"]
            for e, b in self.space["concentration_bounds_mM"].items():
                self.assertTrue(b["min_mM"] <= recipe["concentrations_mM"][e] <= b["max_mM"])
            self.assertLessEqual(sum(recipe["concentrations_mM"].values()), .22 + 1e-9)
            self.assertTrue(900 <= recipe["pulse_limit_C"] <= 1100)
            self.assertIsNone(recipe["temperature_C"])
            ratios.append(recipe["concentrations_mM"]["Fe"] / recipe["concentrations_mM"]["Mg"])
        self.assertGreater(np.ptp(ratios), .01)
        table = recommendation_table(rows)
        self.assertNotIn("alpha", table[0])
        self.assertIn("Fe:", table[0]["concentration_mM"])
        self.assertEqual(json.loads(table[0]["concentrations_mM_json"]), rows[0]["recipe"]["concentrations_mM"])
        content = recommendation_workbook(rows)
        book = openpyxl.load_workbook(io.BytesIO(content))
        self.assertNotIn("alpha", [c.value for c in book["samples"][1]])
        samples = samples_from_rows(read_table(content, "report.xlsx", "samples"), read_table(content, "report.xlsx", "icp"))
        for e, value in samples[0]["concentrations_mM"].items():
            self.assertAlmostEqual(value, rows[0]["recipe"]["concentrations_mM"][e], places=12)

    def test_fixed_element_stays_fixed(self):
        self.space["concentration_bounds_mM"]["Mg"] = {"min_mM": .1, "max_mM": .1}
        state = fit_model([], {}, self.context)
        rows, _ = recommend(state, [], self.context, self.space, "REAL", 6)
        self.assertTrue(all(r["recipe"]["concentrations_mM"]["Mg"] == .1 for r in rows))
        self.assertTrue(all(r["p_screen_pass"] is None for r in rows))

    def test_invalid_or_mismatched_element_ranges_rejected(self):
        for value in (None, {"min_mM": 0, "max_mM": .1}, {"min_mM": .2, "max_mM": .1}):
            bad = copy.deepcopy(self.space)
            bad["concentration_bounds_mM"]["Fe"] = value
            with self.assertRaises(ValueError): validate_space(bad, "REAL")
        bad = copy.deepcopy(self.space)
        del bad["concentration_bounds_mM"]["Mg"]
        with self.assertRaises(ValueError): concentration_bounds(bad, self.context["elements"])

    def test_legacy_settings_are_converted_and_partial_exports_preserve_missing_values(self):
        old = {"alpha_min": .5, "alpha_max": 2, "pulses": [1], "verified": True}
        state = fit_model([], {}, self.context)
        shared = shared_context([], {}, state, self.context, old, "REAL")
        self.assertNotIn("alpha_min", shared["operating_space"])
        self.assertEqual(shared["operating_space"]["concentration_bounds_mM"]["Mg"], {"min_mM": .05, "max_mM": .2})
        self.assertEqual(old["alpha_min"], .5)
        partial = copy.deepcopy(self.context)
        partial["concentrations_mM"]["Mg"] = None
        sample_rows, icp_rows = sample_tables([partial])
        self.assertNotIn("alpha", SAMPLE_COLUMNS)
        self.assertNotIn("alpha", sample_rows[0])
        restored = samples_from_rows(sample_rows, icp_rows)
        self.assertEqual(restored[0]["concentrations_mM"], partial["concentrations_mM"])

    def test_sparse_upgrade_settings_restore_demo_ranges_without_mutation(self):
        saved = {'model_engine': 'reliability', 'descriptor_profile': 'chemistry', 'learning_iterations': 8}
        before = copy.deepcopy(saved)
        demo = copy.deepcopy(self.context)
        demo['origin'] = 'DEMO'
        rows = [demo, {**demo, 'pulse_limit_C': 1100}, {**demo, 'pulse_limit_C': None}]
        restored = restore_operating_space(saved, demo, rows, 'DEMO')
        self.assertEqual(saved, before)
        self.assertEqual(restored['model_engine'], 'reliability')
        self.assertEqual(restored['pulses'], [1, 2, 3, 4, 6, 8])
        self.assertEqual(restored['pulse_limit_min_C'], 950)
        self.assertEqual(restored['pulse_limit_max_C'], 1100)
        self.assertAlmostEqual(concentration_bounds(restored, ['Fe', 'Mg'])['Fe']['min_mM'], demo['concentrations_mM']['Fe'])
        self.assertNotIn('alpha_min', restored)
        validate_space(restored, 'DEMO')

    def test_sparse_real_settings_require_range_review_and_keep_fixed_references(self):
        restored = restore_operating_space({'model_engine': 'reliability', 'verified': True}, self.context, self.rows, 'REAL')
        for e in self.context['elements']:
            self.assertEqual(restored['concentration_bounds_mM'][e],
                {'min_mM': self.context['concentrations_mM'][e], 'max_mM': self.context['concentrations_mM'][e]})
        self.assertEqual(restored['pulse_limit_min_C'], self.context['pulse_limit_C'])
        self.assertEqual(restored['pulse_limit_max_C'], self.context['pulse_limit_C'])
        self.assertFalse(restored['verified'])
        with self.assertRaisesRegex(ValueError, 'device-validated'):
            validate_space(restored, 'REAL')

    def test_entered_ranges_preserved_and_missing_legacy_fields_are_readable_errors(self):
        self.assertEqual(restore_operating_space(self.space, self.context, self.rows, 'REAL')['concentration_bounds_mM'], self.space['concentration_bounds_mM'])
        for saved in ({}, {'alpha_min': .1}, {'alpha_max': 4}):
            with self.assertRaisesRegex(ValueError, 'concentration'):
                concentration_bounds(saved, self.context['elements'])
            with self.assertRaisesRegex(ValueError, 'concentration'):
                validate_space(saved, 'DEMO')
        empty = {**self.space, 'concentration_bounds_mM': {}}
        restored = restore_operating_space(empty, self.context, self.rows, 'DEMO')
        self.assertEqual(restored['concentration_bounds_mM'], {})
        with self.assertRaises(ValueError):
            concentration_bounds(restored, self.context['elements'])
        partial_cutoff = {**self.space}
        del partial_cutoff['pulse_limit_max_C']
        with self.assertRaisesRegex(ValueError, 'both controlled'):
            validate_space(restore_operating_space(partial_cutoff, self.context, self.rows, 'DEMO'), 'DEMO')
        legacy = {'alpha_min': .5, 'alpha_max': 2, 'pulses': [1]}
        migrated = restore_operating_space(legacy, self.context, [], 'DEMO')
        self.assertNotIn('concentration_bounds_mM', migrated)
        self.assertEqual(concentration_bounds(migrated, ['Fe'])['Fe'], {'min_mM': .025, 'max_mM': .1})


if __name__ == "__main__":
    unittest.main()
