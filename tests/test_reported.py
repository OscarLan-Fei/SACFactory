"""Reported screening outcomes, partial entry and full periodic-table support."""
import copy
import tempfile
import unittest

from sacfactory.batching import PERIODS, blank_batch, batch_rows_to_standard
from sacfactory.domain import ELEMENTS, TARGETS, parse_elements, validate_sample, screen, recipe, recipe_key
from sacfactory.importers import samples_from_rows, sample_tables, methods_from_rows, SAMPLE_COLUMNS, ICP_COLUMNS, csv_bytes
from sacfactory.ingestion import preview_import, commit_import
from sacfactory.planning import fit_model, recommend
from sacfactory.storage import Store
from test_science import sample, method


class ReportedTests(unittest.TestCase):
    def compact(self, count=1):
        rows, details = blank_batch([f"DIRECT_{i}" for i in range(count)], ["Fe", "B", "Cd"])
        for row in rows: row.update(icp_result="PASS", xrd_result="PASS", pulse_limit_C=950)
        settings = {"origin": "REAL", "target_elements": "Fe;B;Cd"}
        return samples_from_rows(*batch_rows_to_standard(rows, details, settings))

    def test_all_118_symbols_are_supported_in_periodic_grid_and_import(self):
        self.assertEqual(len(ELEMENTS), 118)
        self.assertEqual(len(set(ELEMENTS)), 118)
        self.assertEqual({e for row in PERIODS for e in row.values()} - {"Ln", "An"}, set(ELEMENTS))
        self.assertEqual(set(parse_elements(" ".join(ELEMENTS))), set(ELEMENTS))
        self.assertEqual(parse_elements(" ".join(TARGETS)), TARGETS)
        with self.assertRaises(ValueError): parse_elements("Xx")
        with self.assertRaises(ValueError): parse_elements("Fe;Fe")

    def test_direct_pass_needs_no_values_limits_files_or_reviewer(self):
        s = self.compact()[0]
        result = screen(s, {})
        self.assertEqual((result["status"], result["label"]), ("PASS", 1))
        self.assertEqual(result["icp_basis"], "USER_REPORTED")
        self.assertEqual(result["xrd_basis"], "USER_REPORTED")
        self.assertEqual(result["numeric_icp"], "PENDING")
        self.assertTrue(all(not a["value"] for a in s["icp"]))
        self.assertEqual(s["batch_id"], "UNKNOWN")

    def test_reported_fail_and_pending_branches(self):
        s = self.compact()[0]
        s["icp_review"]["state"] = "FAIL"
        self.assertEqual(screen(s, {})["status"], "ICP_FAIL")
        s["xrd"]["state"] = "FAIL"
        self.assertEqual(screen(s, {})["status"], "BOTH_FAIL")
        s["icp_review"]["state"] = "PENDING"
        self.assertEqual(screen(s, {})["status"], "XRD_FAIL")
        s["xrd"]["state"] = "PASS"
        self.assertIsNone(screen(s, {})["label"])
        s["icp_review"]["state"] = "PASS"
        s["xrd"]["state"] = "FAIL"
        self.assertEqual(screen(s, {})["status"], "XRD_FAIL")
        s["icp_review"]["state"] = "FAIL"
        s["xrd"]["state"] = "PENDING"
        self.assertEqual(screen(s, {})["label"], 0)

    def test_partial_conditions_save_without_imputation_and_do_not_train(self):
        s = self.compact()[0]
        s.update(pulse_count="", temperature_C="", pulse_limit_C="", concentrations_mM={e: None for e in s["elements"]})
        validate_sample(s)
        self.assertIsNone(s["pulse_count"])
        self.assertIsNone(s["temperature_C"])
        self.assertEqual(screen(s, {})["status"], "PASS")
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder, "REAL")
            ss, ii = sample_tables([s])
            checked = preview_import([("samples.csv", csv_bytes(ss, SAMPLE_COLUMNS)), ("icp.csv", csv_bytes(ii, ICP_COLUMNS))], store)
            commit_import(checked, store)
            restored = store.get("sample", s["sample_id"])
            self.assertEqual(restored["concentrations_mM"], s["concentrations_mM"])
            self.assertIsNone(restored["pulse_limit_C"])
            self.assertEqual(screen(restored, {})["status"], "PASS")
        state = fit_model([s], {}, s)
        self.assertEqual(state.info["n_labeled"], 0)
        self.assertIn("Incomplete synthesis conditions", state.info["exclusions"])
        with self.assertRaises(ValueError): recommend(state, [s], s, {}, "REAL")

    def test_reported_judgments_round_trip_separately_from_assay_values(self):
        s = self.compact()[0]
        s["icp_review"].update(source="Notebook page 1", reviewer="A")
        restored = samples_from_rows(*sample_tables([s]))[0]
        self.assertEqual(restored["icp_review"], s["icp_review"])
        self.assertEqual(restored["xrd"]["mode"], "MANUAL")
        self.assertEqual(recipe_key(restored), recipe_key(s))

    def test_direct_outcomes_complete_reserved_recipe_without_numeric_values(self):
        s = self.compact()[0]; s["batch_id"] = "ACTUAL_BATCH"
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder, "REAL")
            store.put_many("recommendation", [{"origin": "REAL", "recommendation_id": s["sample_id"], "state": "OPEN", "recipe": copy.deepcopy(s)}], "recommendation_id")
            ss, ii = sample_tables([s])
            checked = preview_import([("samples.csv", csv_bytes(ss, SAMPLE_COLUMNS)), ("icp.csv", csv_bytes(ii, ICP_COLUMNS))], store)
            self.assertEqual(len(checked.queue_updates), 1)
            commit_import(checked, store)
            self.assertEqual(store.get("recommendation", s["sample_id"])["state"], "COMPLETED")

    def test_complete_recipes_with_reported_labels_can_train_without_assays(self):
        rows = []
        for i in range(18):
            s = sample(i, alpha=.3+.12*i)
            s["icp"] = []
            s["icp_review"] = {"state": "PASS" if i >= 9 else "FAIL"}
            s["xrd"] = {"state": "PASS", "mode": "MANUAL"}
            s["pulse_limit_C"] = 850+20*i
            rows.append(s)
        state = fit_model(rows, {}, rows[0])
        self.assertIsNotNone(state.model)
        self.assertEqual(state.info["reported_label_count"], 18)

    def test_conflicting_numeric_and_direct_results_are_visible_and_excluded(self):
        s = sample(); s["icp_review"] = {"state": "FAIL"}
        result = screen(s, method())
        self.assertEqual(result["numeric_icp"], "PASS")
        self.assertEqual(result["icp"], "FAIL")
        self.assertTrue(result["review_conflicts"])
        state = fit_model([s], method(), s)
        self.assertEqual(state.info["n_labeled"], 0)
        self.assertIn("Conflicting reported and numeric judgments", state.info["exclusions"])

    def test_old_evidence_mode_and_invalid_conditions_remain_checked(self):
        s = sample(); s["xrd"]["source"] = ""
        self.assertEqual(screen(s, method())["status"], "PENDING")
        for field, value in (("pulse_count", 1.5), ("temperature_C", -1)):
            s = sample(); s[field] = value
            with self.assertRaises(ValueError): validate_sample(s)
        s = sample(); s["icp_review"] = {"state": "YES"}
        with self.assertRaises(ValueError): validate_sample(s)


if __name__ == "__main__": unittest.main()
