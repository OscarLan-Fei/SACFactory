import copy
import json
import tempfile
import unittest
from pathlib import Path

from sacfactory.batching import PERIODS, BATCH_COLUMNS, SETTINGS_COLUMNS, blank_batch, batch_rows_to_standard
from sacfactory.domain import TARGETS, recipe_key, cohort_key, screen, validate_sample
from sacfactory.importers import samples_from_rows, sample_tables, csv_bytes, ICP_COLUMNS, read_table, recommendation_workbook
from sacfactory.ingestion import preview_import, commit_import
from sacfactory.planning import feature, fit_model, recommend
from sacfactory.storage import Store
from test_science import sample, method


class BatchTests(unittest.TestCase):
    def settings(self):
        return {"batch_id": "B_NEW", "origin": "DEMO", "target_elements": "Fe", "precursor_family": "NITRATES_V1", "support_batch": "CARBON_A", "protocol_id": "P1", "method_id": "M1",
                "xrd_source": "XRD fixture", "xrd_criteria": "Support/reference comparison", "xrd_reviewer": "TESTER", "stem_source": "", "stem_reviewer": ""}

    def input(self):
        rows, details = blank_batch(["NEW1"], ["Fe"])
        rows[0].update(pulse_limit_C=950, temperature_C=973, voltage_V=12, current_A=4, xrd_result="PASS")
        details[0].update(value="14")
        return rows, details

    def test_periodic_positions_cover_the_target_pool(self):
        visible = {e for row in PERIODS for e in row.values()}
        self.assertTrue(set(TARGETS) <= visible)
        self.assertTrue({"B", "N", "Cd"} <= visible)

    def test_batch_conditions_and_no_qc_round_trip(self):
        rows, details = self.input()
        ss, ii = batch_rows_to_standard(rows, details, self.settings())
        s = samples_from_rows(ss, ii)[0]
        self.assertNotIn("qc", s)
        self.assertEqual((s["voltage_V"], s["current_A"], s["pulse_limit_C"], s["temperature_C"]), (12, 4, 950, 973))
        self.assertEqual(screen(s, method("DEMO"))["status"], "PASS")
        restored = samples_from_rows(*sample_tables([s]))[0]
        self.assertEqual(recipe_key(restored), recipe_key(s))
        self.assertEqual(restored["voltage_V"], 12)

    def test_inline_and_per_element_concentrations(self):
        rows, details = blank_batch(["NEW1"], ["Mg", "Fe"])
        settings = {**self.settings(), "target_elements": "Mg;Fe"}
        rows[0].update(concentration_mM="Mg:0.16;Fe:0.08", icp_result="Mg:20;Fe:ND", pulse_limit_C=1000)
        s = samples_from_rows(*batch_rows_to_standard(rows, details, settings))[0]
        self.assertEqual(s["concentrations_mM"], {"Mg": .16, "Fe": .08})
        self.assertEqual(s["icp"][1]["flag"], "ND")
        rows[0]["concentration_mM"] = .04
        s = samples_from_rows(*batch_rows_to_standard(rows, details, settings))[0]
        self.assertEqual(set(s["concentrations_mM"].values()), {.04})

    def test_duplicate_or_conflicting_results_are_rejected(self):
        rows, details = self.input()
        rows[0]["icp_result"] = "Fe:12"
        with self.assertRaises(ValueError): batch_rows_to_standard(rows, details, self.settings())
        rows[0]["icp_result"] = ""
        with self.assertRaises(ValueError): batch_rows_to_standard(rows, details + details, self.settings())

    def test_optional_stem_is_separate_from_training_label(self):
        s = sample()
        for state in ("NOT_TESTED", "CONFIRMED", "NOT_CONFIRMED", "INCONCLUSIVE"):
            s["stem"] = {"state": state, "source": "STEM fixture", "reviewer": "TESTER"}
            self.assertEqual(screen(s, method())["label"], 1)
            self.assertEqual(screen(s, method())["stem"], state)
        s["stem"]["source"] = ""
        self.assertEqual(screen(s, method())["stem"], "INCOMPLETE_EVIDENCE")

    def test_cutoff_is_a_predictor_measured_peak_and_electrical_readings_are_not(self):
        s = sample(); s["pulse_limit_C"] = 950
        reference = feature(s)
        s.update(temperature_C=1400, voltage_V=50, current_A=20)
        self.assertEqual(feature(s), reference)
        self.assertEqual(len(reference), 3)
        s["pulse_limit_C"] = 1100
        self.assertNotEqual(feature(s), reference)

    def test_cutoff_temperatures_share_model_cohort_but_not_recipe_identity(self):
        s = sample(); s["pulse_limit_C"] = 900
        t = copy.deepcopy(s); t.update(pulse_limit_C=1100, temperature_C=1140)
        self.assertEqual(cohort_key(s), cohort_key(t))
        self.assertNotEqual(recipe_key(s), recipe_key(t))
        legacy = sample()
        self.assertNotEqual(cohort_key(s), cohort_key(legacy))

    def test_three_input_fit_and_temperature_recommendations(self):
        samples = []
        for i in range(18):
            s = sample(i, alpha=.3 + .13 * i, pulses=i % 3 + 1)
            s["pulse_limit_C"] = 800 + 25 * i
            s["icp"][0]["value"] = "5" if i < 9 else "15"
            samples.append(s)
        context = samples[0]
        state = fit_model(samples + [sample(99)], method(), context)
        self.assertEqual(state.model.n_features_in_, 3)
        self.assertEqual(state.info["n_labeled"], 18)
        space = {"alpha_min": .5, "alpha_max": 2, "pulses": [1, 2], "max_total_mM": 1, "verified": True, "baseline_repeat": False, "pulse_limit_min_C": 900, "pulse_limit_max_C": 1100}
        rows, note = recommend(state, samples, context, space, "REAL", 6)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(900 <= r["recipe"]["pulse_limit_C"] <= 1100 for r in rows))
        self.assertTrue(all(r["recipe"]["temperature_C"] is None for r in rows))
        self.assertTrue(all(r["p_screen_pass"] is not None for r in rows))
        space["pulse_limit_max_C"] = 800
        with self.assertRaises(ValueError): recommend(state, samples, context, space, "REAL")

    def test_compact_csv_import_uses_existing_method_and_completes_a_controlled_recipe(self):
        rows, details = self.input()
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder, "DEMO")
            store.put_many("method", list(method("DEMO").values()), "method_id")
            ss, ii = batch_rows_to_standard(rows, details, self.settings())
            reserved_sample = samples_from_rows(ss, ii)[0]
            reserved_sample["temperature_C"] = None
            store.put_many("recommendation", [{"origin": "DEMO", "recommendation_id": "NEW1", "state": "OPEN", "recipe": reserved_sample}], "recommendation_id")
            files = [("batch.csv", csv_bytes(rows, BATCH_COLUMNS)), ("icp.csv", csv_bytes(details, ICP_COLUMNS)), ("settings.csv", csv_bytes([self.settings()], SETTINGS_COLUMNS))]
            preview = preview_import(files, store)
            self.assertEqual(len(preview.queue_updates), 1)
            commit_import(preview, store)
            self.assertEqual(store.get("recommendation", "NEW1")["state"], "COMPLETED")
            self.assertEqual(store.all("sample")[0]["temperature_C"], 973)

    def test_compact_template_has_fillable_conditions_and_blank_measurements(self):
        path = Path(__file__).resolve().parents[1] / "examples/compact_batch_template.xlsx"
        rows = read_table(path.read_bytes(), path.name, "batch")
        self.assertEqual(set(rows[0]), set(BATCH_COLUMNS) - {"sample_condition"})
        self.assertTrue(all(r["icp_result"] == "AUTO" and not r["temperature_C"] and not r["pulse_limit_C"] for r in rows))
        details = read_table(path.read_bytes(), path.name, "icp")
        self.assertTrue(all(not r["w_wt_percent"] for r in details))
        self.assertNotIn("qc", rows[0])

    def test_one_batch_can_report_different_sample_compositions(self):
        rows, details = blank_batch(["ONE"], ["Fe"])
        second, other = blank_batch(["TWO"], ["Fe", "Mg"])
        for r in rows + second: r["pulse_limit_C"] = 1000
        restored = samples_from_rows(*batch_rows_to_standard(rows + second, details + other, self.settings()))
        self.assertEqual(restored[0]["elements"], ["Fe"])
        self.assertEqual(restored[1]["elements"], ["Mg", "Fe"])

    def test_compact_proposal_export_never_copies_outcomes_or_actual_temperature(self):
        import io
        import zipfile
        from sacfactory.batching import compact_reporting_zip
        s = sample(); s["pulse_limit_C"] = 950
        state = fit_model([s], method(), s)
        space = {"alpha_min": 1, "alpha_max": 1, "pulses": [1], "max_total_mM": .05, "verified": True, "baseline_repeat": True, "pulse_limit_min_C": 900, "pulse_limit_max_C": 1100}
        proposals = recommend(state, [s], s, space, "REAL", 2)[0]
        with zipfile.ZipFile(io.BytesIO(compact_reporting_zip(proposals))) as archive:
            batch = read_table(archive.read("batch.csv"), "batch.csv")
            self.assertTrue(all(r["pulse_limit_C"] and not r["temperature_C"] and r["icp_result"] == "AUTO" for r in batch))
            self.assertTrue(all(not r["w_wt_percent"] for r in read_table(archive.read("icp.csv"), "icp.csv")))


if __name__ == "__main__": unittest.main()
