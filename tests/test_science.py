from sacfactory.bootstrap import load_demo_seed
import copy
import json
import tempfile
import unittest
from pathlib import Path

from sacfactory.domain import *
from sacfactory.importers import *
from sacfactory.storage import Store
from sacfactory.planning import *
from sacfactory.dialogue import reply


def sample(i=0, alpha=1, pulses=1, origin="REAL"):
    return validate_sample({"sample_id": f"S{i}", "batch_id": f"B{i}", "origin": origin,
        "elements": ["Fe"], "concentrations_mM": recipe(["Fe"], alpha), "pulse_count": pulses, "temperature_C": 1000,
        "precursor_family": "NITRATES_V1", "support_batch": "CARBON_A", "protocol_id": "P1", "qc": "VALID",
        "xrd": {"state": "PASS", "source": "xrd.txt", "criteria": "validated background comparison", "reviewer": "tester"},
        "icp": [{"element": "Fe", "value": "12", "unit": "wt%", "flag": "MEASURED", "method_id": "M1", "source": "icp.csv"}], "notes": ""})


def method(origin="REAL"):
    return {"M1": {"method_id": "M1", "origin": origin, "instrument": "ICP-AES", "rule": "LOQ", "confirmed": True,
        "unit": "wt%", "version": "v1", "source": "validated limits.csv", "limits": {"Fe": {"lod": 2, "loq": 10}}}}


class ScreeningTests(unittest.TestCase):
    def test_baseline_total_and_special_elements(self):
        self.assertAlmostEqual(sum(recipe(TARGETS).values()), 2.75)
        self.assertEqual(recipe(["Fe", "Mg"]), {"Fe": .05, "Mg": .1})

    def test_complete_pass_is_candidate_label(self):
        r = screen(sample(), method())
        self.assertEqual((r["status"], r["label"], r["weakest"]), ("PASS", 1, ("Fe", 1.2)))

    def test_missing_limits_never_pass(self):
        m = method(); m["M1"]["confirmed"] = False
        self.assertEqual(screen(sample(), m)["status"], "PENDING")

    def test_method_namespace_isolation(self):
        self.assertEqual(screen(sample(), method("DEMO"))["status"], "PENDING")

    def test_no_xrd_source_is_pending(self):
        s = sample(); s["xrd"]["source"] = ""
        self.assertEqual(screen(s, method())["status"], "PENDING")

    def test_missing_any_element_pending(self):
        s = sample(); s["elements"] += ["Co"]; s["concentrations_mM"]["Co"] = .05
        self.assertEqual(screen(s, method())["status"], "PENDING")

    def test_censoring_interval_not_imputed(self):
        s = sample(); s["icp"][0]["value"] = "<5"
        self.assertEqual(screen(s, method())["status"], "ICP_FAIL")
        self.assertEqual(parse_measurement("<5")["kind"], "censored")
        s["icp"][0]["value"] = "<20"
        self.assertEqual(screen(s, method())["status"], "PENDING")
        s["icp"][0]["value"] = "≤10"
        self.assertEqual(screen(s, method())["status"], "PENDING")

    def test_nd_not_zero(self):
        s = sample(); s["icp"][0].update(value="ND", flag="ND")
        self.assertEqual(screen(s, method())["label"], 0)

    def test_blq_cannot_mean_not_detected(self):
        s = sample(); s["icp"][0].update(value="ND", flag="BLQ")
        m = method(); m["M1"]["rule"] = "LOD"
        self.assertEqual(screen(s, m)["status"], "PENDING")

    def test_legacy_qc_is_preserved_but_not_a_screening_gate(self):
        for value in ("RETEST", "PENDING", None):
            s = sample(); s["qc"] = value
            self.assertEqual(screen(s, method())["label"], 1)
        s.pop("qc")
        self.assertEqual(screen(validate_sample(s), method())["status"], "PASS")

    def test_failure_branches(self):
        s = sample(); s["xrd"]["state"] = "FAIL"
        self.assertEqual(screen(s, method())["status"], "XRD_FAIL")
        s["icp"][0]["value"] = "1"
        self.assertEqual(screen(s, method())["status"], "BOTH_FAIL")

    def test_invalid_temperature_pulse_unit(self):
        for key, value in (("pulse_count", 1.5), ("temperature_C", float("nan"))):
            s = sample(); s[key] = value
            with self.assertRaises(ValueError): validate_sample(s)
        s = sample(); s["icp"][0]["unit"] = "ug/L"
        with self.assertRaises(ValueError): validate_sample(s)


class WorkflowTests(unittest.TestCase):
    def test_transaction_rolls_back(self):
        with tempfile.TemporaryDirectory() as path:
            db = Store(path, "REAL")
            db.put_many("sample", [sample()], "sample_id")
            with self.assertRaises(ValueError):
                db.import_bundle([sample()], list(method().values()), overwrite=False)
            self.assertEqual(db.all("method"), [])

    def test_revision_keeps_old_value(self):
        with tempfile.TemporaryDirectory() as path:
            db = Store(path, "REAL")
            db.put_many("sample", [sample()], "sample_id")
            s = sample(); s["qc"] = "RETEST"
            db.put_many("sample", [s], "sample_id", True)
            audit = db.export()["audit"]
            self.assertEqual(len(audit), 2)
            self.assertEqual(json.loads(audit[1]["before_json"])["qc"], "VALID")
            with self.assertRaises(ValueError): db.put_many("sample", [sample(origin="DEMO")], "sample_id")

    def test_roundtrip_custom_element_concentrations(self):
        s = sample(); s["elements"] += ["Co"]; s["concentrations_mM"]["Co"] = .08
        ss, ii = sample_tables([s])
        restored = samples_from_rows(ss, ii)[0]
        self.assertEqual(restored["concentrations_mM"], s["concentrations_mM"])
        self.assertIsNone(common_alpha(restored))

    def test_excel_template_read(self):
        path = Path(__file__).resolve().parents[1] / "examples/bulk_import_template.xlsx"
        rows = read_table(path.read_bytes(), path.name, "samples")
        self.assertEqual(rows[0]["target_elements"], "Fe;Co")

    def test_real_and_demo_seed_integrity(self):
        root = Path(__file__).resolve().parents[1]
        from sacfactory.bootstrap import seed
        with tempfile.TemporaryDirectory() as folder:
            real = Store(folder, "REAL")
            seed(real, root)
            self.assertEqual(real.all("sample"), [])
            self.assertEqual(real.all("method"), [])
        demo = load_demo_seed(root)
        counts = summary(demo["samples"], {m["method_id"]: m for m in demo["methods"]})["counts"]
        self.assertEqual(counts, demo['status_counts'])
        self.assertEqual(sum(counts.values()), 1476)
        self.assertTrue({'PASS', 'BURNED_OUT', 'ICP_FAIL', 'XRD_FAIL', 'RETEST', 'PENDING'} <= set(counts))
        self.assertTrue(set(counts)<={'PASS','BURNED_OUT','ICP_FAIL','XRD_FAIL','BOTH_FAIL','RETEST','PENDING'})

    def test_source_code_contains_no_cjk_text(self):
        import re
        root = Path(__file__).resolve().parents[1]
        for path in root.rglob("*.py"):
            self.assertIsNone(re.search(r"[\u4e00-\u9fff]", path.read_text(encoding="utf-8")), str(path))


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.space = {"alpha_min": .5, "alpha_max": 2, "pulses": [1, 2, 3], "max_total_mM": 1, "verified": True, "baseline_repeat": False}

    def training(self):
        rows = [sample(i, alpha=.3 + .13 * i, pulses=i % 3 + 1) for i in range(18)]
        for i, s in enumerate(rows):
            if i < 8: s["icp"][0]["value"] = "1"
        return rows

    def test_one_class_initial_design_no_fake_probability(self):
        rows = [sample(i, .5 + i * .1) for i in range(20)]
        state = fit_model(rows, method(), rows[0])
        self.assertIsNone(state.model)
        recs, _ = recommend(state, rows, rows[0], self.space, "REAL")
        self.assertTrue(all(r["p_screen_pass"] is None for r in recs))

    def test_real_rejects_demo_training(self):
        rows = self.training()
        for s in rows: s["origin"] = "DEMO"
        state = fit_model(rows, method("DEMO"), sample(), "REAL")
        self.assertEqual(state.info["n_labeled"], 0)

    def test_model_inputs_do_not_contain_outcomes(self):
        s = sample()
        before = feature(s)
        s["icp"][0]["value"] = "2000"; s["xrd"]["state"] = "FAIL"
        self.assertEqual(before, feature(s))

    def test_new_combination_does_not_reuse_model(self):
        context = sample(); context["elements"] = ["Co"]; context["concentrations_mM"] = recipe(["Co"])
        state = fit_model(self.training(), method(), context)
        self.assertIsNone(state.model)
        self.assertEqual(state.info["n_labeled"], 0)

    def test_repeated_recipe_and_batch_are_same_cv_group(self):
        rows = [sample(0), sample(1), sample(2, 2), sample(3, 3)]
        rows[2]["batch_id"] = rows[1]["batch_id"]
        groups = independent_groups(rows)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])

    def test_training_and_prediction_with_group_validation(self):
        rows = self.training()
        state = fit_model(rows, method(), rows[0])
        self.assertIsNotNone(state.model)
        self.assertIsNotNone(state.info["validation"])
        recs, _ = recommend(state, rows, rows[0], self.space, "REAL", count=6)
        self.assertEqual(len(recs), 6)
        self.assertTrue(all(0 <= r["p_screen_pass"] <= 1 for r in recs))

    def test_unverified_space_rejected(self):
        self.space["verified"] = False
        with self.assertRaises(ValueError): validate_space(self.space, "REAL")

    def test_reserved_and_observed_recipes_excluded(self):
        rows = [sample()]
        state = fit_model(rows, method(), rows[0])
        first, _ = recommend(state, rows, rows[0], self.space, "REAL", 6)
        second, _ = recommend(state, rows, rows[0], self.space, "REAL", 6, first)
        self.assertFalse({recipe_key(r["recipe"]) for r in first} & {recipe_key(r["recipe"]) for r in second})
        self.assertNotIn(recipe_key(rows[0]), {recipe_key(r["recipe"]) for r in first + second})

    def test_explicit_baseline_repeat_only(self):
        s = sample()
        state = fit_model([s], method(), s)
        narrow = {**self.space, "alpha_min": 1, "alpha_max": 1, "pulses": [1], "baseline_repeat": True}
        rows, note = recommend(state, [s], s, narrow, "REAL", 6)
        self.assertEqual(len(rows), 1)
        self.assertIn("repeat", rows[0]["purpose"])
        second, _ = recommend(state, [s], s, narrow, "REAL", 6, rows)
        self.assertEqual(second, [])

    def test_proposals_and_workbook_do_not_copy_previous_outcomes(self):
        s = sample(); state = fit_model([s], method(), s)
        proposals, _ = recommend(state, [s], s, self.space, "REAL")
        self.assertTrue(all("icp" not in r["recipe"] and "xrd" not in r["recipe"] and "qc" not in r["recipe"] for r in proposals))
        content = recommendation_workbook(proposals)
        sr = read_table(content, "results.xlsx", "samples")
        ir = read_table(content, "results.xlsx", "icp")
        restored = samples_from_rows(sr, ir)
        self.assertTrue(all(screen(s, method())["label"] is None for s in restored))
        self.assertTrue(all(r["w_wt_percent"] == "" and r["unit"] == "wt%" for r in ir))

    def test_chat_respects_fixed_context_and_supported_pulses(self):
        s = sample(); state = fit_model([s], method(), s)
        ans = reply("Next batch 6 only 2 pulses", [s], method(), state, s, self.space, "REAL")
        self.assertTrue(all(r["recipe"]["pulse_count"] == 2 for r in ans.recommendations))
        ans = reply("Next batch only 9 pulses", [s], method(), state, s, self.space, "REAL")
        self.assertEqual(ans.recommendations, [])
        ans = reply("Recommend 6, change concentration to 0.2mM", [s], method(), state, s, self.space, "REAL")
        self.assertEqual(ans.recommendations, [])
        for prompt in ("Recommend 0", "Recommend 100"):
            ans = reply(prompt, [s], method(), state, s, self.space, "REAL")
            self.assertEqual(ans.recommendations, [])


if __name__ == "__main__":
    unittest.main()
