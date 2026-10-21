"""Transfer is a traceable hypothesis, with source/target and validation isolation."""
import copy
import unittest

import numpy as np
from sklearn.base import clone

from sacfactory.domain import ELEMENTS, recipe
from sacfactory.planning import fit_model, recommend, recommendation_table
from sacfactory.transfer import POSITIONS, descriptors, transfer_feature, ElementTransferKernel
from test_science import sample


def source(i, element="Fe", passed=True, concentration=None):
    s = sample(i)
    s.update(elements=[element], concentrations_mM={element: concentration or .015 + i * .008}, pulse_limit_C=900 + (i % 4) * 50, temperature_C=1020,
             icp=[], icp_review={"state": "PASS" if passed else "FAIL"}, xrd={"state": "PASS", "mode": "MANUAL"})
    return s


def context(element="Co"):
    s = source(100, element)
    s.update(concentrations_mM={element: .05}, pulse_limit_C=1000)
    return s


def space(element="Co"):
    return {"concentration_bounds_mM": {element: {"min_mM": .01, "max_mM": .2}}, "pulses": [1, 2, 3],
            "pulse_limit_min_C": 900, "pulse_limit_max_C": 1050, "max_total_mM": .2, "verified": True, "baseline_repeat": False}


class TransferTests(unittest.TestCase):
    def test_periodic_positions_cover_all_elements_without_detached_group_invention(self):
        self.assertEqual(set(POSITIONS), set(ELEMENTS))
        self.assertEqual(POSITIONS["Fe"][:2], (4, 8))
        self.assertEqual(POSITIONS["Co"][:2], (4, 9))
        self.assertIsNone(POSITIONS["La"][1])
        self.assertEqual(len(descriptors("Og")), 9)

    def test_kernel_is_positive_semidefinite_and_cloneable(self):
        x = np.array([transfer_feature(source(i, element)) for i, element in enumerate(("Fe", "Co", "La", "Cs", "Og"))])
        kernel = clone(ElementTransferKernel())
        matrix, gradient = kernel(x, eval_gradient=True)
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-10)
        self.assertEqual(gradient.shape, (5, 5, 0))
        self.assertTrue(np.allclose(np.diag(matrix), kernel.diag(x)))
        self.assertTrue(np.allclose(matrix, kernel(x, x)))

    def test_fe_data_fit_and_predict_unseen_co_with_provenance(self):
        rows = [source(i, passed=i >= 8) for i in range(18)]
        state = fit_model(rows, {}, context(), transfer=True)
        self.assertIsNotNone(state.model)
        self.assertEqual(state.info["source_elements"], {"Fe": 18})
        self.assertEqual(state.info["target_labeled_n"], 0)
        self.assertIsNone(state.info["transfer_validation"])
        recs, _ = recommend(state, rows, context(), space(), "REAL", 3)
        self.assertEqual(len(recs), 3)
        self.assertTrue(all(0 < r["p_screen_pass"] < 1 for r in recs))
        self.assertTrue(all(r["chemistry_extrapolation"] for r in recs))
        self.assertTrue(all(r["source_elements"] == ["Fe"] for r in recs))
        self.assertTrue(all(r["prediction_mode"] == "TRANSFER_HYPOTHESIS" for r in recs))
        self.assertIn("source_sample_ids", recommendation_table(recs)[0])

    def test_one_fe_success_guides_co_bridge_without_fake_probability(self):
        rows = [source(0, concentration=.08)]
        state = fit_model(rows, {}, context(), transfer=True)
        recs, _ = recommend(state, rows, context(), space(), "REAL", 3)
        self.assertIsNone(state.model)
        self.assertEqual(recs[0]["recipe"]["concentrations_mM"], {"Co": .08})
        self.assertEqual(recs[0]["recipe"]["pulse_limit_C"], 900)
        self.assertIn("bridge", recs[0]["purpose"])
        self.assertTrue(all(r["p_screen_pass"] is None for r in recs))
        self.assertEqual(recs[0]["source_sample_ids"], ["S0"])

    def test_source_rows_are_filtered_for_process_and_namespace_while_mixtures_join(self):
        rows = [source(i) for i in range(6)]
        rows[1]["origin"] = "DEMO"
        rows[2]["support_batch"] = "OTHER"
        rows[3].update(elements=["Fe", "Co"], concentrations_mM=recipe(["Fe", "Co"]))
        rows[4]["batch_id"] = "UNKNOWN"
        rows[5]["icp_review"]["state"] = "PENDING"
        state = fit_model(rows, {}, context(), transfer=True)
        self.assertEqual(state.info["n_labeled"], 2)
        self.assertEqual(len(state.info["exclusions"]), 4)

    def test_target_only_is_separate_and_transfer_pools_into_mixtures(self):
        rows = [source(i, passed=i > 8) for i in range(18)]
        self.assertIsNone(fit_model(rows, {}, context(), transfer=False).model)
        mixed = context()
        mixed.update(elements=["Fe", "Co"], concentrations_mM=recipe(["Fe", "Co"]))
        state = fit_model(rows, {}, mixed, transfer=True)
        self.assertIsNotNone(state.model)
        self.assertEqual(state.info["feature_kind"], "COMPOSITION_TRANSFER")
        self.assertEqual(state.info["target_labeled_n"], 0)

    def test_co_results_join_source_pool_and_update_target_count(self):
        rows = [source(i, passed=i >= 8) for i in range(18)] + [source(30, "Co")]
        state = fit_model(rows, {}, context(), transfer=True)
        self.assertEqual(state.info["target_labeled_n"], 1)
        self.assertFalse(state.info["unseen_target"])
        self.assertEqual(len(state.training), 19)

    def test_held_out_elements_do_not_cross_shared_batches(self):
        rows = [source(i, element, i % 18 >= 8) for i, element in enumerate(["Fe"] * 18 + ["Co"] * 18)]
        state = fit_model(rows, {}, context("Ni"), transfer=True)
        self.assertIsNotNone(state.info["transfer_validation"])
        self.assertTrue(state.info["transfer_validation"]["full_coverage"])
        for s in rows: s["batch_id"] = "SHARED_BATCH"
        for i, s in enumerate(rows):
            if i < 8: continue
            s["batch_id"] = f"INDEPENDENT_{i}"
        state = fit_model(rows, {}, context("Ni"), transfer=True)
        self.assertIsNotNone(state.model)
        self.assertIn("purges", state.info["transfer_validation"]["type"])

    def test_features_never_use_assays_or_measured_peak(self):
        s = source(0)
        before = transfer_feature(s)
        s.update(temperature_C=5000, voltage_V=500, current_A=50, icp_review={"state": "FAIL"}, xrd={"state": "FAIL"})
        self.assertEqual(before, transfer_feature(s))


if __name__ == "__main__":
    unittest.main()
