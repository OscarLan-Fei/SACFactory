"""Single-to-mixture transfer preserves composition, inputs and evidence boundaries."""
import copy
import tempfile
import unittest

import numpy as np
from sklearn.base import clone

from sacfactory.composition import CompositionTransferKernel, transfer_feature, source_evidence, composition_key
from sacfactory.domain import ELEMENTS, TARGETS, recipe, screen, RULE_VERSION
from sacfactory.planning import fit_model, recommend, fingerprint
from sacfactory.storage import Store
from test_transfer import source, context, space


def mixture(i=100, elements=("Fe", "Co"), passed=True, concentrations=None):
    s = source(i, passed=passed)
    s.update(elements=list(elements), concentrations_mM=concentrations or recipe(elements), pulse_limit_C=1000)
    return s


def mixture_space(elements=("Fe", "Co")):
    return {**space(), "concentration_bounds_mM": {e: {"min_mM": recipe([e])[e] * .5, "max_mM": recipe([e])[e] * 1.5} for e in elements},
            "max_total_mM": sum(recipe(elements).values()) * 1.5, "pulses": [1, 2, 3]}


class CompositionTests(unittest.TestCase):
    def sources(self):
        return [source(i, "Fe" if i < 18 else "Co", passed=i % 18 >= 8) for i in range(36)]

    def test_fixed_schema_and_permutation_invariance_for_1_2_3_50_118(self):
        elements = ["Fe", "Co", "Ni"]
        a, b = mixture(elements=elements), mixture(elements=list(reversed(elements)))
        self.assertEqual(transfer_feature(a), transfer_feature(b))
        dimensions = {len(transfer_feature(mixture(elements=members))) for members in (["Fe"], elements[:2], elements, TARGETS, ELEMENTS)}
        self.assertEqual(dimensions, {2 * len(ELEMENTS) + 4})
        with self.assertRaises(ValueError):
            bad = mixture()
            bad["concentrations_mM"]["Fe"] = None
            transfer_feature(bad)

    def test_ratio_and_total_concentration_and_member_count_are_distinct(self):
        a = mixture(concentrations={"Fe": .04, "Co": .06})
        b = mixture(concentrations={"Fe": .06, "Co": .04})
        xa, xb = transfer_feature(a), transfer_feature(b)
        self.assertEqual(xa[-4:], xb[-4:])
        self.assertNotEqual(xa, xb)
        single = source(0, concentration=.1)
        self.assertEqual(transfer_feature(single)[-4], xa[-4])
        self.assertNotEqual(transfer_feature(single)[-3], xa[-3])

    def test_mixed_cardinality_kernel_is_psd_with_unmeasured_pair_variance(self):
        members = [["Fe"], ["Co"], ["Fe", "Co"], ["Fe", "Ni"], ["Fe", "Co", "Ni"], TARGETS, ELEMENTS]
        x = np.array([transfer_feature(mixture(elements=e)) for e in members])
        k = clone(CompositionTransferKernel())
        matrix, gradient = k(x, eval_gradient=True)
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-10)
        self.assertTrue(np.allclose(np.diag(matrix), k.diag(x)))
        self.assertEqual(gradient.shape, (7, 7, 0))
        without_pairs = CompositionTransferKernel(weights=(.45, 0, .35))(x)
        residual = matrix - without_pairs
        self.assertTrue(np.allclose(residual[:2], 0))
        self.assertAlmostEqual(residual[2, 2], .2)

    def test_single_source_models_predict_unseen_mixtures_without_fake_labels(self):
        rows = self.sources()
        before = copy.deepcopy(rows)
        for members in (["Fe", "Co"], ["Fe", "Co", "Ni"], TARGETS):
            ctx = mixture(elements=members)
            state = fit_model(rows, {}, ctx, transfer=True)
            self.assertIsNotNone(state.model)
            self.assertEqual(state.info["target_labeled_n"], 0)
            self.assertEqual(state.info["unobserved_pair_count"], len(members) * (len(members) - 1) // 2)
            proposals, _ = recommend(state, rows, ctx, mixture_space(members), "REAL", 2)
            self.assertEqual(len(proposals), 2)
            self.assertTrue(all(0 < p["p_screen_pass"] < 1 for p in proposals))
            self.assertTrue(all(p["expansion_step"] and p["chemistry_extrapolation"] for p in proposals))
            self.assertTrue(all(set(p["recipe"]["concentrations_mM"]) == set(members) for p in proposals))
            if len(members) == 50:
                self.assertTrue(all(p["support_level"] == "LOW_KERNEL_SUPPORT" for p in proposals))
        self.assertEqual(rows, before)

    def test_sparse_bridge_combines_known_member_recipes_without_product_probability(self):
        rows = [source(0, "Fe", concentration=.04), source(1, "Co", concentration=.06)]
        rows[1]["pulse_limit_C"] = 900
        ctx = mixture()
        state = fit_model(rows, {}, ctx, transfer=True)
        proposals, _ = recommend(state, rows, ctx, mixture_space(), "REAL", 2)
        self.assertIsNone(proposals[0]["p_screen_pass"])
        self.assertEqual(proposals[0]["recipe"]["concentrations_mM"], {"Fe": .04, "Co": .06})
        self.assertEqual(proposals[0]["bridge_sources"]["Fe"]["sample_id"], "S0")
        self.assertEqual(proposals[0]["bridge_sources"]["Co"]["sample_id"], "S1")

    def test_mixture_failures_update_target_posterior_and_stage_validation(self):
        rows = self.sources()
        ctx = mixture(concentrations={"Fe": .05, "Co": .05})
        before = fit_model(rows, {}, ctx, transfer=True)
        probability_before = before.model.predict_proba([transfer_feature(ctx)])[0, 1]
        mixed = [mixture(i=50 + i, passed=False, concentrations={"Fe": .04 + .002 * i, "Co": .06 - .001 * i}) for i in range(12)]
        # Mixed test outcomes include both classes for a meaningful stage check.
        mixed[-1]["icp_review"]["state"] = "PASS"
        after = fit_model(rows + mixed, {}, ctx, transfer=True)
        self.assertLess(after.model.predict_proba([transfer_feature(ctx)])[0, 1], probability_before)
        self.assertEqual(after.info["target_labeled_n"], 12)
        self.assertEqual(after.info["source_stages"], {1: 36, 2: 12})
        self.assertEqual(after.info["unobserved_pair_count"], 0)
        self.assertIsNotNone(after.info["expansion_validation"])
        self.assertEqual(after.info["expansion_validation"]["n"], 12)
        self.assertIn("no equal/higher", after.info["expansion_validation"]["type"])
        next_stage = fit_model(rows + mixed, {}, mixture(elements=["Fe", "Co", "Ni"]), transfer=True)
        self.assertEqual(next_stage.info["highest_source_stage"], 2)
        self.assertEqual(next_stage.info["source_compositions"]["Fe+Co"], 12)
        self.assertEqual(next_stage.info["unobserved_pair_count"], 2)

    def test_pending_mixture_preserves_sources_and_receives_no_measured_result(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, "REAL")
            rows = self.sources()
            store.put_many("sample", rows, "sample_id")
            rows = store.all("sample")
            ctx, operating = mixture(), mixture_space()
            state = fit_model(rows, {}, ctx, transfer=True)
            proposals, _ = recommend(state, rows, ctx, operating, "REAL", 2)
            for p in proposals:
                p["snapshot"] = fingerprint({"context": ctx, "space": operating, "samples": rows, "methods": {}, "rule_version": RULE_VERSION})
            ids = store.enqueue_pending(proposals, ctx, operating)
            s = store.get("sample", ids[0])
            self.assertEqual(set(s["elements"]), {"Fe", "Co"})
            self.assertEqual(screen(s, {})["status"], "PENDING")
            self.assertIsNone(s["temperature_C"])
            self.assertEqual(s["proposal"]["unobserved_pair_count"], 1)
            self.assertEqual(s["proposal"]["bridge_sources"], proposals[0]["bridge_sources"])

    def test_exact_composition_disabled_transfer_is_unchanged(self):
        state = fit_model(self.sources(), {}, mixture(), transfer=False)
        self.assertIsNone(state.model)
        self.assertEqual(state.info["n_labeled"], 0)

    def test_expansion_validation_purges_connected_source_batches(self):
        rows = self.sources()
        mixed = [mixture(i=60 + i, passed=i % 2 == 0) for i in range(8)]
        # Every source belongs to a test batch, so none may leak into the stage fold.
        for i, s in enumerate(rows): s["batch_id"] = mixed[i % 8]["batch_id"]
        state = fit_model(rows + mixed, {}, mixture(), transfer=True)
        self.assertIsNotNone(state.model)
        self.assertIsNone(state.info["expansion_validation"])


if __name__ == "__main__":
    unittest.main()
