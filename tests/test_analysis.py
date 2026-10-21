"""Descriptive analysis preserves missingness and explicit outcome denominators."""
import copy
import unittest

from sacfactory.analysis import experiment_frame, metrics, grouped_rates, parameter_rates, correlations, wilson_interval
from test_science import sample


class AnalysisTests(unittest.TestCase):
    def frame(self, statuses=("PASS", "XRD_FAIL", "PENDING", "ICP_FAIL")):
        samples = [sample(i, alpha=i + 1) for i in range(len(statuses))]
        outcomes = {s["sample_id"]: {"status": status} for s, status in zip(samples, statuses)}
        return experiment_frame(samples, outcomes)

    def test_pending_excluded_from_success_denominator(self):
        stats = metrics(self.frame())
        self.assertEqual((stats["labeled"], stats["passed"], stats["failed"], stats["pending"]), (3, 1, 2, 1))
        self.assertAlmostEqual(stats["pass_rate"], 1 / 3)

    def test_conflicting_record_keeps_status_but_is_excluded(self):
        samples = [sample(0), sample(1)]
        data = experiment_frame(samples, {"S0": {"status": "PASS", "review_conflicts": ["conflict"]}, "S1": {"status": "ICP_FAIL"}})
        self.assertEqual(data.iloc[0]["status"], "PASS")
        self.assertEqual(metrics(data)["conflicts"], 1)
        self.assertEqual(metrics(data)["labeled"], 1)
        self.assertEqual(metrics(data)["pass_rate"], 0)

    def test_no_assay_access_or_record_mutation(self):
        s = sample()
        s["icp"] = "Invalid raw data deliberately ignored by analysis"
        s["xrd"] = "Invalid spectrum deliberately ignored by analysis"
        before = copy.deepcopy(s)
        frame = experiment_frame([s], {s["sample_id"]: {"status": "PASS"}})
        self.assertEqual(metrics(frame)["passed"], 1)
        self.assertEqual(s, before)
        self.assertNotIn("icp", frame)
        self.assertNotIn("xrd", frame)

    def test_missing_conditions_remain_missing_and_zero_is_preserved(self):
        s = sample()
        s.update(temperature_C=None, pulse_limit_C=950, voltage_V=0, current_A=float("inf"))
        s["concentrations_mM"]["Fe"] = None
        frame = experiment_frame([s], {})
        self.assertTrue(frame["temperature_C"].isna().all())
        self.assertTrue(frame["Fe_mM"].isna().all())
        self.assertTrue(frame["current_A"].isna().all())
        self.assertEqual(frame.iloc[0]["voltage_V"], 0)
        self.assertEqual(metrics(frame)["distinct_recipes"], 0)
        self.assertEqual(frame.iloc[0]["pulse_limit_C"], 950)

    def test_empty_and_all_pending_have_no_rate(self):
        for frame in (experiment_frame([], {}), self.frame(("PENDING",))):
            self.assertIsNone(metrics(frame)["pass_rate"])
        self.assertTrue(grouped_rates(experiment_frame([], {}), "batch_id").empty)
        self.assertEqual(wilson_interval(0, 0), (None, None))

    def test_batch_counts_and_replicates_are_distinct(self):
        a, b = sample(0), sample(1)
        b["batch_id"] = ""
        frame = experiment_frame([a, b], {"S0": {"status": "PASS"}, "S1": {"status": "PASS"}})
        self.assertEqual(metrics(frame)["batches"], 1)
        self.assertEqual(metrics(frame)["distinct_recipes"], 1)
        rates = grouped_rates(frame, "batch_id")
        self.assertIn("UNASSIGNED", rates["batch_id"].tolist())
        self.assertEqual(rates["labeled"].sum(), 2)

    def test_wilson_single_pass_and_balanced_interval(self):
        low, high = wilson_interval(1, 1)
        self.assertAlmostEqual(low, .206549, places=5)
        self.assertAlmostEqual(high, 1)
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low + high, 1)

    def test_pairwise_spearman_uses_only_complete_nonconstant_pairs(self):
        frame = self.frame()
        frame["temperature_C"] = [100, 200, 300, None]
        pairs = correlations(frame, ["Fe_mM", "temperature_C", "pulse_count"])
        first = pairs.iloc[0]
        self.assertEqual(first["paired_n"], 3)
        self.assertAlmostEqual(first["spearman_rho"], 1)
        self.assertTrue(pairs.loc[pairs["parameter_2"] == "pulse_count", "spearman_rho"].isna().all())
        self.assertTrue(correlations(frame.iloc[:2], ["Fe_mM", "temperature_C"])["spearman_rho"].isna().all())

    def test_parameter_bins_preserve_counts_and_numeric_order(self):
        frame = self.frame()
        frame["pulse_count"] = [1., 2., 10., None]
        rates = parameter_rates(frame, "pulse_count")
        self.assertEqual(rates["parameter_range"].tolist(), ["1", "2", "10"])
        self.assertEqual(rates["samples"].sum(), 3)
        frame["pulse_count"] = [1., 2., 10., 20.]
        self.assertEqual(parameter_rates(frame, "pulse_count", bins=2)["samples"].sum(), 4)
        frame["Fe_mM"] = [.050000001, .050000002, .05, .05]
        self.assertEqual(len(parameter_rates(frame, "Fe_mM")), 3)


if __name__ == "__main__":
    unittest.main()
