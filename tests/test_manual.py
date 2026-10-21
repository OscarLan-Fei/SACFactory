"""Manual revision and feedback tests use temporary synthetic records only."""
import copy
import tempfile
import unittest

from sacfactory.domain import RULE_VERSION, screen
from sacfactory.manual import apply_element_rows, element_rows, pending_samples, save_manual_revisions
from sacfactory.planning import fingerprint, fit_model, recommend
from sacfactory.queueing import prepare_reports
from sacfactory.storage import Store
from test_science import sample, method
from test_transfer import context, space


class ManualRevisionTests(unittest.TestCase):
    def test_nullable_editor_cells_remain_missing_without_fabricated_measurements(self):
        import numpy as np
        import pandas as pd
        from sacfactory.ui_manual import clean_cells
        sample = self.store.get('sample', self.ids[0])
        rows = element_rows(sample)
        rows[0].update(value=pd.NA, flag=pd.NA, method_id=pd.NA, source=pd.NA)
        cleaned = clean_cells(rows)
        self.assertTrue(all(cleaned[0][key] is None for key in ('value', 'flag', 'method_id', 'source')))
        updated = apply_element_rows(sample, cleaned)
        self.assertEqual(updated['icp'][0]['value'], '')
        self.assertEqual(updated['icp'][0]['flag'], 'MISSING')
        self.assertEqual(screen(updated, {})['status'], 'PENDING')
        self.assertIsNone(clean_cells([{'value': np.float32(np.nan)}])[0]['value'])

    def test_precursor_identity_changes_persist_and_update_pending_recipe(self):
        from sacfactory.precursor_catalog import planning_draft
        old = self.store.get('sample', self.ids[0])
        updated = copy.deepcopy(old)
        updated['precursors'] = planning_draft(updated['elements'])
        result = self.save([updated], 'pending')
        saved = self.store.get('sample', self.ids[0])
        queue = self.store.get('recommendation', self.proposals[0]['recommendation_id'])
        self.assertEqual(result['samples'], 1)
        self.assertEqual(saved['precursors'], updated['precursors'])
        self.assertEqual(queue['recipe']['precursors'], updated['precursors'])
        self.assertTrue(queue['prediction_stale'])
        self.assertTrue(saved['proposal']['prediction_stale'])
        self.assertEqual(queue['original_proposal']['recipe'], old['proposal']['recipe'])
        prepare_reports([saved], self.store, {})

    def test_actual_precursor_confirmation_survives_record_and_feedback_edits(self):
        from sacfactory.precursor_catalog import planning_draft
        from sacfactory.precursors import confirm_catalog_precursors, precursor_coverage
        for ident, mode in zip(self.ids, ['records', 'feedback']):
            old = self.store.get('sample', ident)
            old['precursors'] = planning_draft(old['elements'])
            self.store.put_many('sample', [old], 'sample_id', overwrite=True)
            confirmed = confirm_catalog_precursors(old)
            self.save([confirmed], mode)
            stored = self.store.get('sample', ident)
            self.assertEqual(stored['precursors'], confirmed['precursors'])
            self.assertEqual(precursor_coverage(stored, for_training=True)[0]['type_used_by_model'], 'CHLORIDE')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(self.tmp.name, "REAL")
        self.context, self.space = context(), space()
        rows, _ = recommend(fit_model([], {}, self.context, transfer=True), [], self.context, self.space, "REAL", 2)
        for row in rows:
            row["snapshot"] = fingerprint({"samples": [], "methods": {}, "context": self.context, "space": self.space, "rule_version": RULE_VERSION})
        self.proposals = rows
        self.ids = self.store.enqueue_pending(rows, self.context, self.space)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, updated, mode="feedback", originals=None):
        originals = originals or [self.store.get("sample", s["sample_id"]) for s in updated]
        return save_manual_revisions(self.store, updated, {}, mode, {s["sample_id"]: fingerprint(s) for s in originals})

    def test_pending_amendment_preserves_original_and_clears_prediction(self):
        old = self.store.get("sample", self.ids[0])
        changed = copy.deepcopy(old)
        changed.update(pulse_count=7, pulse_limit_C=1050, elements=["Fe", "Co"], concentrations_mM={"Fe": .02, "Co": .04}, notes="Manual planned conditions")
        result = self.save([changed], "pending")
        self.assertEqual(result["open_tasks"], 1)
        queue = self.store.get("recommendation", self.proposals[0]["recommendation_id"])
        self.assertEqual(queue["recipe"]["concentrations_mM"], {"Fe": .02, "Co": .04})
        self.assertEqual(queue["original_proposal"]["recipe"], self.proposals[0]["recipe"])
        self.assertIsNone(queue["p_screen_pass"])
        self.assertTrue(queue["prediction_stale"])
        saved = self.store.get("sample", self.ids[0])
        self.assertEqual(saved["proposal"]["recipe"], old["proposal"]["recipe"])
        self.assertEqual(saved["icp"], [])
        self.assertIsNone(saved["temperature_C"])
        self.assertEqual(screen(saved, {})["status"], "PENDING")
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0003"])

    def test_batch_return_completes_tasks_without_numerical_assays(self):
        revised = []
        for ident in self.ids:
            s = self.store.get("sample", ident)
            s.update(batch_id="ACTUAL_BATCH_1", temperature_C=1015, icp_review={"state": "PASS"}, xrd={"state": "PASS", "mode": "MANUAL"})
            revised.append(s)
        result = self.save(revised)
        self.assertEqual(result["completed_tasks"], 2)
        self.assertEqual(pending_samples(self.store.all("sample"), self.store.all("recommendation"), {}), [])
        self.assertEqual(fit_model(self.store.all("sample"), {}, self.context, transfer=True).info["n_labeled"], 2)

    def test_partial_feedback_stays_open_and_revised_outcomes_reopen_tasks(self):
        s = self.store.get("sample", self.ids[0])
        s.update(batch_id="ACTUAL_BATCH", icp_review={"state": "PASS"})
        self.assertEqual(self.save([s])["open_tasks"], 1)
        s = self.store.get("sample", self.ids[0])
        s["xrd"] = {"state": "PASS", "mode": "MANUAL"}
        self.assertEqual(self.save([s])["completed_tasks"], 1)
        s = self.store.get("sample", self.ids[0])
        s["xrd"]["state"] = "PENDING"
        self.assertEqual(self.save([s], "records")["open_tasks"], 1)
        self.assertIn(self.ids[0], [s["sample_id"] for s in pending_samples(self.store.all("sample"), self.store.all("recommendation"), {})])

    def test_actual_condition_correction_retains_deviation_and_later_import_link(self):
        s = self.store.get("sample", self.ids[0])
        s.update(pulse_count=9, batch_id="ACTUAL_BATCH", icp_review={"state": "PASS"}, xrd={"state": "PASS", "mode": "MANUAL"})
        self.assertEqual(self.save([s], "records")["completed_tasks"], 1)
        stored = self.store.get("sample", s["sample_id"])
        self.assertIn("reported_recipe_deviation", stored)
        self.assertEqual(stored["proposal"]["recipe"], s["proposal"]["recipe"])
        revised, updates = prepare_reports([stored], self.store, {})
        self.assertEqual(revised[0]["pulse_count"], 9)
        another = copy.deepcopy(stored)
        another["pulse_count"] = 10
        with self.assertRaisesRegex(ValueError, "differs"):
            prepare_reports([another], self.store, {})

    def test_pending_rejects_measurements_and_changes_after_synthesis(self):
        s = self.store.get("sample", self.ids[0])
        s["temperature_C"] = 1000
        with self.assertRaisesRegex(ValueError, "Result feedback"):
            self.save([s], "pending")
        self.save([s])
        s = self.store.get("sample", self.ids[0])
        s["pulse_count"] += 1
        with self.assertRaisesRegex(ValueError, "already been measured"):
            self.save([s], "pending")

    def test_stale_editor_and_invalid_batch_roll_back_every_write(self):
        originals = [self.store.get("sample", ident) for ident in self.ids]
        changed = copy.deepcopy(originals)
        changed[0]["notes"] = "First correction"
        self.save([changed[0]], "pending")
        before = self.store.export()
        with self.assertRaisesRegex(ValueError, "changed while editing"):
            self.save(changed, "pending", originals)
        self.assertEqual(self.store.export(), before)
        current = [self.store.get("sample", ident) for ident in self.ids]
        current[0]["notes"] = "Should roll back"
        current[1]["pulse_count"] = -1
        with self.assertRaises(ValueError):
            self.save(current)
        self.assertEqual(self.store.export(), before)

    def test_numeric_conflict_does_not_complete_task(self):
        s = self.store.get("sample", self.ids[0])
        element = s["elements"][0]
        s.update(batch_id="ACTUAL_BATCH", icp=[{"element": element, "value": "0.001", "unit": "wt%", "flag": "MEASURED", "method_id": "M1"}],
                 icp_review={"state": "PASS"}, xrd={"state": "PASS", "mode": "MANUAL"})
        m = method()["M1"]
        m["limits"] = {element: {"lod": .0001, "loq": .01}}
        result = save_manual_revisions(self.store, [s], {"M1": m}, "feedback", {s["sample_id"]: fingerprint(self.store.get("sample", s["sample_id"]))})
        self.assertEqual(result["completed_tasks"], 0)
        self.assertEqual(result["open_tasks"], 1)

    def test_element_edit_preserves_zero_and_censoring_without_scale_changes(self):
        s = self.store.get("sample", self.ids[0])
        rows = element_rows(s)
        for value, flag in ((0, "MEASURED"), ("0.25", "MEASURED"), ("<0.005", "MEASURED"), ("ND", "ND"), ("BLQ", "BLQ")):
            rows[0].update(value=value, flag="MISSING")
            updated = apply_element_rows(s, rows)
            self.assertEqual(updated["icp"][0]["value"], str(value))
            self.assertEqual(updated["icp"][0]["flag"], flag)
            self.assertEqual(updated["icp"][0]["unit"], "wt%")

    def test_namespace_and_removed_identity_are_rejected(self):
        s = self.store.get("sample", self.ids[0])
        s["origin"] = "DEMO"
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            self.save([s])
        self.store.delete_samples([self.ids[0]])
        with self.assertRaisesRegex(ValueError, "no longer exists|changed while editing"):
            self.save([s], originals=[s])

    def test_non_target_assays_survive_manual_target_table_edit(self):
        s = self.store.get("sample", self.ids[0])
        s["icp"] = [{"element": "Cd", "value": "ND", "unit": "wt%", "flag": "ND", "method_id": "", "source": "Original report"}]
        updated = apply_element_rows(s, element_rows(s))
        self.assertEqual(updated["icp"][-1], s["icp"][0])
        self.assertNotIn("Cd", updated["elements"])


if __name__ == "__main__":
    unittest.main()
