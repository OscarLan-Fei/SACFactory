"""Atomic pending creation and report feedback never fabricate measurements."""
import copy
import tempfile
import unittest

from sacfactory.domain import screen, RULE_VERSION
from sacfactory.planning import fit_model, recommend, fingerprint
from sacfactory.queueing import prepare_reports
from sacfactory.storage import Store
from sacfactory.importers import sample_tables, csv_bytes, SAMPLE_COLUMNS, ICP_COLUMNS
from sacfactory.ingestion import preview_import, commit_import
from test_transfer import context, space


class PendingTests(unittest.TestCase):
    def test_changed_campaign_policy_rejects_previously_generated_preview(self):
        rows = self.preview(1)
        self.store.put_many('meta', [{'id': 'element_campaign', 'origin': 'REAL',
            'enabled': True, 'strategy': 'SINGLE_ELEMENT_FIRST'}], 'id')
        before = self.store.export()
        with self.assertRaisesRegex(ValueError, 'Planning settings changed'):
            self.store.enqueue_pending(rows, self.context, self.space)
        self.assertEqual(self.store.export(), before)

    def test_restored_catalog_and_range_defaults_reserve_but_actual_changes_still_reject(self):
        from sacfactory.precursor_catalog import register_defaults, resolve_planning_precursors
        from sacfactory.operating_space import restore_planning_space
        old_context, old_space = copy.deepcopy(self.context), copy.deepcopy(self.space)
        library = register_defaults(self.store)
        self.store.put_many('meta', [
            {'id': 'plan_context', 'origin': 'REAL', 'context': old_context},
            {'id': 'space', 'origin': 'REAL', 'space': old_space}], 'id')
        self.context['precursors'] = resolve_planning_precursors(library, old_context, old_context['elements'])
        self.space = restore_planning_space(old_space, self.context, [], {}, 'REAL')
        rows = self.preview(1)
        ids = self.store.enqueue_pending(rows, self.context, self.space)
        self.assertEqual(len(ids), 1)
        self.assertEqual(self.store.get('sample', ids[0])['precursors'], self.context['precursors'])
        fresh = self.preview(1)
        changed = copy.deepcopy(old_space)
        changed['pulses'] = [99]
        self.store.put_many('meta', [{'id': 'space', 'origin': 'REAL', 'space': changed}], 'id', overwrite=True)
        before = self.store.export()
        with self.assertRaisesRegex(ValueError, 'Planning settings changed'):
            self.store.enqueue_pending(fresh, self.context, self.space)
        self.assertEqual(self.store.export(), before)

    def test_changed_reagent_default_invalidates_old_catalog_preview_even_if_context_was_saved(self):
        from sacfactory.precursor_catalog import register_defaults, resolve_planning_precursors
        library = register_defaults(self.store)
        self.context['precursors'] = resolve_planning_precursors(library, self.context, self.context['elements'])
        self.store.put_many('meta', [{'id': 'plan_context', 'origin': 'REAL', 'context': self.context},
            {'id': 'space', 'origin': 'REAL', 'space': self.space}], 'id')
        rows = self.preview(1)
        element = self.context['elements'][0]
        library[element].update(compound='User-selected alternative', type='OTHER',
            oxidation_state=None, source='Explicit replacement reagent')
        self.store.put_many('meta', [{'id': 'reagent_library', 'origin': 'REAL', 'precursors': library}], 'id', overwrite=True)
        before = self.store.export()
        with self.assertRaisesRegex(ValueError, 'Planning settings changed'):
            self.store.enqueue_pending(rows, self.context, self.space)
        self.assertEqual(self.store.export(), before)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(self.tmp.name, "REAL")
        self.context, self.space = context(), space()

    def tearDown(self):
        self.tmp.cleanup()

    def preview(self, count=3):
        samples = self.store.all("sample")
        methods = {m["method_id"]: m for m in self.store.all("method")}
        state = fit_model(samples, methods, self.context, transfer=True)
        rows, _ = recommend(state, samples, self.context, self.space, "REAL", count, self.store.all("recommendation"))
        for row in rows:
            row["snapshot"] = fingerprint({"context": self.context, "space": self.space, "samples": samples, "methods": methods, "rule_version": RULE_VERSION})
        return rows

    def test_two_pending_batches_continue_ids_and_remain_unmeasured(self):
        first = self.preview(10)
        ids = self.store.enqueue_pending(first, self.context, self.space)
        self.assertEqual(ids, [f"SAC_{i:04d}" for i in range(1, 11)])
        second = self.preview(2)
        self.assertEqual(self.store.enqueue_pending(second, self.context, self.space), ["SAC_0011", "SAC_0012"])
        for s in self.store.all("sample"):
            self.assertEqual(screen(s, {})["status"], "PENDING")
            self.assertEqual(s["batch_id"], "UNKNOWN")
            self.assertIsNone(s["temperature_C"])
            self.assertEqual(s["icp"], [])
            self.assertIsNone(s["voltage_V"])
        self.assertEqual(fit_model(self.store.all("sample"), {}, self.context, transfer=True).info["n_labeled"], 0)

    def test_repeated_import_is_idempotent_even_after_reserving(self):
        rows = self.preview()
        self.store.put_many("recommendation", rows, "recommendation_id")
        ids = self.store.enqueue_pending(rows, self.context, self.space)
        self.assertEqual(self.store.enqueue_pending(rows, self.context, self.space), ids)
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0004"])

    def test_stale_preview_is_rejected_without_consuming_numbers(self):
        rows = self.preview()
        self.store.put_many("sample", [context("Fe")], "sample_id")
        with self.assertRaisesRegex(ValueError, "Data changed"):
            self.store.enqueue_pending(rows, self.context, self.space)
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0001"])
        self.assertEqual(self.store.all("recommendation"), [])

    def test_bad_recipe_duplicate_or_namespace_rolls_back_entire_batch(self):
        for mode in ("duplicate", "out_of_range", "namespace"):
            rows = self.preview()
            if mode == "duplicate": rows[1]["recipe"] = copy.deepcopy(rows[0]["recipe"])
            if mode == "out_of_range": rows[1]["recipe"]["pulse_count"] = 900
            if mode == "namespace": rows[1]["origin"] = "DEMO"
            with self.assertRaises(ValueError): self.store.enqueue_pending(rows, self.context, self.space)
            self.assertEqual(self.store.all("sample"), [])
            self.assertEqual(self.store.all("recommendation"), [])

    def test_actual_batch_and_resolved_result_complete_the_correct_task(self):
        row = self.preview(1)[0]
        ident = self.store.enqueue_pending([row], self.context, self.space)[0]
        s = self.store.get("sample", ident)
        s.update(icp_review={"state": "PASS"}, xrd={"state": "PASS", "mode": "MANUAL"})
        revised, updates = prepare_reports([s], self.store, {})
        self.assertEqual(updates[0]["state"], "OPEN")
        s.update(batch_id="ACTUAL_1", temperature_C=1015)
        revised, updates = prepare_reports([s], self.store, {})
        self.store.import_bundle(revised, [], overwrite=True, queue_updates=updates)
        self.assertEqual(self.store.get("recommendation", row["recommendation_id"])["state"], "COMPLETED")
        self.assertEqual(fit_model(self.store.all("sample"), {}, self.context, transfer=True).info["target_labeled_n"], 1)

    def test_imported_reports_preserve_link_and_complete_via_sac_ids(self):
        rows = self.preview(1)
        ident = self.store.enqueue_pending(rows, self.context, self.space)[0]
        s = self.store.get("sample", ident)
        s.update(batch_id="ACTUAL_IMPORT", icp_review={"state": "PASS"}, xrd={"state": "PASS", "mode": "MANUAL"})
        ss, ii = sample_tables([s])
        files = [("samples.csv", csv_bytes(ss, SAMPLE_COLUMNS)), ("icp.csv", csv_bytes(ii, ICP_COLUMNS))]
        result = commit_import(preview_import(files, self.store, overwrite=True), self.store)
        self.assertEqual(result["completed_tasks"], 1)
        self.assertEqual(self.store.get("sample", ident)["proposal"], s["proposal"])
        self.assertEqual(self.store.get("recommendation", rows[0]["recommendation_id"])["state"], "COMPLETED")

    def test_changed_recipe_requires_new_id_and_deletion_cancels_link(self):
        rows = self.preview(1)
        ident = self.store.enqueue_pending(rows, self.context, self.space)[0]
        s = self.store.get("sample", ident)
        s["pulse_count"] += 1
        with self.assertRaisesRegex(ValueError, "differs"): prepare_reports([s], self.store, {})
        self.store.delete_samples([ident])
        self.assertEqual(self.store.get("recommendation", rows[0]["recommendation_id"])["state"], "CANCELLED")


if __name__ == "__main__":
    unittest.main()
