"""Observed material outcomes and collision-safe display-ID realignment."""
import copy
import tempfile
import unittest

from sacfactory.domain import validate_sample, screen, summary, RULE_VERSION
from sacfactory.analysis import experiment_frame, metrics
from sacfactory.importers import sample_tables, csv_bytes, SAMPLE_COLUMNS, ICP_COLUMNS, samples_from_rows
from sacfactory.ingestion import preview_import, commit_import
from sacfactory.manual import save_manual_revisions
from sacfactory.planning import fingerprint, fit_model, recommend
from sacfactory.queueing import prepare_reports
from sacfactory.storage import Store
from test_science import sample, method
from test_transfer import context, space


class MaterialTests(unittest.TestCase):
    def test_burnout_is_negative_without_fabricating_assays(self):
        s = sample()
        s.update(sample_condition="BURNED_OUT", icp=[], icp_review={"state": "PENDING"}, xrd={"state": "PENDING", "mode": "MANUAL"})
        result = screen(validate_sample(s), {})
        self.assertEqual((result["status"], result["label"], result["icp"], result["xrd"]), ("BURNED_OUT", 0, "PENDING", "PENDING"))
        self.assertEqual(summary([s], {})["labeled"], 1)
        frame = experiment_frame([s], {s["sample_id"]: result})
        self.assertEqual(metrics(frame)["failed"], 1)
        self.assertEqual(fit_model([s], {}, s).info["n_fail"], 1)

    def test_dual_failure_never_infers_burnout(self):
        s = sample()
        s.update(icp_review={"state": "FAIL"}, xrd={"state": "FAIL", "mode": "MANUAL"})
        result = screen(s, {})
        self.assertEqual(result["status"], "BOTH_FAIL")
        self.assertEqual(result["sample_condition"], "NOT_RECORDED")

    def test_invalid_runs_have_no_training_label_even_with_fail_judgments(self):
        for condition in ("ACCIDENTAL_LOSS", "EQUIPMENT_ABORT", "INSUFFICIENT_SAMPLE"):
            s = sample()
            s.update(sample_condition=condition, icp_review={"state": "FAIL"}, xrd={"state": "FAIL", "mode": "MANUAL"})
            result = screen(validate_sample(s), {})
            self.assertEqual((result["status"], result["label"]), ("RETEST", None))
            self.assertEqual(fit_model([s], {}, s).info["n_labeled"], 0)

    def test_partial_loss_can_pass_and_condition_survives_csv_roundtrip(self):
        s = sample()
        s["sample_condition"] = "PARTIAL_LOSS"
        self.assertEqual(screen(s, method())["label"], 1)
        ss, ii = sample_tables([s])
        restored = samples_from_rows(ss, ii)[0]
        self.assertEqual(restored["sample_condition"], "PARTIAL_LOSS")
        s["sample_condition"] = "GUESS"
        with self.assertRaises(ValueError):
            validate_sample(s)


class RealignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(self.tmp.name, "REAL")

    def tearDown(self):
        self.tmp.cleanup()

    def register(self, ids):
        self.store.put_many("sample", [{**sample(i), "sample_id": ident, "notes": "Physical sample " + ident} for i, ident in enumerate(ids)], "sample_id")

    def pending(self, count=3):
        ctx, operating = context(), space()
        rows, _ = recommend(fit_model([], {}, ctx, transfer=True), [], ctx, operating, "REAL", count)
        for row in rows:
            row["snapshot"] = fingerprint({"samples": [], "methods": {}, "context": ctx, "space": operating, "rule_version": RULE_VERSION})
        ids = self.store.enqueue_pending(rows, ctx, operating)
        return rows, ids

    def test_delete_middle_compacts_in_numeric_order_and_continues_next_id(self):
        self.register(["SAC_0001", "SAC_0002", "SAC_0003", "SAC_0004", "CUSTOM_A"])
        self.store.delete_samples(["SAC_0002"])
        self.assertEqual({s["sample_id"] for s in self.store.all("sample")}, {"SAC_0001", "SAC_0002", "SAC_0003", "CUSTOM_A"})
        self.assertEqual(self.store.get("sample", "SAC_0002")["notes"], "Physical sample SAC_0003")
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0004"])
        self.assertEqual(Store(self.tmp.name, "REAL").next_sample_ids(), ["SAC_0004"])
        self.assertEqual(len({s["record_uid"] for s in self.store.all("sample")}), 4)

    def test_pending_links_follow_physical_samples_and_deleted_task_stays_cancelled(self):
        rows, ids = self.pending()
        old_recipe = self.store.get("sample", ids[2])["concentrations_mM"]
        self.store.delete_samples([ids[1]])
        moved = self.store.get("sample", "SAC_0002")
        self.assertEqual(moved["recommendation_id"], rows[2]["recommendation_id"])
        self.assertEqual(moved["concentrations_mM"], old_recipe)
        self.assertEqual(self.store.get("recommendation", rows[2]["recommendation_id"])["pending_sample_id"], "SAC_0002")
        cancelled = self.store.get("recommendation", rows[1]["recommendation_id"])
        self.assertEqual(cancelled["state"], "CANCELLED")
        self.assertNotIn("pending_sample_id", cancelled)
        moved.update(batch_id="ACTUAL_AFTER_REALIGN", sample_condition="BURNED_OUT")
        revised, updates = prepare_reports([moved], self.store, {})
        self.store.import_bundle(revised, [], overwrite=True, queue_updates=updates)
        self.assertEqual(self.store.get("recommendation", rows[2]["recommendation_id"])["state"], "COMPLETED")
        self.assertEqual(self.store.get("recommendation", rows[1]["recommendation_id"])["state"], "CANCELLED")

    def test_sources_context_chat_and_history_follow_renumbering_without_changing_prose(self):
        self.register(["SAC_0001", "SAC_0002", "SAC_0003"])
        self.store.put_many("meta", [{"origin": "REAL", "id": "plan_context", "context": self.store.get("sample", "SAC_0003")}], "id")
        self.store.put_many("recommendation", [{"origin": "REAL", "recommendation_id": "PLAN", "state": "OPEN", "source_sample_ids": ["SAC_0002", "SAC_0003"], "bridge_sources": {"Fe": {"sample_id": "SAC_0003"}}}], "recommendation_id")
        self.store.message("assistant", "Historical SAC_0003", {"sample_id": "SAC_0003"})
        self.store.delete_samples(["SAC_0002"])
        self.assertEqual(self.store.get("meta", "plan_context")["context"]["sample_id"], "SAC_0002")
        queue = self.store.get("recommendation", "PLAN")
        self.assertEqual(queue["source_sample_ids"], ["SAC_0002"])
        self.assertEqual(queue["retired_source_sample_ids"], ["SAC_0002"])
        self.assertEqual(queue["bridge_sources"]["Fe"]["sample_id"], "SAC_0002")
        self.assertEqual(self.store.messages()[0]["content"], "Historical SAC_0003")
        self.assertEqual(self.store.messages()[0]["payload"]["sample_id"], "SAC_0002")
        audits = [a for a in self.store.export()["audit"] if a["kind"] == "sample" and a["id"] == "SAC_0002"]
        self.assertTrue(all(a["action"] != "delete" for a in audits))
        self.assertTrue(any('"sample_id": "SAC_0003"' in (a["before_json"] or a["after_json"] or "") for a in audits))

    def test_old_revision_without_uid_or_with_wrong_uid_is_rejected(self):
        self.register(["SAC_0001", "SAC_0002", "SAC_0003"])
        old = self.store.get("sample", "SAC_0002")
        self.store.delete_samples(["SAC_0001"])
        ss, ii = sample_tables([old])
        files = [("samples.csv", csv_bytes(ss, SAMPLE_COLUMNS)), ("icp.csv", csv_bytes(ii, ICP_COLUMNS))]
        with self.assertRaisesRegex(ValueError, "realigned"):
            preview_import(files, self.store, overwrite=True)
        current = self.store.get("sample", "SAC_0001")
        ss, ii = sample_tables([current])
        files = [("samples.csv", csv_bytes(ss, SAMPLE_COLUMNS)), ("icp.csv", csv_bytes(ii, ICP_COLUMNS))]
        commit_import(preview_import(files, self.store, overwrite=True), self.store)
        wrong = copy.deepcopy(current)
        wrong["record_uid"] = self.store.get("sample", "SAC_0002")["record_uid"]
        with self.assertRaises(ValueError):
            self.store.put_many("sample", [wrong], "sample_id", overwrite=True)

    def test_repeated_delete_keeps_uids_and_delete_all_restarts_at_one(self):
        self.register(["SAC_0010", "SAC_0020", "SAC_0030"])
        self.store.realign_sample_ids()
        uid = self.store.get("sample", "SAC_0003")["record_uid"]
        self.store.delete_samples(["SAC_0001"])
        self.assertEqual(self.store.get("sample", "SAC_0002")["record_uid"], uid)
        self.store.delete_samples(["SAC_0001"])
        self.assertEqual(self.store.get("sample", "SAC_0001")["record_uid"], uid)
        self.store.delete_samples(["SAC_0001"])
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0001"])

    def test_failed_multi_delete_rolls_back_ids_links_and_audits(self):
        self.register(["SAC_0001", "SAC_0003"])
        before = self.store.export()
        with self.assertRaises(ValueError):
            self.store.delete_samples(["SAC_0001", "MISSING"])
        self.assertEqual(before, self.store.export())
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0004"])

    def test_new_pending_after_compaction_gets_identity_without_reusing_links(self):
        rows, ids = self.pending(1)
        self.store.delete_samples(ids)
        ctx, operating = context(), space()
        next_rows, _ = recommend(fit_model([], {}, ctx, transfer=True), [], ctx, operating, "REAL", 1, self.store.all("recommendation"))
        for row in next_rows:
            row["snapshot"] = fingerprint({"samples": [], "methods": {}, "context": ctx, "space": operating, "rule_version": RULE_VERSION})
        new_ids = self.store.enqueue_pending(next_rows, ctx, operating)
        self.assertEqual(new_ids, ["SAC_0001"])
        new = self.store.get("sample", "SAC_0001")
        self.assertTrue(new["record_uid"])
        self.assertNotEqual(new["recommendation_id"], rows[0]["recommendation_id"])
        self.assertEqual(self.store.get("recommendation", new["recommendation_id"])["pending_record_uid"], new["record_uid"])

    def test_legacy_deleted_task_links_are_detached_before_number_reuse(self):
        self.register(["SAC_0001", "SAC_0002"])
        self.store.put_many("recommendation", [{"origin": "REAL", "recommendation_id": "OLD_PLAN", "state": "CANCELLED", "pending_sample_id": "SAC_0003", "deletion_note": "Linked sample was deleted"}], "recommendation_id")
        self.store.realign_sample_ids()
        queue = self.store.get("recommendation", "OLD_PLAN")
        self.assertNotIn("pending_sample_id", queue)
        self.assertEqual(queue["deleted_pending_sample_id"], "SAC_0003")
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0003"])


if __name__ == "__main__":
    unittest.main()
