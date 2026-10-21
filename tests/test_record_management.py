import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from sacfactory.storage import Store
from sacfactory.bootstrap import seed
from test_science import sample


class RecordManagementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(self.tmp.name, "REAL")

    def tearDown(self):
        self.tmp.cleanup()

    def samples(self, ids):
        return [{**sample(i), "sample_id": ident} for i, ident in enumerate(ids)]

    def test_two_batches_continue_numbering_and_restart_preserves_sequence(self):
        first = self.store.next_sample_ids(10)
        self.assertEqual((first[0], first[-1]), ("SAC_0001", "SAC_0010"))
        self.assertEqual(self.store.next_sample_ids(10), first)
        self.store.import_bundle(self.samples(first), [])
        second = Store(self.tmp.name, "REAL").next_sample_ids(10)
        self.assertEqual((second[0], second[-1]), ("SAC_0011", "SAC_0020"))
        self.store.import_bundle(self.samples(second), [])
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0021"])

    def test_manual_imports_advance_numeric_high_water_and_expand_beyond_four_digits(self):
        self.store.put_many("sample", self.samples(["CUSTOM_ID", "SAC_001", "SAC_0120"]), "sample_id")
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0121"])
        self.store.import_bundle(self.samples(["SAC_9999"]), [])
        self.assertEqual(self.store.next_sample_ids(), ["SAC_10000"])

    def test_delete_removes_record_revisions_without_reusing_sample_numbers(self):
        original = self.samples(["SAC_0001", "SAC_0002"])
        self.store.put_many("sample", original, "sample_id")
        self.store.put_many("recommendation", [{"origin": "REAL", "recommendation_id": "PLAN_1", "state": "COMPLETED", "reported_sample_id": "SAC_0002"}], "recommendation_id")
        self.store.delete_samples(["SAC_0002"], realign=False)
        self.assertIsNone(self.store.get("sample", "SAC_0002"))
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0003"])
        self.assertEqual(self.store.get("recommendation", "PLAN_1")["state"], "CANCELLED")
        history = [r for r in self.store.export()["audit"] if r["kind"] == "sample" and r["id"] == "SAC_0002"]
        self.assertEqual(len(history), 1)
        self.assertIsNone(history[0]["before_json"])
        self.assertIsNone(history[0]["after_json"])
        self.store.delete_samples(["SAC_0001"], realign=False)
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0003"])
        self.assertEqual(self.store.all("sample"), [])

    def test_deletion_and_import_collisions_are_atomic(self):
        self.store.put_many("sample", self.samples(["SAC_0001"]), "sample_id")
        with self.assertRaises(ValueError):
            self.store.delete_samples(["SAC_0001", "MISSING"])
        self.assertIsNotNone(self.store.get("sample", "SAC_0001"))
        self.assertFalse(any(r["action"] == "delete" for r in self.store.export()["audit"]))
        with self.assertRaises(ValueError):
            self.store.import_bundle(self.samples(["SAC_0002", "SAC_0001"]), [])
        self.assertIsNone(self.store.get("sample", "SAC_0002"))
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0002"])

    def test_clear_without_backup_reseeding_or_touching_demo(self):
        self.store.put_many("sample", self.samples(["SAC_0007"]), "sample_id")
        self.store.put_many("meta", [{"id": "plan_context", "origin": "REAL", "context": copy.deepcopy(sample())}], "id")
        self.store.message("assistant", "Old REAL history")
        raw = self.store.source("fixture.csv", b"source fixture")
        demo = Store(self.tmp.name, "DEMO")
        demo_sample = sample(origin="DEMO")
        demo.put_many("sample", [demo_sample], "sample_id")
        self.store.clear_namespace()
        self.assertEqual(self.store.all("sample"), [])
        self.assertEqual(self.store.messages(), [])
        self.assertEqual(self.store.export()["audit"], [])
        self.assertEqual(self.store.all("method"), [])
        self.assertEqual(self.store.next_sample_ids(), ["SAC_0001"])
        self.assertIsNone(self.store.get("meta", "plan_context"))
        seed(self.store, Path(__file__).resolve().parents[1])
        self.assertEqual(self.store.all("sample"), [])
        self.assertEqual(demo.all("sample"), [demo_sample])
        self.assertFalse(Path(raw["file"]).exists())
        self.assertFalse((Path(self.tmp.name) / "backups").exists())


if __name__ == "__main__":
    unittest.main()
