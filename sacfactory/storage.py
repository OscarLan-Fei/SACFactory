from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    """Namespace-per-file SQLite, immutable source copies and append-only revisions."""
    def __init__(self, root, origin):
        if origin not in {"REAL", "DEMO"}:
            raise ValueError("Invalid data namespace")
        self.root, self.origin = Path(root), origin
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{origin.lower()}.sqlite3"
        with self.connect() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS items(kind TEXT, id TEXT, payload TEXT, PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS audit(seq INTEGER PRIMARY KEY, created TEXT, action TEXT, kind TEXT, id TEXT, before_json TEXT, after_json TEXT);
                CREATE TABLE IF NOT EXISTS messages(seq INTEGER PRIMARY KEY, created TEXT, role TEXT, content TEXT, payload TEXT);
            """)
        self._migrate_icp_units()

    def _migrate_icp_units(self):
        """Convert active legacy content once; preserve original files and audits."""
        from .units import normalize_method, normalize_payload
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            if c.execute("SELECT 1 FROM items WHERE kind='meta' AND id='icp-mass-percent-v1'").fetchone():
                return
            changed = 0
            for kind, ident, encoded in c.execute("SELECT kind,id,payload FROM items").fetchall():
                old = json.loads(encoded)
                new = normalize_method(old, legacy=True) if kind == "method" else normalize_payload(old)
                if new != old:
                    updated = json.dumps(new, ensure_ascii=False, allow_nan=False)
                    c.execute("UPDATE items SET payload=? WHERE kind=? AND id=?", (updated, kind, ident))
                    c.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'convert_icp_unit',?,?,?,?)", (now(), kind, ident, encoded, updated))
                    changed += 1
            marker = {"id": "icp-mass-percent-v1", "origin": self.origin, "unit": "wt%", "converted_items": changed}
            c.execute("INSERT INTO items VALUES('meta','icp-mass-percent-v1',?)", (json.dumps(marker),))
            if changed:
                epoch = {"id": "data-epoch", "origin": self.origin, "value": uuid4().hex}
                c.execute("INSERT OR REPLACE INTO items VALUES('meta','data-epoch',?)", (json.dumps(epoch),))

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=20)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def all(self, kind):
        with self.connect() as c:
            return [json.loads(r[0]) for r in c.execute("SELECT payload FROM items WHERE kind=? ORDER BY id", (kind,))]

    def get(self, kind, ident):
        with self.connect() as c:
            r = c.execute("SELECT payload FROM items WHERE kind=? AND id=?", (kind, ident)).fetchone()
            return json.loads(r[0]) if r else None

    def _sample_high_water(self, c):
        row = c.execute("SELECT payload FROM items WHERE kind='meta' AND id='sample-numbering'").fetchone()
        high = int(json.loads(row[0])["high_water"]) if row else 0
        compacted = bool(row and json.loads(row[0]).get("compacted"))
        ids = c.execute("SELECT id FROM items WHERE kind='sample'" if compacted else "SELECT id FROM items WHERE kind='sample' UNION SELECT id FROM audit WHERE kind='sample'")
        for (ident,) in ids:
            match = re.fullmatch(r"SAC_([0-9]+)", ident)
            if match:
                high = max(high, int(match.group(1)))
        return high

    def _update_sample_numbering(self, c):
        from .numbering import numbering_meta
        value = {**numbering_meta(c), "id": "sample-numbering", "origin": self.origin, "high_water": self._sample_high_water(c)}
        c.execute("INSERT OR REPLACE INTO items VALUES('meta','sample-numbering',?)", (json.dumps(value),))
        self._update_batch_numbering(c)

    def _batch_high_water(self, c):
        """Use live records so deleted trailing batch numbers can be reused."""
        high = 0
        for (encoded,) in c.execute("SELECT payload FROM items WHERE kind='sample'"):
            batch = str(json.loads(encoded).get('batch_id') or '')
            match = re.fullmatch(r'BATCH_([0-9]+)', batch)
            if match:
                high = max(high, int(match.group(1)))
        return high

    def _update_batch_numbering(self, c):
        value = {'id': 'batch-numbering', 'origin': self.origin,
            'high_water': self._batch_high_water(c), 'policy': 'current-record-maximum-v2'}
        c.execute("INSERT OR REPLACE INTO items VALUES('meta','batch-numbering',?)", (json.dumps(value),))

    def next_batch_id(self):
        """Preview one above the largest live numeric Batch ID, or one if empty."""
        with self.connect() as c:
            return f'BATCH_{self._batch_high_water(c) + 1:04d}'

    def next_sample_ids(self, count=1):
        """Preview consecutive IDs without consuming numbers before a successful save."""
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Sample count must be a positive integer.")
        with self.connect() as c:
            start = self._sample_high_water(c) + 1
        return [f"SAC_{number:04d}" for number in range(start, start + count)]

    def delete_samples(self, ids, realign=True):
        """Delete records and optionally compact surviving SAC IDs atomically."""
        ids = list(dict.fromkeys(ids))
        if not ids:
            raise ValueError("Select at least one sample to delete.")
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            self._update_sample_numbering(c)
            for ident in ids:
                old = c.execute("SELECT payload FROM items WHERE kind='sample' AND id=?", (ident,)).fetchone()
                if not old:
                    raise ValueError(f"Sample {ident} no longer exists. Refresh the record list.")
                c.execute("DELETE FROM audit WHERE kind='sample' AND id=?", (ident,))
                c.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'delete','sample',?,NULL,NULL)", (now(), ident))
                c.execute("DELETE FROM items WHERE kind='sample' AND id=?", (ident,))
            context = c.execute("SELECT payload FROM items WHERE kind='meta' AND id='plan_context'").fetchone()
            if not c.execute("SELECT 1 FROM items WHERE kind='sample' LIMIT 1").fetchone() or (context and json.loads(context[0])["context"].get("sample_id") in ids):
                c.execute("DELETE FROM items WHERE kind='meta' AND id IN ('plan_context','space')")
            # A deleted report must not continue to appear as a completed queue task.
            for ident, encoded in c.execute("SELECT id,payload FROM items WHERE kind='recommendation'").fetchall():
                obj = json.loads(encoded)
                if ident in ids or obj.get("reported_sample_id") in ids or obj.get("pending_sample_id") in ids:
                    obj.update(state="CANCELLED", deletion_note="Linked sample was deleted")
                    if realign:
                        for field in ("pending_sample_id", "reported_sample_id"):
                            if obj.get(field) in ids:
                                obj["deleted_" + field] = obj.pop(field)
                    updated = json.dumps(obj, allow_nan=False)
                    c.execute("UPDATE items SET payload=? WHERE kind='recommendation' AND id=?", (updated, ident))
                    c.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'revise','recommendation',?,?,?)", (now(), ident, encoded, updated))
            if realign:
                from .numbering import compact_ids
                compact_ids(c, self.origin, ids)
            self._update_batch_numbering(c)
        return len(ids)

    def realign_sample_ids(self):
        """Realign existing SAC gaps without deleting samples or changing measurements."""
        from .numbering import compact_ids
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            return compact_ids(c, self.origin)

    def clear_namespace(self):
        """Clear this namespace permanently, without backup; start numbering at one."""
        folders = [(self.root / name / self.origin.lower()).resolve() for name in ("raw", "backups")]
        if any(not path.is_relative_to(self.root.resolve()) for path in folders):
            raise ValueError("Namespace data paths must remain inside the data directory.")
        with self.connect() as c:
            c.execute("PRAGMA secure_delete=ON")
            c.execute("BEGIN IMMEDIATE")
            c.execute("DELETE FROM items")
            c.execute("DELETE FROM audit")
            c.execute("DELETE FROM messages")
            for obj in ({"id": "seed-v1", "origin": self.origin},
                        {"id": "sample-numbering", "origin": self.origin, "high_water": 0},
                        {"id": "data-epoch", "origin": self.origin, "value": uuid4().hex}):
                c.execute("INSERT INTO items VALUES('meta',?,?)", (obj["id"], json.dumps(obj)))
        for path in folders:
            if path.exists():
                shutil.rmtree(path)
        with self.connect() as c:
            c.execute("VACUUM")

    def put_many(self, kind, objects, id_field, overwrite=False):
        from .units import normalize_method, normalize_payload
        objects = [normalize_method(o) if kind == "method" else normalize_payload(o) for o in objects]
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            for obj in objects:
                if obj.get("origin") != self.origin:
                    raise ValueError("REAL and DEMO data cannot be imported into each other.")
                ident = str(obj[id_field])
                old = c.execute("SELECT payload FROM items WHERE kind=? AND id=?", (kind, ident)).fetchone()
                if old and not overwrite:
                    raise ValueError(f"{ident} already exists; select revision mode to preserve history.")
                if kind == "sample":
                    from .numbering import sample_identity
                    obj = sample_identity(c, obj, json.loads(old[0]) if old else None)
                encoded = json.dumps(obj, ensure_ascii=False, allow_nan=False)
                c.execute("INSERT OR REPLACE INTO items VALUES(?,?,?)", (kind, ident, encoded))
                c.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,?,?,?,?,?)",
                          (now(), "revise" if old else "create", kind, ident, old[0] if old else None, encoded))
            if kind == "sample":
                self._update_sample_numbering(c)

    def source(self, name, content):
        digest = hashlib.sha256(content).hexdigest()
        folder = self.root / "raw" / self.origin.lower()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{digest}_{Path(name).name}"
        if not path.exists():
            path.write_bytes(content)
        return {"file": str(path), "sha256": digest, "original_name": Path(name).name}

    def enqueue_pending(self, rows, context, space):
        """Atomically reserve proposals and allocate unmeasured SAC sample records."""
        import copy
        from .domain import validate_sample, recipe_key, cohort_key, RULE_VERSION
        from .planning import fingerprint, validate_space, concentration_bounds
        if not rows or len(rows) > 12 or len({r["recommendation_id"] for r in rows}) != len(rows):
            raise ValueError("Provide one nonempty preview with unique proposal IDs (maximum 12).")
        _, _, pulses, max_total = validate_space(space, self.origin)
        bounds = concentration_bounds(space, context["elements"])
        if any(r.get("origin") != self.origin or r["recipe"].get("origin") != self.origin for r in rows):
            raise ValueError("REAL and DEMO proposals cannot be mixed.")
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            def all_items(kind):
                return [json.loads(encoded) for (encoded,) in c.execute("SELECT payload FROM items WHERE kind=? ORDER BY id", (kind,))]
            existing = {r["recommendation_id"]: r for r in all_items("recommendation")}
            sample_ids = {s["sample_id"] for s in all_items("sample")}
            linked = [existing.get(r["recommendation_id"], {}).get("pending_sample_id") for r in rows]
            if all(linked) and all(ident in sample_ids for ident in linked) and all(existing[row["recommendation_id"]]["state"] != "CANCELLED" and not existing[row["recommendation_id"]].get("deletion_note") for row in rows):
                return linked
            if any(linked):
                raise ValueError("This preview was partly imported or its linked samples were deleted. Generate a new preview.")
            meta = {m['id']: m for m in all_items('meta')}
            stored_context = meta.get('plan_context', {}).get('context', context)
            stored_space = meta.get('space', {}).get('space', space)
            from .precursor_catalog import resolve_planning_precursors
            from .operating_space import restore_planning_space
            resolved_context = {**stored_context, 'precursors': resolve_planning_precursors(
                meta.get('reagent_library', {}).get('precursors', {}),
                stored_context, stored_context['elements'])}
            current_samples = all_items('sample')
            current_methods = {m['method_id']: m for m in all_items('method')}
            campaign = meta.get('element_campaign', {})
            resolved_space = restore_planning_space(stored_space, context, current_samples,
                current_methods, self.origin, campaign)
            has_catalog_defaults = any(entry.get('source', '').startswith('PLANNED_CATALOG_ASSUMPTION:')
                for entry in context.get('precursors', {}).values())
            context_matches = (fingerprint(resolved_context) == fingerprint(context) or
                not has_catalog_defaults and fingerprint(stored_context) == fingerprint(context))
            # Legacy callers may supply the raw saved range. UI callers supply
            # restored defaults; both retain strict settings/snapshot checks.
            raw_space_matches = (not campaign.get('enabled') and
                '_single_element_campaign' not in space and fingerprint(stored_space) == fingerprint(space))
            if not context_matches or not (raw_space_matches or fingerprint(resolved_space) == fingerprint(space)):
                raise ValueError("Planning settings changed. Generate a new preview.")
            snapshot = fingerprint({"samples": current_samples, "methods": current_methods,
                                    "context": context, "space": space, "rule_version": RULE_VERSION})
            open_keys = {recipe_key(r["recipe"]) for r in existing.values() if r["state"] == "OPEN"}
            proposed = set()
            for row in rows:
                r = row["recipe"]
                if row.get("origin") != self.origin or r.get("origin") != self.origin:
                    raise ValueError("REAL and DEMO proposals cannot be mixed.")
                if row.get("snapshot") != snapshot:
                    raise ValueError("Data changed after this preview. Generate a new preview.")
                if row["recommendation_id"] in existing and existing[row["recommendation_id"]]["state"] != "OPEN":
                    raise ValueError("This proposal is no longer open.")
                key = recipe_key(r)
                same_reservation = existing.get(row["recommendation_id"], {})
                already_reserved = key in open_keys and (not same_reservation or recipe_key(same_reservation["recipe"]) != key)
                if key in proposed or already_reserved:
                    raise ValueError("An identical recipe is already pending or reserved.")
                if same_reservation and recipe_key(same_reservation["recipe"]) != key:
                    raise ValueError("A reserved proposal was changed. Generate a new preview.")
                if cohort_key(r) != cohort_key(context):
                    raise ValueError("A proposal does not match the current composition and process context.")
                proposed.add(key)
                if set(r["elements"]) != set(bounds) or any(not bounds[e]["min_mM"] <= r["concentrations_mM"][e] <= bounds[e]["max_mM"] for e in bounds) or r["pulse_count"] not in pulses or sum(r["concentrations_mM"].values()) > max_total + 1e-9:
                    raise ValueError("A proposal falls outside the registered operating range.")
                if context.get("pulse_limit_C") is not None and (r.get("pulse_limit_C") is None or not space["pulse_limit_min_C"] <= r["pulse_limit_C"] <= space["pulse_limit_max_C"]):
                    raise ValueError("A proposal falls outside the controlled cutoff range.")
            start = self._sample_high_water(c) + 1
            ids = [f"SAC_{number:04d}" for number in range(start, start + len(rows))]
            objects = []
            for ident, row in zip(ids, rows):
                r = row["recipe"]
                pending = {k: copy.deepcopy(r[k]) for k in ("origin", "elements", "concentrations_mM", "pulse_count", "precursor_family", "support_batch", "protocol_id")}
                from .preparation import copy_preparation
                copy_preparation(r, pending)
                if 'precursors' in r:
                    pending['precursors'] = copy.deepcopy(r['precursors'])
                pending.update(sample_id=ident, batch_id="UNKNOWN", temperature_C=None, voltage_V=None, current_A=None,
                               pulse_limit_C=r.get("pulse_limit_C"), planned_temperature_C=r.get("temperature_C"), recommendation_id=row["recommendation_id"],
                               icp=[], icp_review={"state": "PENDING", "source": "", "reviewer": ""},
                               xrd={"state": "PENDING", "mode": "MANUAL", "source": "", "criteria": "", "reviewer": ""},
                               stem={"state": "NOT_TESTED", "source": "", "reviewer": "", "notes": ""},
                               notes="Planned experiment; actual synthesis batch and results have not been recorded.",
                               proposal={k: copy.deepcopy(row.get(k)) for k in ("recipe", "model_id", "model_info", "p_screen_pass", "purpose", "source_sample_ids", "source_elements", "source_affinity", "prediction_mode", "created", "bridge_sources", "support_level", "source_compositions", "expansion_step", "unobserved_pair_count", "unseen_elements", "evaluation_capture", "policy",
                                   'p_screen_low', 'p_screen_high', 'bald_bits', 'joint_information_bits', 'incremental_information_bits', 'model_disagreement', 'calibration_state', 'acquisition', 'interval_note', 'is_repeat', 'repeat_rationale', 'repeat_strategy', 'robustness', 'icp_prediction')})
                pending = validate_sample(pending)
                from .numbering import sample_identity
                pending = sample_identity(c, pending)
                queue = copy.deepcopy(row)
                queue.update(state="OPEN", pending_sample_id=ident)
                if pending.get("record_uid"):
                    queue["pending_record_uid"] = pending["record_uid"]
                objects.extend([("sample", ident, pending), ("recommendation", row["recommendation_id"], queue)])
            for kind, ident, obj in objects:
                old = c.execute("SELECT payload FROM items WHERE kind=? AND id=?", (kind, ident)).fetchone()
                encoded = json.dumps(obj, ensure_ascii=False, allow_nan=False)
                c.execute("INSERT OR REPLACE INTO items VALUES(?,?,?)", (kind, ident, encoded))
                c.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,?,?,?,?,?)", (now(), "revise" if old else "create", kind, ident, old[0] if old else None, encoded))
            self._update_sample_numbering(c)
        return ids

    def import_bundle(self, samples, methods, overwrite=False, queue_updates=()):
        """Atomic import: a bad sample must not leave a partly-updated method."""
        from .units import normalize_method, normalize_payload
        samples = [normalize_payload(s) for s in samples]
        methods = [normalize_method(m) for m in methods]
        queue_updates = [normalize_payload(r) for r in queue_updates]
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            for kind, objects, id_field, replace in (("method", methods, "method_id", True), ("sample", samples, "sample_id", overwrite), ("recommendation", queue_updates, "recommendation_id", True)):
                for obj in objects:
                    if obj.get("origin") != self.origin:
                        raise ValueError("REAL and DEMO data cannot be imported into each other.")
                    ident = str(obj[id_field])
                    old = c.execute("SELECT payload FROM items WHERE kind=? AND id=?", (kind, ident)).fetchone()
                    if old and not replace:
                        raise ValueError(f"{ident} already exists.")
                    if kind == "sample":
                        from .numbering import sample_identity
                        obj = sample_identity(c, obj, json.loads(old[0]) if old else None)
                    encoded = json.dumps(obj, ensure_ascii=False, allow_nan=False)
                    c.execute("INSERT OR REPLACE INTO items VALUES(?,?,?)", (kind, ident, encoded))
                    c.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,?,?,?,?,?)",
                              (now(), "revise" if old else "create", kind, ident, old[0] if old else None, encoded))
            self._update_sample_numbering(c)

    def message(self, role, content, payload=None):
        with self.connect() as c:
            c.execute("INSERT INTO messages(created,role,content,payload) VALUES(?,?,?,?)",
                      (now(), role, content, json.dumps(payload or {}, ensure_ascii=False, allow_nan=False)))

    def messages(self, limit=12):
        with self.connect() as c:
            rows = c.execute("SELECT created,role,content,payload FROM messages ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        return [{"created": r[0], "role": r[1], "content": r[2], "payload": json.loads(r[3])} for r in reversed(rows)]

    def clear_messages(self):
        """Delete only conversation rows in this namespace; keep experiment provenance."""
        with self.connect() as c:
            count = c.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
            c.execute('DELETE FROM messages')
        return count

    def export(self):
        with self.connect() as c:
            audits = [dict(zip(["seq", "created", "action", "kind", "id", "before_json", "after_json"], r)) for r in c.execute("SELECT * FROM audit")]
        return {"origin": self.origin, "samples": self.all("sample"), "methods": self.all("method"), "recommendations": self.all("recommendation"), "chat": self.messages(10000), "audit": audits}
