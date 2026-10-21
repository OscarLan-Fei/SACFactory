"""Transactional display-ID compaction with stable physical-record identities."""
import copy
import json
import re
from uuid import uuid4


def numbering_meta(connection):
    row = connection.execute("SELECT payload FROM items WHERE kind='meta' AND id='sample-numbering'").fetchone()
    return json.loads(row[0]) if row else {}


def sample_identity(connection, sample, old=None):
    result = copy.deepcopy(sample)
    old = old or {}
    uid = old.get("record_uid")
    if uid and result.get("record_uid") not in {None, "", uid}:
        raise ValueError("The record UID belongs to a different physical sample. Re-export current data after renumbering.")
    if uid or numbering_meta(connection).get("compacted"):
        result["record_uid"] = uid or result.get("record_uid") or uuid4().hex
    if result.get("record_uid"):
        for ident, encoded in connection.execute("SELECT id,payload FROM items WHERE kind='sample'"):
            if ident != result["sample_id"] and json.loads(encoded).get("record_uid") == result["record_uid"]:
                raise ValueError("This physical-record UID already belongs to another current Sample ID.")
    return result


def rewrite_references(payload, mapping, deleted=()):
    """Relabel structured references; never rewrite prose, raw files or audit payloads."""
    result = copy.deepcopy(payload)
    if isinstance(result, list):
        return [rewrite_references(v, mapping, deleted) for v in result]
    if not isinstance(result, dict):
        return result
    for key, value in list(result.items()):
        if key == "original_proposal":
            continue
        if key in {"sample_id", "pending_sample_id", "reported_sample_id"} and isinstance(value, str):
            if value in deleted:
                result["deleted_" + key] = value
                result[key] = None
            elif value in mapping:
                result[key] = mapping[value]
        elif key in {"source_sample_ids", "successful_source_ids"} and isinstance(value, list):
            retired = [ident for ident in value if ident in deleted]
            if retired:
                result["retired_" + key] = retired
            result[key] = [mapping.get(ident, ident) for ident in value if ident not in deleted]
        elif isinstance(value, (dict, list)):
            result[key] = rewrite_references(value, mapping, deleted)
    return result


def compact_ids(connection, origin, deleted=()):
    """Called only inside the caller's SQLite write transaction."""
    from .storage import now
    records = [(ident, json.loads(encoded), encoded) for ident, encoded in connection.execute("SELECT id,payload FROM items WHERE kind='sample' ORDER BY id")]
    numeric = sorted([ident for ident, _, _ in records if re.fullmatch(r"SAC_[0-9]+", ident)], key=lambda ident: (int(ident[4:]), ident))
    mapping = {old: f"SAC_{i:04d}" for i, old in enumerate(numeric, 1)}
    identities = {ident: obj.get("record_uid") or uuid4().hex for ident, obj, _ in records}
    # Deleted IDs may now identify a different record. Keep tombstones outside live sample history.
    connection.execute("UPDATE audit SET kind='deleted_sample' WHERE kind='sample' AND action='delete'")
    for seq, ident in connection.execute("SELECT seq,id FROM audit WHERE kind='sample'").fetchall():
        if ident in mapping:
            connection.execute("UPDATE audit SET id=? WHERE seq=?", (mapping[ident], seq))
    connection.execute("DELETE FROM items WHERE kind='sample'")
    for ident, obj, before in records:
        updated = rewrite_references(obj, mapping, deleted)
        updated["sample_id"] = mapping.get(ident, ident)
        updated["record_uid"] = identities[ident]
        if updated["sample_id"] != ident:
            updated.setdefault("previous_sample_ids", []).append(ident)
        encoded = json.dumps(updated, ensure_ascii=False, allow_nan=False)
        connection.execute("INSERT INTO items VALUES('sample',?,?)", (updated["sample_id"], encoded))
        if updated != obj:
            connection.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'realign_id','sample',?,?,?)", (now(), updated["sample_id"], before, encoded))
    for kind, ident, encoded in connection.execute("SELECT kind,id,payload FROM items WHERE kind!='sample'").fetchall():
        obj = json.loads(encoded)
        updated = rewrite_references(obj, mapping, deleted)
        if kind == "recommendation":
            if obj.get("state") == "CANCELLED" and obj.get("deletion_note"):
                for field in ("pending_sample_id", "reported_sample_id"):
                    if obj.get(field) not in identities and obj.get(field):
                        updated["deleted_" + field] = obj[field]
                        updated.pop(field, None)
            for field in ("pending_sample_id", "reported_sample_id"):
                old_link = obj.get(field)
                if old_link in identities:
                    updated[field.replace("sample_id", "record_uid")] = identities[old_link]
            # Legacy report tasks may use a Sample ID as their task ID; task identities stay fixed.
            linked = obj.get("pending_sample_id") or obj.get("reported_sample_id") or ident
            if linked in identities and not obj.get("pending_sample_id") and not obj.get("reported_sample_id"):
                current_id = mapping.get(linked, linked)
                row = connection.execute("SELECT payload FROM items WHERE kind='sample' AND id=?", (current_id,)).fetchone()
                if row:
                    sample = json.loads(row[0])
                    sample["recommendation_id"] = ident
                    connection.execute("UPDATE items SET payload=? WHERE kind='sample' AND id=?", (json.dumps(sample, allow_nan=False), current_id))
        if updated != obj:
            changed = json.dumps(updated, ensure_ascii=False, allow_nan=False)
            connection.execute("UPDATE items SET payload=? WHERE kind=? AND id=?", (changed, kind, ident))
            connection.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'realign_references',?,?,?,?)", (now(), kind, ident, encoded, changed))
    for seq, encoded in connection.execute("SELECT seq,payload FROM messages").fetchall():
        payload = rewrite_references(json.loads(encoded), mapping, deleted)
        payload["numbering_note"] = "Message prose keeps the Sample IDs used when written; structured references use current IDs."
        connection.execute("UPDATE messages SET payload=? WHERE seq=?", (json.dumps(payload, allow_nan=False), seq))
    prior = numbering_meta(connection)
    meta = {"id": "sample-numbering", "origin": origin, "high_water": len(numeric), "compacted": True, "generation": prior.get("generation", 0) + 1}
    connection.execute("INSERT OR REPLACE INTO items VALUES('meta','sample-numbering',?)", (json.dumps(meta),))
    epoch = {"id": "data-epoch", "origin": origin, "value": uuid4().hex}
    connection.execute("INSERT OR REPLACE INTO items VALUES('meta','data-epoch',?)", (json.dumps(epoch),))
    connection.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'realign_sample_ids','numbering',?,NULL,?)", (now(), epoch["value"], json.dumps({"mapping": mapping, "record_uids": identities, "generation": meta["generation"]})))
    return mapping
