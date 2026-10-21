"""Preview and commit normalized batch files through one UI-independent API."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from .domain import STATUS_NAMES, RULE_VERSION, screen, parse_measurement
from .importers import read_table, samples_from_rows, methods_from_rows
from .queueing import prepare_reports


@dataclass
class ImportPreview:
    samples: list
    methods: list
    files: list
    rows: list
    method_changes: list
    queue_updates: list
    overwrite: bool
    defaults: dict | None = None


def method_content(method):
    """File attachments do not alter the analytical method definition."""
    return {k: v for k, v in method.items() if k not in {"import_sources"}}


def preview_import(files, store, overwrite=False, defaults=None):
    parsed = {}
    for name, content in files:
        if name.lower().endswith(".xlsx"):
            import io
            import openpyxl
            book = openpyxl.load_workbook(io.BytesIO(content), read_only=True)
            names = book.sheetnames
            book.close()
            if "batch" in names:
                if parsed: raise ValueError("Upload one compact batch workbook at a time.")
                from .batching import batch_rows_to_standard
                settings = read_table(content, name, "settings")
                if len(settings) != 1: raise ValueError("Compact workbook requires exactly one settings row.")
                parsed["samples"], parsed["icp"] = batch_rows_to_standard(read_table(content, name, "batch"), read_table(content, name, "icp"), settings[0], name)
                parsed["methods"] = read_table(content, name, "methods") if "methods" in names else []
                continue
            for kind in ("samples", "icp", "methods"):
                if kind in parsed:
                    raise ValueError("Upload only one file per table type.")
                parsed[kind] = read_table(content, name, kind)
        else:
            rows = read_table(content, name)
            headers = set(rows[0]) if rows else set()
            kind = "batch" if "concentration_mM" in headers else "settings" if "target_elements" in headers and "sample_id" not in headers else "samples" if "target_elements" in headers else ("icp" if headers & {"value", "w_wt_percent", "W", "W / wt%", "W (wt%)"} else ("methods" if headers & {"loq_wt_percent", "loq_ug_g"} else None))
            if not kind or kind in parsed:
                raise ValueError("Unrecognized or duplicate CSV table type.")
            parsed[kind] = rows
    if "batch" in parsed:
        if "samples" in parsed: raise ValueError("Do not mix compact and advanced sample tables in one import.")
        from .batching import batch_rows_to_standard
        setting_rows = parsed.get("settings", [])
        if len(setting_rows) != 1:
            raise ValueError("Compact CSV import requires one settings CSV row, plus matching ICP detail when using DETAIL concentrations.")
        parsed["samples"], parsed["icp"] = batch_rows_to_standard(parsed["batch"], parsed.get("icp", []), setting_rows[0], "Uploaded compact CSV tables")
    if "icp" in parsed and "samples" not in parsed:
        raise ValueError("ICP reporting requires a matching sample table to verify recipes and batches.")
    samples = samples_from_rows(parsed["samples"], parsed.get("icp", [])) if parsed.get("samples") else []
    from .reagent_links import link_precursors
    library = (store.get('meta', 'reagent_library') or {}).get('precursors', {})
    samples = [link_precursors(s, library) if not store.get('sample', s['sample_id'])
        and not s.get('recommendation_id') and not store.get('recommendation', s['sample_id']) else s for s in samples]
    imported_methods = methods_from_rows(parsed["methods"]) if parsed.get("methods") else []
    if not samples and not imported_methods:
        raise ValueError("No sample or method rows were supplied.")
    if any(s["origin"] != store.origin for s in samples + imported_methods):
        raise ValueError("The uploaded namespace differs from the selected space.")
    old_methods = {m["method_id"]: m for m in store.all("method")}
    changes, methods = [], []
    for m in imported_methods:
        old = old_methods.get(m["method_id"])
        if old and method_content(old) == method_content(m):
            continue
        if old and not overwrite:
            raise ValueError(f"Method {m['method_id']} already exists with different settings. Enable revision mode and review affected samples.")
        changes.append({"method_id": m["method_id"], "action": "revise" if old else "create", "old_version": old.get("version", "") if old else "", "new_version": m["version"], "affected_existing_samples": sum(any(a.get("method_id") == m["method_id"] for a in s.get("icp", [])) for s in store.all("sample"))})
        methods.append(m)
    revised_methods = {**old_methods, **{m["method_id"]: m for m in methods}}
    rows, queue_updates = [], []
    for s in samples:
        old = store.get("sample", s["sample_id"])
        if s.get("record_uid"):
            owner = next((record for record in store.all("sample") if record.get("record_uid") == s["record_uid"] and record["sample_id"] != s["sample_id"]), None)
            if owner:
                raise ValueError(f"This physical sample now has ID {owner['sample_id']}. Re-export current reporting tables.")
        if old and old.get("record_uid") and s.get("record_uid") != old["record_uid"]:
            raise ValueError("Sample IDs were realigned. Re-export current samples/ICP tables with record_uid before revising by file, or use the manual editor.")
        if old and not overwrite:
            raise ValueError(f"Sample {s['sample_id']} already exists. Enable revision mode to report updated results.")
        result = screen(s, revised_methods)
        rows.append({"sample_id": s["sample_id"], "targets": len(s["elements"]), "action": "revise" if old else "create", "status": STATUS_NAMES[result["status"]]})
    samples, queue_updates = prepare_reports(samples, store, revised_methods)
    return ImportPreview(samples, methods, files, rows, changes, queue_updates, overwrite, defaults)


def commit_import(preview, store):
    """Revalidate against the current DB, then commit samples/methods/queue atomically."""
    checked = preview_import(preview.files, store, preview.overwrite, preview.defaults)
    sources = [store.source(name, data) for name, data in checked.files]
    for kind, objects, field in (("sample", checked.samples, "sample_id"), ("method", checked.methods, "method_id")):
        for obj in objects:
            old = store.get(kind, obj[field]) or {}
            prior = old.get("import_sources", [])
            obj["import_sources"] = list({m["sha256"]: m for m in prior + sources}.values())
    store.import_bundle(checked.samples, checked.methods, checked.overwrite, checked.queue_updates)
    return {"samples": len(checked.samples), "methods": len(checked.methods), "completed_tasks": len(checked.queue_updates)}
