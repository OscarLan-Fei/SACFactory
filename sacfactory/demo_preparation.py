"""Restore declared simulator inputs without imputing experimental metadata."""
import copy
import json
import math
from uuid import uuid4

VERSION = 'demo-preparation-inputs-v2'
FIELDS = ('pan_mass_g', 'solvent_volume_mL', 'carbonization_loss_fraction')


def restore_inputs(sample):
    """Copy missing inputs from the v6 simulator ledger, never from outcomes."""
    if sample.get('origin') != 'DEMO' or not str(sample.get('record_uid', '')).startswith('demo-v6-'):
        return sample
    ledger = sample.get('demo_mass_balance')
    if not isinstance(ledger, dict):
        return sample
    restored = copy.deepcopy(sample)
    changed = []
    for field in FIELDS:
        if restored.get(field) not in (None, ''):
            continue
        value = ledger.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        if not (0 <= value < 1 if field == 'carbonization_loss_fraction' else value > 0):
            continue
        restored[field] = value
        changed.append(field)
    forecast_fields = []
    proposal = restored.get('proposal', {})
    recipe = proposal.get('recipe', {})
    if proposal.get('policy') == 'SIMULATED_SEQUENTIAL_STAGE_RF' and recipe.get('origin') == 'DEMO':
        for field in FIELDS:
            value = ledger.get(field)
            if recipe.get(field) not in (None, '') or isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            if 0 <= value < 1 if field == 'carbonization_loss_fraction' else value > 0:
                recipe[field] = value
                forecast_fields.append(field)
    if not changed and not forecast_fields:
        return sample
    restored['demo_preparation_provenance'] = {
        'version': VERSION, 'source': 'demo_mass_balance simulator inputs',
        'restored_fields': changed, 'restored_forecast_input_fields': forecast_fields,
        'status': 'SIMULATED_INPUTS_NOT_MEASUREMENTS',
        'forecast_status': 'Saved probabilities unchanged; only original simulator inputs restored'}
    from .domain import validate_sample
    return validate_sample(restored)


def restore_content(content):
    """Use the same input contract for JSON and compressed demo fixtures."""
    content['samples'] = [restore_inputs(sample) for sample in content['samples']]
    if content.get('planning_context'):
        content['planning_context'] = restore_inputs(content['planning_context'])
    return content


def migrate_store(store):
    """Idempotently repair existing bundled DEMO records; never touch REAL."""
    if store.origin != 'DEMO' or store.get('meta', VERSION):
        return 0
    from .storage import now
    changed = 0
    with store.connect() as connection:
        connection.execute('BEGIN IMMEDIATE')
        # Check again under the write lock for concurrent browser sessions.
        if connection.execute("SELECT 1 FROM items WHERE kind='meta' AND id=?", (VERSION,)).fetchone():
            return 0
        rows = connection.execute("SELECT id,payload FROM items WHERE kind='sample'").fetchall()
        for ident, encoded in rows:
            original = json.loads(encoded)
            restored = restore_inputs(original)
            if restored == original:
                continue
            updated = json.dumps(restored, allow_nan=False)
            connection.execute("UPDATE items SET payload=? WHERE kind='sample' AND id=?", (updated, ident))
            connection.execute(
                "INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'restore_demo_preparation','sample',?,?,?)",
                (now(), ident, encoded, updated))
            changed += 1
        row = connection.execute("SELECT payload FROM items WHERE kind='meta' AND id='plan_context'").fetchone()
        if row:
            meta = json.loads(row[0])
            before = meta.get('context', {})
            after = restore_inputs(before)
            if after != before:
                meta['context'] = after
                connection.execute("UPDATE items SET payload=? WHERE kind='meta' AND id='plan_context'",
                                   (json.dumps(meta, allow_nan=False),))
        marker = {'id': VERSION, 'origin': 'DEMO', 'restored_samples': changed}
        connection.execute("INSERT INTO items VALUES('meta',?,?)", (VERSION, json.dumps(marker)))
        if changed:
            epoch = {'id': 'data-epoch', 'origin': 'DEMO', 'value': uuid4().hex}
            connection.execute("INSERT OR REPLACE INTO items VALUES('meta','data-epoch',?)", (json.dumps(epoch),))
    return changed
