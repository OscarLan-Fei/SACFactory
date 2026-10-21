"""Explicit, one-time correction of user-declared historical mmol entry errors."""
import copy
import json

from .domain import validate_sample
from .preparation import to_mM
from .storage import now


def reinterpret_recorded_values_as_mmol(store, expected, correction_id):
    """Correct exactly the reviewed REAL snapshot, atomically and without backups.

    Never call automatically at startup. The caller must have the user's explicit
    instruction that the historical numeric mM entries actually denote mmol.
    """
    if store.origin != 'REAL' or not expected or not correction_id:
        raise ValueError('An explicit REAL sample snapshot and correction ID are required.')
    from .planning import fingerprint
    marker = 'quantity-correction-' + correction_id
    changed = []
    with store.connect() as connection:
        connection.execute('BEGIN IMMEDIATE')
        if connection.execute("SELECT 1 FROM items WHERE kind='meta' AND id=?", (marker,)).fetchone():
            return {'corrected_samples':0, 'already_applied':True}
        timestamp = now()
        for ident, signature in expected.items():
            found = connection.execute("SELECT payload FROM items WHERE kind='sample' AND id=?", (ident,)).fetchone()
            if not found:
                raise ValueError('A reviewed sample no longer exists: ' + ident)
            original = json.loads(found[0])
            if fingerprint(original) != signature:
                raise ValueError('A sample changed after review: ' + ident)
            revised = copy.deepcopy(original)
            volume = revised.get('solvent_volume_mL')
            revised['concentrations_mM'] = {e:to_mM(v, 'mmol', volume) for e,v in original['concentrations_mM'].items()}
            revised['concentration_input_unit'] = 'mmol'
            revised['quantity_correction'] = {'id':correction_id, 'created':timestamp,
                'basis':'User explicitly confirmed historical numeric mM entries were mmol.',
                'original_numeric_entries_mmol':copy.deepcopy(original['concentrations_mM']),
                'dmf_volume_mL':volume, 'formula':'metal_mM = 1000 * metal_mmol / DMF_mL'}
            for field in ('proposal', 'manual_prediction'):
                if revised.get(field):
                    revised[field]['prediction_stale'] = True
            revised = validate_sample(revised)
            encoded = json.dumps(revised, allow_nan=False)
            connection.execute("UPDATE items SET payload=? WHERE kind='sample' AND id=?", (encoded,ident))
            connection.execute('INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,?,?,?,?,?)',
                (timestamp,'correct_user_declared_mmol_entry','sample',ident,found[0],encoded))
            changed.append(ident)
        for ident, encoded in connection.execute("SELECT id,payload FROM items WHERE kind='recommendation'").fetchall():
            row = json.loads(encoded)
            row.update(prediction_stale=True, p_screen_pass=None,
                invalidation_reason='Training quantities corrected from mislabeled mmol entries.')
            revised = json.dumps(row, allow_nan=False)
            connection.execute("UPDATE items SET payload=? WHERE kind='recommendation' AND id=?", (revised,ident))
            connection.execute('INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,?,?,?,?,?)',
                (timestamp,'invalidate_after_quantity_correction','recommendation',ident,encoded,revised))
        info = {'id':marker, 'origin':'REAL', 'created':timestamp, 'sample_ids':changed,
            'corrected_samples':len(changed), 'already_applied':False}
        connection.execute("INSERT INTO items VALUES('meta',?,?)", (marker,json.dumps(info)))
    return info
