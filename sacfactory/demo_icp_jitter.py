"""Reproducible extra measurement perturbations for synthetic ICP readings."""
import copy
import hashlib
import json
import math
from uuid import uuid4

import numpy as np

VERSION = 'demo-icp-relative-jitter-7pct-v1'
SEED = 20261021
MAX_RELATIVE_CHANGE = .07


def perturb_sample(sample):
    """Perturb quantified DEMO observations once; leave latent masses untouched."""
    if (sample.get('origin') != 'DEMO'
            or not str(sample.get('record_uid', '')).startswith('demo-v6-')
            or sample.get('demo_icp_jitter', {}).get('version') == VERSION):
        return sample
    quantified = {}
    for assay in sample.get('icp', []):
        if assay.get('flag') != 'MEASURED' or assay.get('unit') != 'wt%':
            continue
        try:
            value = float(assay['value'])
        except (ValueError, TypeError, KeyError):
            continue
        if math.isfinite(value) and 0 < value < 100:
            quantified[assay['element']] = value
    if not quantified:
        return sample
    from ._pulseicp.composition import Composition, inverse_composition
    from ._pulseicp.configuration import physics_settings
    physics = physics_settings()
    composition = None
    if set(quantified) == set(sample.get('elements', [])):
        try:
            composition = Composition(sample['concentrations_mM'], sample['pulse_limit_C'],
                sample['pulse_count'], sample['pan_mass_g'], sample['solvent_volume_mL'],
                sample['carbonization_loss_fraction']).validate()
            inverse_composition(composition, quantified, physics)
        except (ValueError, TypeError, KeyError, OverflowError):
            composition = None
    key = f'{VERSION}:{SEED}:{sample["record_uid"]}'.encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'big'))
    elements = sorted(quantified)
    base = np.asarray([quantified[e] for e in elements])
    details = sample.get('demo_observation', {}).get('assays', {})
    lower = np.asarray([float(details.get(e, {}).get('loq_wt_percent', 0.)) for e in elements])
    for attempt in range(1, 1025):
        # Rejection conditions preserve quantification and previously valid mass bounds.
        candidate = base * (1 + rng.uniform(-MAX_RELATIVE_CHANGE, MAX_RELATIVE_CHANGE, len(base)))
        candidate = np.asarray([float(format(value, '.12g')) for value in candidate])
        if np.any(candidate < lower) or np.any(candidate <= 0) or candidate.sum() >= 100:
            continue
        if composition is not None:
            try:
                inverse_composition(composition, dict(zip(elements, candidate)), physics)
            except ValueError:
                continue
        break
    else:
        raise ValueError(f'{sample["sample_id"]}: unable to draw a valid synthetic ICP perturbation.')
    result = copy.deepcopy(sample)
    changes = {}
    values = dict(zip(elements, candidate))
    for assay in result['icp']:
        element = assay['element']
        if element not in values or assay.get('flag') != 'MEASURED':
            continue
        original = assay['value']
        value = values[element]
        assay['value'] = format(value, '.12g')
        changes[element] = {'base_reported_value': original,
            'perturbed_wt_percent': value, 'relative_change': value / quantified[element] - 1}
        detail = result.get('demo_observation', {}).get('assays', {}).get(element)
        if detail is not None:
            detail['base_uncensored_reading_wt_percent'] = detail['simulated_uncensored_reading_wt_percent']
            detail['simulated_uncensored_reading_wt_percent'] = value
    result['demo_icp_jitter'] = {'version': VERSION, 'seed': SEED,
        'distribution': 'Uniform(-0.07, 0.07), conditioned on quantification and existing physical mass bounds',
        'status': 'INJECTED_SYNTHETIC_PERTURBATION_NOT_A_CONFIDENCE_INTERVAL',
        'max_relative_change': MAX_RELATIVE_CHANGE, 'draw_attempts': attempt,
        'complete_composition_mass_bound_checked': composition is not None,
        'observations': changes}
    return result


def perturb_content(content):
    content['samples'] = [perturb_sample(sample) for sample in content['samples']]
    if content.get('planning_context'):
        content['planning_context'] = perturb_sample(content['planning_context'])
    content['icp_perturbation'] = {'version': VERSION, 'seed': SEED,
        'maximum_relative_change': MAX_RELATIVE_CHANGE,
        'scope': 'Quantified synthetic ICP only; no REAL data or latent masses changed'}
    return content


def migrate_store(store):
    """Update populated DEMO stores once, preserving record identities and labels."""
    if store.origin != 'DEMO' or store.get('meta', VERSION):
        return 0
    from .storage import now
    changed = 0
    with store.connect() as connection:
        connection.execute('BEGIN IMMEDIATE')
        if connection.execute("SELECT 1 FROM items WHERE kind='meta' AND id=?", (VERSION,)).fetchone():
            return 0
        for ident, encoded in connection.execute("SELECT id,payload FROM items WHERE kind='sample'").fetchall():
            original = json.loads(encoded)
            updated = perturb_sample(original)
            if updated == original:
                continue
            payload = json.dumps(updated, allow_nan=False)
            connection.execute("UPDATE items SET payload=? WHERE kind='sample' AND id=?", (payload, ident))
            connection.execute(
                "INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'perturb_synthetic_icp','sample',?,?,?)",
                (now(), ident, encoded, payload))
            changed += 1
        context = connection.execute("SELECT payload FROM items WHERE kind='meta' AND id='plan_context'").fetchone()
        if context:
            meta = json.loads(context[0])
            meta['context'] = perturb_sample(meta.get('context', {}))
            connection.execute("UPDATE items SET payload=? WHERE kind='meta' AND id='plan_context'", (json.dumps(meta),))
        marker = {'id': VERSION, 'origin': 'DEMO', 'changed_samples': changed, 'seed': SEED}
        connection.execute("INSERT INTO items VALUES('meta',?,?)", (VERSION, json.dumps(marker)))
        if changed:
            epoch = {'id': 'data-epoch', 'origin': 'DEMO', 'value': uuid4().hex}
            connection.execute("INSERT OR REPLACE INTO items VALUES('meta','data-epoch',?)", (json.dumps(epoch),))
    return changed
