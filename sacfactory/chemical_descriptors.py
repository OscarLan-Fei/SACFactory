"""Offline attributed elemental properties with explicit missingness and units."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import numpy as np

from .composition import descriptors
from .domain import ELEMENTS


PROPERTIES = ['atomic_weight', 'en_pauling', 'covalent_radius_cordero', 'electron_affinity', 'first_ionization_energy']
SCALES = np.array([250., 4., 250., 4., 25.])


@lru_cache(maxsize=1)
def reference():
    path = Path(__file__).resolve().parents[1]/'resources/element_properties.json'
    content = path.read_bytes()
    data = json.loads(content)
    data['snapshot_sha256'] = hashlib.sha256(content).hexdigest()
    return data


def descriptor_matrix(profile='periodic'):
    periodic = np.array([descriptors(e) for e in ELEMENTS])
    if profile == 'periodic':
        return periodic
    from .precursors import base_profile
    thermal = profile in {'chemistry_thermal', 'chemistry_loss', 'chemistry_mixing', 'chemistry_pulseicp'}
    profile = base_profile(profile)
    if profile not in {'chemistry', 'chemistry_ion', 'chemistry_precursor', 'chemistry_state'}:
        raise ValueError('Select a supported chemical descriptor profile.')
    data = {row['symbol']: row for row in reference()['elements']}
    values = np.array([[np.nan if data[e][p] is None else data[e][p] for p in PROPERTIES] for e in ELEMENTS], dtype=float)
    missing = np.isnan(values)
    # Zero is an encoding placeholder accompanied by a missing mask, not a filled measurement.
    result = np.column_stack([periodic, np.where(missing, 0., values)/SCALES, missing.astype(float)])
    if profile in {'chemistry', 'chemistry_ion', 'chemistry_precursor', 'chemistry_state'}:
        from .precursors import state_reference
        atoms = {row['symbol']: row for row in state_reference()['elements']}
        radii = np.array([np.nan if atoms[e]['atomic_radius_rahm'] is None else atoms[e]['atomic_radius_rahm'] for e in ELEMENTS])
        missing_radius = np.isnan(radii)
        result = np.column_stack([result, np.where(missing_radius, 0., radii)/300., missing_radius.astype(float)])
    if thermal:
        from .physics_features import thermal_matrix
        result = np.column_stack([result, thermal_matrix()])
    return result


def coverage(elements):
    data = {row['symbol']: row for row in reference()['elements']}
    from .precursors import state_reference
    atoms = {row['symbol']: row for row in state_reference()['elements']}
    from .physics_features import THERMAL_FIELDS
    from .thermal_loss import elemental_loss_reference
    return [{'element': e, **{key: data[e][key] for key in PROPERTIES}, 'atomic_radius_rahm': atoms[e]['atomic_radius_rahm'],
             **{key: elemental_loss_reference(e).get(key) for key in THERMAL_FIELDS}} for e in elements]


def model_features(x, profile='chemistry'):
    """Full recipe representation plus concentration-weighted descriptor moments."""
    x = np.atleast_2d(x)
    n = len(ELEMENTS)
    mask = x[:, :n]
    count = mask.sum(axis=1)
    c = mask*np.exp(np.clip(x[:, n:2*n]*np.sqrt(count[:, None]), -30, 30))
    weights = c/c.sum(axis=1, keepdims=True)
    positions = descriptor_matrix(profile)
    mean = weights @ positions
    variance = np.maximum(weights @ (positions*positions)-mean*mean, 0)
    return np.column_stack([x, mean, np.sqrt(variance)])
