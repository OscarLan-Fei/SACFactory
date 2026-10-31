"""Inference-only SAC-Gen releases without private training records or pickles."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from .sac_gen import SACGen
from .token_residual import TokenResidualEnsemble, TransformerDenoiser, TransformerGaussian
from ._pulseicp.features import TrainingScaler
from .gb_residual import GBResidual

FORMAT = 'sac-gen-public-inference-v1'


def bundle_root():
    return Path(__file__).resolve().parents[1] / 'pretrained'


def available():
    path = bundle_root() / 'registry.json'
    return json.loads(path.read_text(encoding='utf-8'))['models'] if path.is_file() else {}


def _checked_file(root, name, expected):
    if Path(name).name != name:
        raise ValueError('A public model asset must be an immediate bundle file.')
    path = root / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Public model asset failed its integrity check.')
    return path


class _ReleasedToken(TokenResidualEnsemble):
    @property
    def model_id(self):
        return self.release_id


class _Trees:
    """Evaluate only numeric inference nodes; no estimator pickle is loaded."""
    def __init__(self, specification):
        self.baseline = float(specification['baseline'])
        self.trees = specification['trees']

    def predict(self, x):
        x = np.asarray(x)
        output = np.full(len(x), self.baseline)
        for tree in self.trees:
            value = np.asarray(tree['value'])
            leaf = np.asarray(tree['is_leaf'], dtype=bool)
            feature = np.asarray(tree['feature_idx'], dtype=int)
            threshold = np.asarray(tree['num_threshold'])
            left = np.asarray(tree['left'], dtype=int)
            right = np.asarray(tree['right'], dtype=int)
            missing_left = np.asarray(tree['missing_go_to_left'], dtype=bool)
            node = np.zeros(len(x), dtype=int)
            for _ in range(len(value)):
                active = np.flatnonzero(~leaf[node])
                if not len(active):
                    break
                at = node[active]
                observed = x[active, feature[at]]
                take_left = np.where(np.isnan(observed), missing_left[at], observed <= threshold[at])
                node[active] = np.where(take_left, left[at], right[at])
            else:
                raise ValueError('Invalid cyclic public tree.')
            output += value[node]
        return output


class _ReleasedGB(GBResidual):
    @property
    def model_id(self):
        return self.release_id


class PretrainedSACGen(SACGen):
    """Read-only forward model; it cannot refit or expose a training snapshot."""
    is_public_release = True

    @property
    def model_id(self):
        return self.release_id

    def fit(self, *args, **kwargs):
        raise ValueError('Train a separate local SAC-Gen candidate; public releases are read-only.')

    def save(self, *args, **kwargs):
        raise ValueError('Use the original inference bundle; this release has no training checkpoint.')

    def predict(self, samples):
        rows = []
        for sample in samples:
            row = copy.deepcopy(sample)
            for key, value in self.process_defaults.items():
                if key in row and row[key] != value:
                    raise ValueError('The public model requires its declared precursor/support/protocol context.')
                row.setdefault(key, value)
            row.setdefault('origin', 'REAL')
            row.setdefault('pan_mass_g', 2.)
            row.setdefault('solvent_volume_mL', 20.)
            row.setdefault('carbonization_loss_fraction', .06)
            row.setdefault('precursors', {})
            if row.get('pulse_limit_C') is None:
                raise ValueError('Supply the controlled cutoff temperature for public prediction.')
            from .discovery.common import query
            query(row)
            rows.append(row)
        output = super().predict(rows)
        for item in output:
            item.update(model_name='Pretrained ' + self.variant_name,
                model_release=FORMAT,
                support_status='Exact training recipes are private; composition membership is not released.',
                outside_training_support=True)
        return output


def load(name='SAC-Gen-GB', device='auto'):
    """Load bundled learned parameters. No database, API or training is needed."""
    import torch
    from .compute_runtime import resolve
    from .accelerated_ensemble import model_references, feature_names
    name = name.removeprefix('Pretrained ')
    entry = available().get(name)
    if entry is None:
        raise ValueError('This pretrained SAC-Gen variant is not bundled.')
    path = _checked_file(bundle_root(), entry['file'], entry['sha256'])
    meta = json.loads(path.read_text(encoding='utf-8'))
    if meta['format'] != FORMAT or meta['references'] != model_references(meta['settings']):
        raise ValueError('Public model format or reference snapshots differ from this software.')
    if meta['feature_names'] != feature_names(meta['settings']):
        raise ValueError('Public model feature schema changed.')
    settings = meta['settings']
    obj = PretrainedSACGen(meta['family'], settings, meta['profile'])
    obj.release_id = entry['sha256'][:20]
    obj.process_defaults = meta['process_defaults']
    obj.origin = 'REAL'
    obj.head = meta['head']
    obj.training_ids = []
    obj.training_samples = []
    obj.xrd_info = {
        'probability_status': 'PRETRAINED_FORECAST; reliability on a new campaign requires evaluation',
        'reason': 'Structural features are unavailable for this recipe; inspect inputs and reference resources.',
    }
    if meta['family'] == 'Gradient boosting residual':
        loading = _ReleasedGB(meta['family'], settings, meta['physics'])
        loading.regressor = _Trees(meta['trees'])
    else:
        loading = _ReleasedToken(meta['family'], settings, meta['physics'])
        runtime = resolve({'device': device})
        torch.set_num_threads(runtime['threads'])
        archive = _checked_file(bundle_root(), meta['weights']['file'], meta['weights']['sha256'])
        with np.load(archive, allow_pickle=False) as values:
            for i in range(settings['members']):
                network = (TransformerGaussian if loading.family == 'Gaussian residual' else TransformerDenoiser)(
                    len(meta['feature_names']), settings['token_width'], settings['attention_heads'], settings['attention_layers'])
                prefix = f'member_{i}/'
                state = {key[len(prefix):]: torch.from_numpy(values[key].copy()) for key in values.files if key.startswith(prefix)}
                network.load_state_dict(state, strict=True)
                loading.members.append(network.to(runtime['device']).eval())
        for key in ('y_mean', 'y_scale', 'clean_limit', 'has_mixture_training'):
            setattr(loading, key, meta[key])
    loading.release_id = obj.release_id
    loading.scaler = TrainingScaler.from_dict(meta['scaler'])
    loading.elements = meta['supported_elements']
    loading.compositions = []
    loading.origin = 'EXPERIMENTAL'
    loading.protocols = [obj.process_defaults['protocol_id']]
    obj.loading = loading
    return obj
