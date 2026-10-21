"""Independent Gaussian and diffusion ensembles with safe checkpoint loading."""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import torch

from . import __version__
from .references import digest, provenance, support_columns
from .features import TrainingScaler, FEATURE_NAMES, features
from .networks import GaussianResidual, ResidualDenoiser
from .diffusion import training_loss, sample
from .mass_decoder import prior
from .composition import as_composition, component_features, decode_composition, INTERACTION_NAMES
ALL_FEATURE_NAMES = FEATURE_NAMES + INTERACTION_NAMES
from .uncertainty import summarize
from .schema import control_domain_columns

FAMILIES = ['Gaussian residual', 'Conditional diffusion']


def feature_names(settings):
    if settings.get('thermodynamic_features', False):
        from .thermodynamics import FEATURE_NAMES as THERMODYNAMIC_NAMES
        return ALL_FEATURE_NAMES + THERMODYNAMIC_NAMES
    return ALL_FEATURE_NAMES


def model_references(settings):
    result = {k: v['snapshot_sha256'] for k, v in provenance().items()}
    if settings.get('thermodynamic_features', False):
        from .thermodynamics import reference_identity
        result['thermodynamics'] = reference_identity()
    return result


class ResidualEnsemble:
    def __init__(self, family, settings, physics):
        if family not in FAMILIES:
            raise ValueError('Select a supported residual model.')
        self.family, self.settings, self.physics = family, copy.deepcopy(settings), copy.deepcopy(physics)
        self.members, self.history, self.bootstrap = [], [], []
        self.calibration = {'status': 'NOT_CALIBRATED', 'gap_wt_percent': None}

    def fit(self, data, indices, callback=None):
        settings = self.settings
        members = int(settings['members'])
        epochs = int(settings['epochs_baseline'] if self.family == FAMILIES[0] else settings['epochs_diffusion'])
        if not 1 <= members <= 5 or not 1 <= epochs <= 2000 or settings['learning_rate'] <= 0:
            raise ValueError('Use 1 to 5 members, positive epochs and learning rate.')
        torch.set_num_threads(int(settings.get('threads', 1)))
        x, y = data.x[indices], data.delta[indices]
        if x.shape[1] != len(feature_names(settings)):
            raise ValueError('Prepared feature schema differs from model settings; prepare with the same thermodynamic option.')
        weight = np.array([1/len(data.records[i]['composition'].elements) for i in indices])
        self.scaler = TrainingScaler().fit(x, weight)
        self.y_mean = float(np.average(y, weights=weight))
        self.y_scale = max(float(np.sqrt(np.average((y-self.y_mean)**2, weights=weight))), .05)
        self.compositions = sorted({data.records[i]['composition'].key for i in indices})
        self.has_mixture_training = any(len(data.records[i]['composition'].elements)>1 for i in indices)
        z = (y-self.y_mean)/self.y_scale
        self.clean_limit = max(6., float(np.max(np.abs(z)))+2.)
        self.elements = sorted({data.records[i]['recipe'].element for i in indices})
        self.origin = data.records[indices[0]]['raw']['origin']
        self.protocols = sorted({data.records[i]['raw']['protocol_id'] for i in indices})
        batches = np.array([str(data.records[i]['raw']['batch_id']) for i in indices])
        unique = sorted(set(batches))
        if len({data.records[i]['raw']['sample_id'] for i in indices}) < 16 or len(unique) < 2:
            raise ValueError('Fitting needs 16 eligible training specimens and at least two actual batches.')
        self.training_ids = [str(data.records[i]['raw']['sample_id']) for i in indices]
        fingerprint_frame = pd.DataFrame([data.records[i]['raw'] for i in indices]).astype(object)
        self.training_fingerprint = digest(fingerprint_frame.where(pd.notna(fingerprint_frame),None).to_dict('records'))
        for member in range(members):
            seed = int(settings['seed'])+member*101
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            blocks = rng.choice(unique, len(unique), replace=True)
            chosen = np.concatenate([np.flatnonzero(batches==b) for b in blocks])
            self.bootstrap.append({'member': member, 'batch_ids_with_replacement': blocks.tolist(),
                'sample_ids_with_replacement': [self.training_ids[i] for i in chosen]})
            xx = torch.from_numpy(self.scaler.transform(x[chosen]))
            zz = torch.tensor(z[chosen, None], dtype=torch.float32)
            ww = torch.tensor(weight[chosen]/weight[chosen].mean(), dtype=torch.float32)
            model = GaussianResidual(x.shape[1]) if self.family == FAMILIES[0] else ResidualDenoiser(x.shape[1])
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings['learning_rate']), weight_decay=1e-4)
            generator = torch.Generator().manual_seed(seed+1)
            for epoch in range(epochs):
                order = torch.randperm(len(xx), generator=generator)
                losses = []
                for ids in order.split(int(settings['batch_size'])):
                    optimizer.zero_grad()
                    if self.family == FAMILIES[0]:
                        mean, logvar = model(xx[ids])
                        per_row = .5*((zz[ids]-mean)**2*torch.exp(-logvar)+logvar).mean(dim=1)
                        loss = (per_row*ww[ids]).mean()
                    else:
                        per_row = training_loss(model, xx[ids], zz[ids], int(settings['diffusion_steps']), generator, reduction=False)
                        loss = (per_row*ww[ids]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError('Training became nonfinite; no model was committed.')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                if epoch == 0 or (epoch+1)%20 == 0 or epoch+1 == epochs:
                    event = {'family': self.family, 'member': member+1, 'members': members,
                             'epoch': epoch+1, 'epochs': epochs, 'training_loss': float(np.mean(losses))}
                    self.history.append(event)
                    if callback:
                        callback(event)
            self.members.append(model.eval())
        return self

    def predict(self, recipes, **kwargs):
        return self.predict_compositions([as_composition(r) for r in recipes], **kwargs)

    @torch.no_grad()
    def predict_compositions(self, compositions, draws=None, seed=None, origin=None, protocol=None):
        if not self.members or not compositions:
            raise ValueError('Fit a model and provide at least one composition.')
        compositions = [as_composition(c) for c in compositions]
        draws = int(self.settings['draws_per_member'] if draws is None else draws)
        if not 2 <= draws <= 512:
            raise ValueError('Use 2 to 512 draws per member.')
        seed = int(self.settings['seed']+700 if seed is None else seed)
        components = [(j, e) for j, c in enumerate(compositions) for e in c.elements]
        raw = np.array([component_features(compositions[j], e, self.physics,
            self.settings.get('thermodynamic_features', False)) for j,e in components])
        xx = torch.from_numpy(self.scaler.transform(raw))
        decoded, truncations = [], []
        for j, model in enumerate(self.members):
            generator = torch.Generator().manual_seed(seed+j*97)
            if self.family == FAMILIES[0]:
                mean, logvar = model(xx)
                z = (mean+torch.exp(.5*logvar)*torch.randn((len(xx), draws), generator=generator)).numpy()
                truncations.append(np.zeros(len(xx)))
            else:
                z, truncated = sample(model, xx, draws, int(self.settings['diffusion_steps']), seed+j*97, self.clean_limit)
                truncations.append(truncated)
            delta = z*self.y_scale+self.y_mean
            member = []
            start = 0
            for c in compositions:
                values = decode_composition(c, {e: delta[start+k] for k,e in enumerate(c.elements)}, self.physics)
                member.extend(values[e] for e in c.elements)
                start += len(c.elements)
            decoded.append(np.array(member))
        summary = summarize(np.asarray(decoded))
        outside = self.scaler.outside(raw)
        records = []
        cursor = 0
        for cindex, c in enumerate(compositions):
            size = len(c.elements)
            # Any unsupported mixture member invalidates the joint calibration claim.
            unsupported = bool(outside[cursor:cursor+size].any() or
                any(e not in self.elements for e in c.elements) or c.key not in self.compositions or
                (origin is not None and origin != self.origin) or
                (protocol is not None and protocol not in self.protocols))
            base = decode_composition(c, dict.fromkeys(c.elements, 0.), self.physics)
            from .thermodynamics import forecast_columns
            thermo = forecast_columns(c, self.settings.get('thermodynamic_features', False))
            for e in c.elements:
                i = cursor
                low, high = float(summary['low'][i]), float(summary['high'][i])
                maximum = prior(c.component(e), self.physics)['maximum_w_wt_percent']
                gap = self.calibration.get('gap_wt_percent')
                calibrated = gap is not None and not unsupported
                if calibrated:
                    low, high = max(0., low-gap), min(maximum, high+gap)
                mode = ('SINGLE_TO_MIXTURE_MASS_COUPLING_ONLY' if size>1 and not self.has_mixture_training
                    else 'UNSEEN_COMPOSITION_TRANSFER' if c.key not in self.compositions
                    else 'LEARNED_COMPOSITION_CONDITIONING' if size>1 else 'WITHIN_ELEMENT')
                if e not in self.elements:
                    mode = 'UNSEEN_ELEMENT_PHYSICS_GUIDED_TRANSFER'
                records.append({**asdict(c.component(e)), **control_domain_columns(c), **support_columns(e), **thermo, 'composition_index': cindex,
                    'composition_key': c.key, 'concentrations_mM_json': c.concentrations_json,
                    'model': self.family, 'model_origin': self.origin,
                    'predicted_mean_w_wt_percent': float(summary['mean'][i]),
                    'predicted_sd_w_wt_percent': float(summary['sd'][i]),
                    'within_model_sd_w_wt_percent': float(summary['within_sd'][i]),
                    'between_model_sd_w_wt_percent': float(summary['between_sd'][i]) if len(self.members)>1 else None,
                    'lower_95_w_wt_percent': low, 'upper_95_w_wt_percent': high,
                    'physical_prior_w_wt_percent': float(base[e]),
                    'unseen_element': e not in self.elements, 'outside_training_support': unsupported,
                    'transfer_mode': mode, 'has_mixture_training': self.has_mixture_training,
                    'calibration_status': 'BATCH_MAX_SPLIT_CALIBRATION' if calibrated else 'PROVISIONAL_UNCALIBRATED',
                    'diffusion_guard_fraction': float(np.mean(truncations, axis=0)[i]),
                    'model_id': self.model_id,
                    'uncertainty_method': 'Shared ensemble member, independent conditional residual noise, joint mass decoder; no learned full residual covariance'})
                cursor += 1
        return pd.DataFrame(records), summary['pooled']

    @property
    def model_id(self):
        return digest({'version': __version__, 'family': self.family, 'settings': self.settings,
            'physics': self.physics, 'training_fingerprint': self.training_fingerprint,
            'calibration': self.calibration,
            'references': model_references(self.settings)})[:20]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        weights = {f'member_{i}': model.state_dict() for i, model in enumerate(self.members)}
        torch.save(weights, directory/'weights.pt')
        from hashlib import sha256
        metadata = {'version': __version__, 'family': self.family, 'settings': self.settings, 'physics': self.physics,
            'feature_schema': 'physical-controls-chemistry-binary-mixing-v1' if self.settings.get('thermodynamic_features', False) else 'physical-controls-chemistry-v1',
            'scaler': self.scaler.to_dict(), 'y_mean': self.y_mean, 'y_scale': self.y_scale,
            'clean_limit': self.clean_limit, 'elements': self.elements, 'origin': self.origin, 'protocols': self.protocols,
            'training_ids': self.training_ids, 'training_fingerprint': self.training_fingerprint,
            'bootstrap': self.bootstrap, 'history': self.history, 'calibration': self.calibration,
            'feature_names': feature_names(self.settings), 'compositions': self.compositions, 'has_mixture_training': self.has_mixture_training, 'references': model_references(self.settings),
            'weights_sha256': sha256((directory/'weights.pt').read_bytes()).hexdigest(), 'model_id': self.model_id}
        (directory/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, directory):
        from hashlib import sha256
        directory = Path(directory)
        metadata = json.loads((directory/'metadata.json').read_text())
        if metadata['version'] != __version__ or metadata['feature_names'] != feature_names(metadata['settings']):
            raise ValueError('Checkpoint schema/version differs from this application.')
        if metadata['references'] != model_references(metadata['settings']):
            raise ValueError('Reference snapshots changed; retraining is required.')
        if sha256((directory/'weights.pt').read_bytes()).hexdigest() != metadata['weights_sha256']:
            raise ValueError('Checkpoint weights failed their integrity check.')
        obj = cls(metadata['family'], metadata['settings'], metadata['physics'])
        obj.scaler = TrainingScaler.from_dict(metadata['scaler'])
        for key in ('y_mean', 'y_scale', 'clean_limit', 'elements', 'origin', 'protocols', 'training_ids',
                    'training_fingerprint', 'bootstrap', 'history', 'calibration', 'compositions', 'has_mixture_training'):
            setattr(obj, key, metadata[key])
        weights = torch.load(directory/'weights.pt', map_location='cpu', weights_only=True)
        for i in range(int(obj.settings['members'])):
            model = GaussianResidual(len(metadata['feature_names'])) if obj.family == FAMILIES[0] else ResidualDenoiser(len(metadata['feature_names']))
            model.load_state_dict(weights[f'member_{i}'])
            obj.members.append(model.eval())
        torch.set_num_threads(int(obj.settings.get('threads', 1)))
        if obj.model_id != metadata['model_id']:
            raise ValueError('Checkpoint metadata failed its model identity check.')
        return obj
