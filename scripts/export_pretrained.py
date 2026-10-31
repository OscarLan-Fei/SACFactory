"""Export allowlisted inference parameters, never an entire private checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sacfactory.sac_gen import SACGen
from sacfactory.pretrained import FORMAT
from sacfactory.accelerated_ensemble import feature_names, model_references

INFERENCE_SETTINGS = ('seed', 'draws_per_member', 'diffusion_steps', 'thermodynamic_features',
    'members', 'token_width', 'attention_heads', 'attention_layers')
NODE_FIELDS = ('value', 'feature_idx', 'num_threshold', 'missing_go_to_left', 'left', 'right', 'is_leaf')


def export(checkpoint, destination):
    model = SACGen.load(checkpoint)
    loading = model.loading
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stem = 'sac_gen_gb' if model.family == 'Gradient boosting residual' else 'sac_gen_diffusion'
    settings = {key: loading.settings[key] for key in INFERENCE_SETTINGS if key in loading.settings}
    # Process labels describe the application protocol, not a measured specimen.
    context = {key: model.training_samples[0][key] for key in ('precursor_family', 'support_batch', 'protocol_id')}
    head_keys = ('mean', 'scale', 'coefficient', 'intercept', 'pass_threshold', 'calibration')
    head = {key: model.head[key] for key in head_keys if key in model.head}
    if 'calibration' in head:
        head['calibration'] = {key: head['calibration'][key] for key in ('method', 'coefficient', 'intercept')}
    meta = {'format': FORMAT, 'name': model.variant_name, 'family': model.family,
        'profile': model.profile, 'settings': settings, 'physics': loading.physics,
        'feature_names': feature_names(settings), 'references': model_references(settings),
        'process_defaults': context, 'head': head, 'scaler': loading.scaler.to_dict(),
        'supported_elements': sorted(loading.elements),
        'disclosure': 'Learned inference parameters and aggregate normalizers only; no experimental rows, identifiers, split memberships or training logs.'}
    if model.family == 'Gradient boosting residual':
        trees = []
        for iteration in loading.regressor._predictors:
            for tree in iteration:
                if tree.nodes['is_categorical'].any():
                    raise ValueError('This public tree format supports numerical nodes only.')
                trees.append({key: tree.nodes[key].tolist() for key in NODE_FIELDS})
        meta['trees'] = {'baseline': float(loading.regressor._baseline_prediction.item()), 'trees': trees}
    else:
        arrays = {f'member_{i}/{key}': tensor.detach().cpu().numpy()
                  for i, network in enumerate(loading.members) for key, tensor in network.state_dict().items()}
        path = destination / (stem + '.weights.npz')
        np.savez_compressed(path, **arrays)
        meta['weights'] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for key in ('y_mean', 'y_scale', 'clean_limit', 'has_mixture_training'):
            meta[key] = getattr(loading, key)
    path = destination / (stem + '.json')
    path.write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return model.variant_name, {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    name, entry = export(args.checkpoint, args.output)
    registry = args.output / 'registry.json'
    content = json.loads(registry.read_text()) if registry.exists() else {'format': FORMAT, 'models': {}}
    content['models'][name] = entry
    registry.write_text(json.dumps(content, indent=2)+'\n', encoding='utf-8')
    print('Exported inference-only model:', name)


if __name__ == '__main__':
    main()
