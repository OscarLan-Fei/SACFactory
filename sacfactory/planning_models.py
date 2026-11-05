"""Select fixed public inference releases or verified local planning checkpoints."""
import copy
import json
from pathlib import Path


def options(root, origin):
    base = Path(root) / 'sac_gen' / origin.lower()
    pointer = json.loads((base/'latest.json').read_text()) if (base/'latest.json').exists() else {}
    registry = json.loads((base/'variants.json').read_text()) if (base/'variants.json').exists() else {}
    result = {}
    for name, item in registry.get('variants', {}).items():
        result['Local '+name] = {**item, 'source': 'local', 'name': name}
    if pointer and not any(item['model_id'] == pointer['model_id'] for item in result.values()):
        result['Local deployment'] = {**pointer, 'source': 'local', 'name': pointer.get('name', 'SAC-Gen')}
    if origin == 'REAL':
        from .pretrained import available
        for name, entry in available().items():
            result['Pretrained '+name] = {'source': 'pretrained', 'name': name,
                                        'model_id': entry['sha256'][:20]}
    return result


def selection(root, origin, preferred=None):
    choices = options(root, origin)
    if preferred in choices:
        return preferred
    # Preserve existing deployments as the default for an established workspace.
    pointer = Path(root)/'sac_gen'/origin.lower()/'latest.json'
    active = json.loads(pointer.read_text()).get('model_id') if pointer.exists() else None
    return next((name for name, item in choices.items() if item['model_id'] == active),
                'Pretrained SAC-Gen-GB' if 'Pretrained SAC-Gen-GB' in choices else next(iter(choices), None))


def load(root, origin, name, samples):
    item = options(root, origin).get(name)
    if item is None:
        raise ValueError('The selected planning model is unavailable. Select an installed model.')
    if item['source'] == 'pretrained':
        from .pretrained import load as load_pretrained
        return load_pretrained(item['name'])
    run = item['run_id']
    if Path(run).name != run or run in {'', '.', '..'}:
        raise ValueError('Invalid local checkpoint path.')
    from .sac_gen import SACGen
    model = SACGen.load(Path(root)/'sac_gen'/origin.lower()/run, samples)
    if model.origin != origin or model.model_id != item['model_id']:
        raise ValueError('The local planning checkpoint identity or data space differs.')
    return model


def selected(store, samples):
    saved = store.get('meta', 'planning_model')
    if saved and saved.get('enabled') is False:
        return None
    preferred = saved.get('selection') if saved else None
    # An explicit selection must not silently switch learners after deletion.
    if not preferred:
        from .sac_gen import latest
        return latest(store.root, store.origin, samples)
    name = preferred
    return load(store.root, store.origin, name, samples) if name else None


def initial_context():
    from .experiment_defaults import PROCESS_DEFAULTS
    return {'sample_id': 'PROPOSED_CONTEXT', 'origin': 'REAL', 'elements': ['Fe'],
            'concentrations_mM': {'Fe': 50.}, 'pulse_count': 1, 'pulse_limit_C': 1000.,
            'temperature_C': None, 'pan_mass_g': 2., 'solvent_volume_mL': 20.,
            'carbonization_loss_fraction': .06, 'icp': [], **PROCESS_DEFAULTS}


def initial_space():
    """Proposed numeric search bounds, not measured device-validation evidence."""
    return {'concentration_bounds_mM': {'Fe': {'min_mM': 10., 'max_mM': 120.}},
            'pulses': [1, 2, 4], 'max_total_mM': 120., 'pulse_limit_min_C': 800.,
            'pulse_limit_max_C': 1400., 'temperature_step_C': 10, 'verified': False,
            'device_validation_recorded': False, 'range_basis': 'PROPOSED_SEARCH',
            'range_source': 'Proposed search bounds; review for the intended equipment',
            'baseline_repeat': False, 'sac_gen_candidate_budget': 256}


def prepare_context(context, model):
    result = copy.deepcopy(context)
    if getattr(model, 'is_public_release', False):
        from .precursors import documented
        for key, value in model.process_defaults.items():
            if not documented(result.get(key)):
                result[key] = value
            elif result[key] != value:
                raise ValueError('The pretrained model requires its PAN/DMF metal-salt process. Select a compatible local model for a different process.')
    return result
