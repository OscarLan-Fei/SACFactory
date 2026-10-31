"""Reproducible random masks that keep each complete recipe together."""
import numpy as np
from sklearn.model_selection import GroupKFold
from .domain import recipe_key

RANDOM_RECIPE_CV = 'Random recipe-grouped cross-validation'


def recipe_folds(samples, n_splits=5, seed=42):
    """Assign groups without reading outcomes; each recipe is held out once."""
    groups = np.asarray([repr(recipe_key(sample)) for sample in samples])
    count = min(int(n_splits), len(set(groups)))
    if count < 2:
        return []
    splitter = GroupKFold(n_splits=count, shuffle=True, random_state=int(seed))
    return list(splitter.split(samples, groups=groups))


def assert_mask(train, test, recipe_grouped=False):
    """Validate specimen separation and the declared independence unit."""
    assert not {s['sample_id'] for s in train} & {s['sample_id'] for s in test}
    field = recipe_key if recipe_grouped else lambda s: s['batch_id']
    assert not {field(s) for s in train} & {field(s) for s in test}
