"""Fixed observational utility transforms, independent of model predictions."""
import numpy as np

LEGACY = 'capped_attainment_v1'
CONTINUOUS = 'continuous_target_match_v1'
DEFINITIONS = {CONTINUOUS: 'Continuous target matching with screening discount',
               LEGACY: 'Legacy capped loading attainment'}
LOG_WIDTH = float(np.log(2.))
FAILURE_DISCOUNT = .25


def loading_utility(w, target, definition=LEGACY):
    """A factor-two departure gives exp(-1/2), a declared preference, not kinetics."""
    if definition not in DEFINITIONS:
        raise ValueError('Select a supported synthesis-result score definition.')
    values=np.asarray(w,dtype=float)
    if not np.isfinite(target) or target<=0 or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Nonnegative finite W and a positive target are required.')
    if definition==LEGACY:
        return np.minimum(values/target,1.)
    positive=np.maximum(values,np.finfo(float).tiny)
    answer=np.exp(-.5*(np.log(positive/target)/LOG_WIDTH)**2)
    return np.where(values==0,0.,answer)
