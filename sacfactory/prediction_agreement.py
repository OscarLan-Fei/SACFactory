"""Post-result prediction accuracy, distinct from synthesis-result utility."""
import math

COMPONENT = 'component_agreement_v1'
LEGACY = 'legacy_inverse_mean_loss'
METHODS = {COMPONENT: 'Continuous ICP and probability agreement',
           LEGACY: 'Legacy inverse weighted error'}


def component_score(endpoint, loss):
    """Transform a paired error, never an observed loading target attainment."""
    loss = float(loss)
    if math.isnan(loss) or loss < 0:
        raise ValueError('Prediction losses must be nonnegative numbers.')
    if endpoint == 'icp':
        return 100 / (1 + loss)
    if not math.isfinite(loss) or loss > 1 + 1e-12:
        raise ValueError('Binary probability squared errors must lie in [0, 1].')
    return 100 * max(0., 1 - loss)


def combine(losses, weights, method=COMPONENT):
    """Require the fixed endpoint set; missing measurements are not failures."""
    if method not in METHODS:
        raise ValueError('Select a supported prediction-agreement definition.')
    if any(k not in losses or losses[k] is None for k in weights):
        return None
    if method == LEGACY:
        return 100 / (1 + sum(weights[k] * losses[k] for k in weights))
    return sum(weights[k] * component_score(k, losses[k]) for k in weights)
