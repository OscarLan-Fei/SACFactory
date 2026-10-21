"""Mass closure and assumption-conditioned inverse ICP coordinates."""
import math
import numpy as np

from .carbon_prior import matrix_mass
from .metal_prior import feed_mass, retention


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0., -z))


def logit(r):
    r = np.asarray(r, dtype=float)
    if not np.isfinite(r).all() or np.any((r <= 0) | (r >= 1)):
        raise ValueError('Logit coordinates require values strictly inside (0, 1).')
    return np.log(r)-np.log1p(-r)


def prior(recipe, settings):
    recipe.validate()
    carrier, feed = matrix_mass(recipe, settings), feed_mass(recipe)
    r, details = retention(recipe, settings)
    if not 0 < r < 1 or not carrier > 0:
        raise ValueError('This prior is outside the scalar residual coordinate domain.')
    return {'matrix_mass_g': carrier, 'feed_metal_g': feed, 'retention_prior': r,
            'retained_metal_g': feed*r, 'lost_metal_g': feed*(1-r),
            'product_mass_g': carrier+feed*r, 'physical_w_wt_percent': 100*feed*r/(carrier+feed*r),
            'maximum_w_wt_percent': 100*feed/(carrier+feed), 'thermal': details}


def decode(recipe, delta, settings):
    p = prior(recipe, settings)
    r = sigmoid(logit(p['retention_prior']) + np.asarray(delta))
    metal = p['feed_metal_g']*r
    return 100*metal/(p['matrix_mass_g']+metal)


def inverse_response(recipe, measured_w, settings):
    w = float(measured_w)
    if not math.isfinite(w) or not 0 < w < 100:
        raise ValueError('The initial residual transform needs quantified W strictly between 0 and 100; zero/boundary observations remain audited.')
    p = prior(recipe, settings)
    r = (w/(100-w))*p['matrix_mass_g']/p['feed_metal_g']
    if not 0 < r < 1:
        raise ValueError('ICP is incompatible with the declared matrix/feed bound; review inputs or assumptions, never clip measured targets.')
    return float(logit(r)-logit(p['retention_prior']))


def decode_mixture(matrix_g, feeds, retentions):
    """Forward physical accounting for future mixtures; no mixture ML claim."""
    if not math.isfinite(matrix_g) or matrix_g <= 0 or not feeds or set(feeds) != set(retentions):
        raise ValueError('A positive matrix and matching element sets are required.')
    metals = {}
    for e, mass in feeds.items():
        r = retentions[e]
        if not math.isfinite(mass) or mass <= 0 or not math.isfinite(r) or not 0 <= r <= 1:
            raise ValueError('Positive feeds and bounded retained fractions required.')
        metals[e] = mass*r
    product = matrix_g+sum(metals.values())
    return {'product_mass_g': product, 'matrix_mass_g': matrix_g, 'retained_metal_g': metals,
            'w_wt_percent': {e: 100*m/product for e, m in metals.items()}}
