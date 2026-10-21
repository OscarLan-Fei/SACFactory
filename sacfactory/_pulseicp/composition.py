"""Joint mass fractions and shared element features for composition transfer."""
from dataclasses import dataclass
import json
import numpy as np

from .schema import Recipe
from .references import elements, thermal
from .features import features, PROPERTY_FIELDS, PROPERTY_SCALES
from .mass_decoder import prior, logit, sigmoid

INTERACTION_NAMES = ['mixture_log_count', 'mixture_log_total_concentration'] + \
    ['mixture_mean_'+k for k in PROPERTY_FIELDS] + ['mixture_missing_'+k for k in PROPERTY_FIELDS]


@dataclass(frozen=True)
class Composition:
    concentrations_mM: dict
    temperature_C: float = 1000.
    pulse_count: int = 1
    pan_mass_g: float = 2.
    volume_mL: float = 20.
    carbonization_loss_fraction: float = .06

    @property
    def elements(self):
        return sorted(self.concentrations_mM)

    def component(self, e):
        return Recipe(e, self.concentrations_mM[e], self.temperature_C, self.pulse_count,
                      self.pan_mass_g, self.volume_mL, self.carbonization_loss_fraction).validate()

    def validate(self):
        if not self.concentrations_mM:
            raise ValueError('Select at least one composition member.')
        for e in self.elements:
            self.component(e)
        return self

    @property
    def key(self):
        return '+'.join(self.elements)

    @property
    def concentrations_json(self):
        return json.dumps(self.concentrations_mM, sort_keys=True)


def as_composition(recipe):
    if isinstance(recipe, Composition):
        return recipe.validate()
    return Composition({recipe.element: recipe.concentration_mM}, recipe.temperature_C,
        recipe.pulse_count, recipe.pan_mass_g, recipe.volume_mL, recipe.carbonization_loss_fraction).validate()


def interaction_features(composition):
    n = len(composition.elements)
    if n == 1:
        return np.zeros(len(INTERACTION_NAMES))
    scale = np.log(n)
    weights = np.array([composition.concentrations_mM[e] for e in composition.elements], dtype=float)
    weights /= weights.sum()
    rows = []
    for e in composition.elements:
        values = {**elements()[e], **thermal()[e]}
        rows.append([np.nan if values.get(k) is None else values[k] for k in PROPERTY_FIELDS])
    raw = np.asarray(rows, dtype=float)
    missing = ~np.isfinite(raw)
    mean = weights @ (np.where(missing, 0., raw)/PROPERTY_SCALES)
    coverage = weights @ missing.astype(float)
    return scale*np.r_[1., np.log(sum(composition.concentrations_mM.values())/50.), mean, coverage]


def component_features(composition, e, settings, thermodynamic_features=False):
    values = np.r_[features(composition.component(e), settings), interaction_features(composition)]
    if thermodynamic_features:
        from .thermodynamics import feature_vector
        values = np.r_[values, feature_vector(composition, e)]
    return values


def decode_composition(composition, deltas, settings):
    """Independent conditional coordinates share weights and a mass denominator.

    The decoder couples mass fractions; it is not a learned full residual covariance.
    """
    matrices = {e: prior(composition.component(e), settings) for e in composition.elements}
    matrix = matrices[composition.elements[0]]['matrix_mass_g']
    metals = {e: p['feed_metal_g']*sigmoid(logit(p['retention_prior'])+deltas[e]) for e, p in matrices.items()}
    total = matrix+sum(metals.values())
    return {e: 100*mass/total for e, mass in metals.items()}


def inverse_composition(composition, observed, settings):
    if set(observed) != set(composition.elements):
        raise ValueError('The initial mixture inverse needs quantified ICP for every target; missing members are not zero.')
    values = np.array([observed[e] for e in composition.elements], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0) or values.sum() >= 100:
        raise ValueError('The inverse needs positive quantified targets summing below 100 wt%; boundary/censored values remain audited.')
    residuals = {}
    for e in composition.elements:
        p = prior(composition.component(e), settings)
        effective = observed[e]/(100-values.sum())*p['matrix_mass_g']/p['feed_metal_g']
        if not 0 < effective < 1:
            raise ValueError(f'{e} ICP is incompatible with the assumed mixture matrix/feed bound; never clip measured targets.')
        residuals[e] = float(logit(effective)-logit(p['retention_prior']))
    return residuals
