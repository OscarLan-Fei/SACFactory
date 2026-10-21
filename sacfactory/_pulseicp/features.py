"""Only future controls and frozen chemical references enter the predictor."""
import numpy as np

from .references import elements, thermal
from .mass_decoder import prior, logit

PROPERTY_FIELDS = ('atomic_weight', 'en_pauling', 'covalent_radius_cordero',
                   'melting_point_K', 'boiling_point_K', 'heat_of_formation_kJ_mol')
PROPERTY_SCALES = np.array([250., 4., 250., 5000., 5000., 1000.])
FEATURE_NAMES = ['log_concentration', 'scaled_temperature', 'log_pulses',
                 'concentration_temperature', 'concentration_pulse', 'temperature_pulse',
                 'log_pan_mass', 'log_volume', 'carbonization_loss', 'log_feed_mass',
                 'log_matrix_mass', 'prior_logit_retention', 'thermal_risk', 'thermal_coverage'] + \
                list(PROPERTY_FIELDS) + [f'{p}_missing' for p in PROPERTY_FIELDS]


def features(recipe, settings):
    recipe.validate()
    p = prior(recipe, settings)
    # Sixteen is the original feature scale, not a maximum allowed pulse count.
    c, t, n = np.log(recipe.concentration_mM/50.), (recipe.temperature_C-1000.)/500., np.log(recipe.pulse_count)/np.log(16.)
    process = [c, t, n, c*t, c*n, t*n, np.log(recipe.pan_mass_g/2.), np.log(recipe.volume_mL/20.),
               recipe.carbonization_loss_fraction, np.log(p['feed_metal_g']/.055845),
               np.log(p['matrix_mass_g']/1.6356), float(logit(p['retention_prior'])),
               p['thermal']['thermal_risk'], p['thermal']['reference_weight_coverage']]
    values = {**elements()[recipe.element], **thermal()[recipe.element]}
    data = np.array([np.nan if values.get(k) is None else values[k] for k in PROPERTY_FIELDS], dtype=float)
    missing = ~np.isfinite(data)
    result = np.r_[process, np.where(missing, 0., data)/PROPERTY_SCALES, missing.astype(float)]
    if not np.isfinite(result).all():
        raise ValueError('Future-control features must be finite.')
    return result


class TrainingScaler:
    def fit(self, x, weights=None):
        self.mean = np.average(x, axis=0, weights=weights)
        sd = np.sqrt(np.average((x-self.mean)**2, axis=0, weights=weights))
        # Raw bounds retain constant source descriptors for support checks.
        self.scale = np.where(sd < 1e-8, 1., sd)
        self.minimum, self.maximum = x.min(axis=0), x.max(axis=0)
        return self

    def transform(self, x):
        result = ((np.asarray(x, dtype=np.float32)-self.mean)/self.scale).astype(np.float32)
        # Unvarying inputs cannot have learned effects; suppress random NN extrapolation.
        result[:, np.abs(self.maximum-self.minimum) < 1e-8] = 0.
        return result

    def outside(self, x):
        x = np.asarray(x, dtype=np.float32)
        tolerance = 1e-6*np.maximum(1., np.maximum(np.abs(self.minimum), np.abs(self.maximum)))
        return np.any((x < self.minimum-tolerance) | (x > self.maximum+tolerance), axis=1)

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ('mean', 'scale', 'minimum', 'maximum')}

    @classmethod
    def from_dict(cls, values):
        obj = cls()
        for key, value in values.items():
            setattr(obj, key, np.asarray(value))
        return obj
