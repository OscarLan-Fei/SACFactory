"""Explicit method grids and physical/prior or fitted-model predictions."""
from dataclasses import asdict
from itertools import product
import numpy as np
import pandas as pd

from .schema import Recipe, numeric, control_domain_columns
from .mass_decoder import prior
from .references import support_columns


def parse_pulse_counts(values):
    """Accept preset or newly typed positive counts without an application cap."""
    result = set()
    for value in values:
        n = numeric(value, 'Pulse count')
        if n < 1 or not n.is_integer():
            raise ValueError('Every pulse count must be a positive integer.')
        result.add(int(n))
    return sorted(result)


def method_grid(element, concentrations, temperatures, pulses, pan_mass_g=2., volume_mL=20., carbonization_loss_fraction=.06):
    count = len(concentrations)*len(temperatures)*len(pulses)
    if not 1 <= count <= 5000:
        raise ValueError('Use 1 to 5000 methods per preview.')
    return [Recipe(element, float(c), float(t), int(n) if float(n).is_integer() else n,
                   pan_mass_g, volume_mL, carbonization_loss_fraction).validate()
            for c, t, n in product(concentrations, temperatures, pulses)]


def physical_forecasts(recipes, settings):
    from .composition import as_composition, decode_composition
    from .thermodynamics import forecast_columns
    rows = []
    for i,r in enumerate(recipes):
        c = as_composition(r)
        values = decode_composition(c,dict.fromkeys(c.elements,0.),settings)
        thermo = forecast_columns(c)
        for e in c.elements:
            rows.append({**asdict(c.component(e)), **control_domain_columns(c), **support_columns(e), **thermo, 'composition_index':i, 'composition_key':c.key,
                'concentrations_mM_json':c.concentrations_json, 'model':'Unfitted physical prior',
                'predicted_mean_w_wt_percent':float(values[e]), 'predicted_sd_w_wt_percent':None,
                'lower_95_w_wt_percent':None, 'upper_95_w_wt_percent':None,
                'calibration_status':'UNCERTAINTY_NOT_ESTIMATED'})
    return pd.DataFrame(rows)


def composition_grid(concentrations, temperatures, pulses, pan_mass_g=2., volume_mL=20., carbonization_loss_fraction=.06):
    from .composition import Composition
    keys = sorted(concentrations)
    count = int(np.prod([len(concentrations[e]) for e in keys]))*len(temperatures)*len(pulses)
    if not keys or not 1<=count<=5000:
        raise ValueError('Choose members and 1 to 5000 physical methods per preview.')
    return [Composition(dict(zip(keys,values)),float(t),n,pan_mass_g,volume_mL,carbonization_loss_fraction).validate()
        for values,t,n in product(product(*(concentrations[e] for e in keys)),temperatures,pulses)]
