"""Explicitly synthetic combined ICP observations with a distinct teacher response."""
from datetime import datetime, timezone, timedelta
import math
import numpy as np
import pandas as pd

from .schema import Recipe, COLUMNS
from .mass_decoder import prior, logit, sigmoid


def response_shift(c, temperature, n):
    """Illustrative incorporation/condition interactions, not measured kinetics."""
    dose = math.log(c/50.)
    return (.35*math.tanh((temperature-850)/120) - .45*max(temperature-1450, 0)/350 +
            .35*(1-math.exp(-n/3))*math.exp(-((temperature-1100)/350)**2/2) -
            .40*n/16*(temperature-1000)/600 + .12*dose-.18*dose*dose)


def simulate(settings, n_samples=512, seed=20261014, element='Fe', day_offset=0):
    if n_samples < 64 or n_samples > 20000:
        raise ValueError('Choose 64 to 20000 synthetic specimens.')
    rng = np.random.default_rng(seed)
    rows = []
    reference = response_shift(50., 1000., 1)
    batch_effect = 0.
    for i in range(n_samples):
        if i % 4 == 0:
            batch_effect = float(rng.normal(0, .07))
        recipe = Recipe(element, float(np.exp(rng.uniform(np.log(10.), np.log(150.)))),
                        float(rng.uniform(800., 1800.)), int(rng.integers(1, 17)))
        p = prior(recipe, settings)
        shift = response_shift(recipe.concentration_mM, recipe.temperature_C, recipe.pulse_count)-reference
        r_teacher = float(sigmoid(logit(p['retention_prior']) + shift + batch_effect + rng.normal(0, .07)))
        # The teacher matrix differs slightly from the student's frozen matrix prior.
        matrix_teacher = p['matrix_mass_g'] * (1+.025*math.sin((recipe.temperature_C-1000)/500)+.01*(1-math.exp(-(recipe.pulse_count-1)/4)))
        w_true = 100*p['feed_metal_g']*r_teacher/(matrix_teacher+p['feed_metal_g']*r_teacher)
        measured = float(w_true*math.exp(rng.normal(0, .015)))
        date = datetime(2026, 1, 1, tzinfo=timezone.utc)+timedelta(days=day_offset+i//4)
        rows.append({'sample_id': f'SIM_{element}_{i+1:05d}', 'batch_id': f'SIM_{element}_BATCH_{i//4+1:04d}',
            **recipe.__dict__, 'icp_w_wt_percent': measured, 'icp_flag': 'MEASURED',
            'origin': 'SIMULATED', 'protocol_id': 'SYNTHETIC_PAN_DMF_PULSE_V1',
            'result_reported_at': date.isoformat()})
    return pd.DataFrame(rows, columns=COLUMNS)


def simulate_mixtures(settings, members=('Fe', 'Co'), n_samples=128, seed=20261019, day_offset=0):
    from .composition import Composition
    if len(set(members))<2 or not 64<=n_samples<=20000:
        raise ValueError('Use at least two distinct members and 64 to 20000 specimens.')
    rng = np.random.default_rng(seed)
    tag = '_'.join(sorted(members))
    rows, batch_effect = [], 0.
    reference = response_shift(50., 1000., 1)
    for i in range(n_samples):
        if i%4==0:
            batch_effect = rng.normal(0, .07)
        c = {e:float(np.exp(rng.uniform(np.log(10.),np.log(80.)))) for e in members}
        comp = Composition(c, float(rng.uniform(800,1800)), int(rng.integers(1,17))).validate()
        ps = {e:prior(comp.component(e), settings) for e in members}
        interaction = -.15*np.log(len(members))-.1*max(sum(c.values())/150.-1,0)
        metals = {}
        for e in members:
            shift = response_shift(c[e], comp.temperature_C, comp.pulse_count)-reference
            r = sigmoid(logit(ps[e]['retention_prior'])+shift+interaction+batch_effect+rng.normal(0,.07))
            metals[e] = ps[e]['feed_metal_g']*r
        matrix = next(iter(ps.values()))['matrix_mass_g']*(1+.025*math.sin((comp.temperature_C-1000)/500)+.01*(1-math.exp(-(comp.pulse_count-1)/4)))
        denominator = matrix+sum(metals.values())
        for e in members:
            measured = float(100*metals[e]/denominator*math.exp(rng.normal(0,.015)))
            date = datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(days=day_offset+i//4)
            rows.append({'sample_id':f'SIM_{tag}_{i+1:05d}', 'batch_id':f'SIM_{tag}_BATCH_{i//4+1:04d}',
                **comp.component(e).__dict__, 'icp_w_wt_percent':measured, 'icp_flag':'MEASURED',
                'origin':'SIMULATED', 'protocol_id':'SYNTHETIC_PAN_DMF_PULSE_V1', 'result_reported_at':date.isoformat()})
    return pd.DataFrame(rows, columns=COLUMNS)


def simulate_campaign(settings, seed=20261014):
    """1280 specimens: Fe, other singles, then mixtures; all outcomes are synthetic."""
    frames = []
    offset = 0
    for j,(e,count) in enumerate([('Fe',512),('Co',192),('Ni',192),('Cu',128)]):
        frames.append(simulate(settings,count,seed+j*101,e,offset))
        offset += (count+3)//4
    for j,members in enumerate([('Fe','Co'),('Fe','Co','Ni')]):
        frames.append(simulate_mixtures(settings,members,128,seed+1000+j*101,offset))
        offset += 32
    return pd.concat(frames, ignore_index=True)
