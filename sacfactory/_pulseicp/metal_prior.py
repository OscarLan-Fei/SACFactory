"""Bounded empirical metal-retention prior with sourced thermal corrections."""
import math

from .references import elements, thermal, positive


def feed_mass(recipe):
    return recipe.concentration_mM * recipe.volume_mL * elements()[recipe.element]['atomic_weight'] / 1e6


def thermal_risk(element, temperature_C, settings):
    row, anchor = thermal()[element], thermal()['Fe']
    components = []
    for channel, field in [('melting', 'melting_point_K'), ('boiling', 'boiling_point_K')]:
        weight = settings['thermal_weights'][channel]
        if positive(row.get(field)) and weight > 0:
            components.append((weight, anchor[field]/row[field], field))
    weight = settings['thermal_weights']['escape']
    sublimation = positive(row.get('reported_sublimation_temperature_K')) and positive(row.get('sublimation_pressure_Pa')) and bool(row.get('sublimation_source')) and math.isclose(row['sublimation_pressure_Pa'], settings['reference_pressure_Pa'], rel_tol=1e-6)
    if weight > 0 and sublimation:
        components.append((weight, anchor['boiling_point_K']/row['reported_sublimation_temperature_K'], 'pressure_matched_sublimation_temperature'))
    elif weight > 0 and positive(row.get('heat_of_formation_kJ_mol')):
        components.append((weight, anchor['heat_of_formation_kJ_mol']/row['heat_of_formation_kJ_mol'], 'gas_atom_energy_proxy'))
    if not components:
        raise ValueError(f'No usable thermal channels for {element}.')
    factor = math.exp(sum(w*math.log(r) for w, r, _ in components)/sum(w for w, _, _ in components))
    risk = factor * ((temperature_C+273.15)/(settings['reference_temperature_C']+273.15)) ** settings['temperature_exponent']
    return risk, {'channels': [k for _, _, k in components], 'reference_weight_coverage': sum(w for w, _, _ in components)/sum(settings['thermal_weights'].values())}


def retention(recipe, settings):
    g, detail = thermal_risk(recipe.element, recipe.temperature_C, settings)
    carrier = settings['anchor_pan_g']*(1-settings['anchor_carbonization_loss'])*(1-settings['carrier_one_pulse_losses'][0])
    fraction = settings['anchor_fe_w_wt_percent']/100
    feed = settings['anchor_fe_mM']*settings['anchor_volume_mL']*elements()['Fe']['atomic_weight']/1e6
    a = fraction/(1-fraction)*carrier/feed
    b = settings['reference_repeat_ratio']
    if not 0 < a < 1 or not 0 < b < 1:
        raise ValueError('The anchor and repeat retention must lie in (0, 1).')
    first, plateau = math.exp(math.log(a)*g), math.exp(math.log(a*b)*g)
    tau = settings['metal_tau_pulses']/g**settings['metal_tau_exponent']
    if tau <= 0:
        raise ValueError('Metal pulse timescale must be positive.')
    r = first-(first-plateau)*(-math.expm1(-(recipe.pulse_count-1)/tau))
    return r, {**detail, 'thermal_risk': g, 'nominal_fe_anchor_retention': a}
