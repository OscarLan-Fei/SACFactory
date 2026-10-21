"""Elicited cumulative encounter opportunity for rapid thermal cycles."""
from functools import lru_cache
import json
import math
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from .thermal_profile import equivalent_exposure, phase_durations, heating_rate_C_s, KB_EV_K
from .element_migration import element_parameters


@lru_cache(maxsize=16)
def parameter_draws(serialized):
    cfg = json.loads(serialized)
    n = cfg['draws']
    rng = np.random.default_rng(cfg['seed']+1709)
    barrier = np.maximum(0.,rng.normal(cfg['migration_barrier_eV_mean'],cfg['migration_barrier_eV_sd'],n))
    sd = cfg['migration_reference_length_ratio_log_sd']
    ratio = cfg['migration_reference_length_ratio_mean']*rng.lognormal(-.5*sd*sd,sd,n)
    survival = np.clip(rng.normal(cfg['migration_mobile_survival_mean'],cfg['migration_mobile_survival_sd'],n),0.,1.)
    weight = np.maximum(0.,rng.normal(cfg['migration_score_weight_mean'],cfg['migration_score_weight_sd'],n))
    return {'barrier':barrier,'length_ratio':ratio,'survival':survival,'weight':weight,
            'cooling_u':rng.uniform(0.,1.,n)}


def _effective_pulses(count, survival):
    """Sum mobile fractions across cycles without resetting traveled positions."""
    q = np.asarray(survival,dtype=float)
    result = np.ones_like(q)
    middle = (q>0)&(q<1)
    result[middle] = -np.expm1(count*np.log(q[middle]))/(1-q[middle])
    result[q==1] = float(count)
    return result


@lru_cache(maxsize=512)
def _exposure_context(temperature_C, serialized, element='Fe'):
    cfg = json.loads(serialized)
    params = parameter_draws(serialized)
    current,reference = cfg['pulse_profile'],cfg['migration_reference_profile']
    cooling = current['cooling_rate_min_C_s']+params['cooling_u']*(current['cooling_rate_max_C_s']-current['cooling_rate_min_C_s'])
    ref_cooling = reference['cooling_rate_min_C_s']+params['cooling_u']*(reference['cooling_rate_max_C_s']-reference['cooling_rate_min_C_s'])
    shared=tuple(params['barrier'])
    barrier,prefactor,metadata=element_parameters(element,serialized,shared)
    fe_barrier,_,_=element_parameters('Fe',serialized,shared)
    actual = equivalent_exposure(temperature_C,current,barrier,cfg['reference_temperature_C'],cooling)
    anchor = equivalent_exposure(cfg['reference_temperature_C'],reference,fe_barrier,cfg['reference_temperature_C'],ref_cooling)
    if np.any(anchor<=0):
        raise ValueError('Migration reference exposure must be positive.')
    # equivalent_exposure normalizes each element at T_ref. Restore its relative
    # Boltzmann rate before comparing different barriers to the common Fe anchor.
    log_rate=(fe_barrier-barrier)/(KB_EV_K*(cfg['reference_temperature_C']+273.15))
    if np.max(log_rate)>650:
        raise ValueError('Relative element mobility exceeds the numerical domain.')
    actual=actual*prefactor*np.exp(log_rate)
    metadata={**metadata,'element':element,
        'barrier_eV_mean':float(barrier.mean()),
        'barrier_eV_lower':float(np.quantile(barrier,.025)),
        'barrier_eV_upper':float(np.quantile(barrier,.975)),
        'relative_prefactor_mean':float(prefactor.mean()),
        'integrated_mobility_relative_to_Fe_reference':float(np.mean(actual/anchor))}
    return actual,anchor,metadata


def initial_feed_dose(composition):
    """Metal mmol per pre-pulse matrix gram; no true surface area is inferred."""
    mmol = sum(composition.concentrations_mM.values())*composition.volume_mL/1000.
    matrix = composition.pan_mass_g*(1-composition.carbonization_loss_fraction)
    return mmol/matrix


def _components(composition, reference, cfg):
    serialized=json.dumps(cfg,sort_keys=True)
    params=parameter_draws(serialized)
    effective=_effective_pulses(composition.pulse_count,params['survival'])
    dose_ratio=initial_feed_dose(composition)/initial_feed_dose(reference)
    total=sum(composition.concentrations_mM.values())
    fractions=np.array([composition.concentrations_mM[e]/total for e in composition.elements])
    steps,exposures,rows=[],[],[]
    for e,fraction in zip(composition.elements,fractions):
        actual,anchor,metadata=_exposure_context(composition.temperature_C,serialized,e)
        step=params['length_ratio']**2*(actual/anchor)*dose_ratio
        steps.append(step)
        exposures.append(actual)
        rows.append({**metadata,'molar_fraction':float(fraction),
            'cumulative_self_encounter_opportunity':float(np.mean(-np.expm1(-step*effective)))})
    one_step=np.asarray(steps)
    if not np.isfinite(one_step*effective).all():
        raise ValueError('Cumulative migration exposure exceeds the numerical domain.')
    return params,effective,fractions,one_step,np.asarray(exposures),rows


def migration_pair_table(composition, reference, settings):
    """All self/unlike pairs for one method; computed on demand, not per grid row."""
    _,effective,fractions,steps,_,_=_components(composition,reference,settings)
    records=[]
    for i,j in combinations_with_replacement(range(len(fractions)),2):
        weight=fractions[i]*fractions[j]*(1 if i==j else 2)
        hazard=.5*(steps[i]+steps[j])*effective
        opportunity=-np.expm1(-hazard)
        records.append({'element_a':composition.elements[i],'element_b':composition.elements[j],
            'pair_type':'SELF' if i==j else 'UNLIKE','molar_pair_weight':float(weight),
            'encounter_opportunity_mean':float(opportunity.mean()),
            'encounter_opportunity_lower':float(np.quantile(opportunity,.025)),
            'encounter_opportunity_upper':float(np.quantile(opportunity,.975)),
            'weighted_encounter_contribution':float(weight*opportunity.mean())})
    return pd.DataFrame(records)


def cumulative_migration(composition, reference, settings):
    """Return a score term and conditional encounter diagnostics, not XRD truth.

    Random-walk ratio: (L_rms / spacing)^2 ~ exposure * initial dose.
    Encounter opportunity = 1-exp(-ratio^2). Unknown diffusivity prefactor,
    spacing and support anchoring are absorbed into explicitly elicited priors.
    """
    cfg = settings
    zeros = np.zeros(cfg['draws'])
    if composition.temperature_C < cfg['pulse_profile']['baseline_temperature_C']:
        return zeros,{'xrd_migration_status':'PROFILE_UNAVAILABLE_BELOW_BASELINE',
            'xrd_migration_encounter_probability':None,'xrd_migration_reference_encounter_probability':None}
    params,effective,fractions,steps,exposures,element_rows=_components(composition,reference,cfg)
    one_step=fractions@steps
    squared_ratio=one_step*effective
    component_hazard=steps*effective
    # Sum x_i*x_j*[1-exp(-(eta_i+eta_j)/2)] exactly, including self pairs.
    # This factorization is linear in element count, with no N^2 grid storage.
    if len(fractions)==1:
        encounter=-np.expm1(-component_hazard[0])
    else:
        with np.errstate(divide='ignore'):
            log_no_encounter=2*np.log(fractions@np.exp(-component_hazard/2))
        encounter=-np.expm1(log_no_encounter)
    self_contribution=(fractions**2)@(-np.expm1(-component_hazard))
    unlike_contribution=np.maximum(0.,encounter-self_contribution)
    reference_encounter = -np.expm1(-params['length_ratio']**2)
    score = params['weight']*(encounter-reference_encounter)
    timing = phase_durations(composition.temperature_C,cfg['pulse_profile'])
    details = {'xrd_migration_status':'ELICITED_CUMULATIVE_ENCOUNTER_PRIOR',
        'xrd_migration_encounter_probability':float(encounter.mean()),
        'xrd_migration_encounter_lower':float(np.quantile(encounter,.025)),
        'xrd_migration_encounter_upper':float(np.quantile(encounter,.975)),
        'xrd_migration_reference_encounter_probability':float(reference_encounter.mean()),
        'xrd_migration_element_mode':'ELEMENT_CONDITIONED_PRIOR' if cfg.get('migration_element_specific',False) else 'SHARED_BARRIER_ABLATION',
        'xrd_migration_element_parameters_json':json.dumps(element_rows,sort_keys=True),
        'xrd_migration_self_encounter_contribution':float(self_contribution.mean()),
        'xrd_migration_unlike_encounter_contribution':float(unlike_contribution.mean()),
        'xrd_migration_unlike_pair_weight':float(max(0.,1-np.sum(fractions**2))),
        'xrd_migration_single_pulse_rms_to_spacing':float(np.sqrt(one_step).mean()),
        'xrd_migration_cumulative_rms_to_spacing':float(np.sqrt(squared_ratio).mean()),
        'xrd_migration_effective_pulses':float(effective.mean()),
        'xrd_migration_equivalent_hot_seconds_per_pulse':float(np.mean(fractions@exposures)),
        'xrd_migration_barrier_eV_mean':cfg['migration_barrier_eV_mean'],
        'xrd_profile_baseline_temperature_C':cfg['pulse_profile']['baseline_temperature_C'],
        'xrd_profile_heating_input_mode':cfg['pulse_profile'].get('heating_input_mode','AVERAGE_RATE'),
        'xrd_profile_parameters_json':json.dumps(cfg['pulse_profile'],sort_keys=True),
        'xrd_profile_heating_rate_C_s':heating_rate_C_s(cfg['pulse_profile']),
        'xrd_profile_peak_hold_ms':cfg['pulse_profile']['peak_hold_ms'],
        'xrd_profile_cooling_rate_min_C_s':cfg['pulse_profile']['cooling_rate_min_C_s'],
        'xrd_profile_cooling_rate_max_C_s':cfg['pulse_profile']['cooling_rate_max_C_s'],
        **{'xrd_profile_'+k:v for k,v in timing.items()}}
    return score,details
