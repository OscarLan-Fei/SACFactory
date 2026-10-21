"""Synthetic observation layer: censored ICP and a separate crystalline-phase surrogate."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import expit

from .chemical_descriptors import reference
from .demo_physics import dose_factor,mass_balance
from .thermal_loss import elemental_loss_reference


@lru_cache(maxsize=1)
def observation_settings():
    raw=(Path(__file__).resolve().parents[1]/'resources/demo_observation_settings.json').read_bytes()
    result=json.loads(raw)
    result['settings_sha256']=hashlib.sha256(raw).hexdigest()
    return result


def apply_batch_retention(balance,scale):
    """Shared batch variability changes retained mass, never feed or mass closure."""
    for row in balance['elements'].values():
        row['retention_fraction']=min(1.,max(0.,row['retention_fraction']*scale))
        row['retained_metal_g']=row['feed_metal_g']*row['retention_fraction']
        row['lost_metal_g']=row['feed_metal_g']-row['retained_metal_g']
    balance['total_retained_metal_g']=sum(r['retained_metal_g'] for r in balance['elements'].values())
    balance['recovered_sample_g']=balance['carrier_after_pulse_g']+balance['total_retained_metal_g']
    for row in balance['elements'].values():
        row['true_w_wt_percent']=100*row['retained_metal_g']/balance['recovered_sample_g']
    balance['simulated_batch_retention_factor']=scale


def aggregation_parameters(balance,temperature_C,pulse_count,settings=None):
    """An explicit observation prior, not an atomic-structure measurement."""
    settings=settings or observation_settings()
    atoms={r['symbol']:r for r in reference()['elements']}
    anchor=mass_balance({'Fe':50},1000,1)
    anchor_mmol=1000*anchor['elements']['Fe']['retained_metal_g']/atoms['Fe']['atomic_weight']
    capacity=settings['aggregation_reference_input_mmol']*anchor_mmol/anchor['carrier_after_pulse_g']
    amount={e:1000*r['retained_metal_g']/atoms[e]['atomic_weight']/dose_factor(e) for e,r in balance['elements'].items()}
    total=sum(amount.values())
    if total>0:
        log_melting=sum(amount[e]*math.log(elemental_loss_reference('Fe')['melting_point_K']/elemental_loss_reference(e)['melting_point_K']) for e in amount)/total
    else:
        log_melting=0.
    mobility=math.exp(settings['mobility_temperature_coefficient']*((temperature_C+273.15)/1273.15-1)
        +settings['mobility_melting_exponent']*log_melting)
    mobility*=1+settings['repeat_mobility_increment']*(1-math.exp(-(pulse_count-1)/settings['repeat_mobility_tau']))
    burden=(total/balance['carrier_after_pulse_g'])/capacity
    aggregation=float(expit((burden*mobility-1)/settings['aggregation_transition_width']))
    conversion=float(expit((settings['conversion_reference_temperature_C']-temperature_C)/settings['conversion_transition_width_C']))
    return {'retained_effective_mmol':total,'assumed_capacity_effective_mmol_per_g':capacity,
        'relative_site_burden':burden,'mobility_factor':mobility,
        'aggregation_probability':aggregation,'crystalline_residue_probability':conversion,
        'xrd_fail_probability':1-(1-aggregation)*(1-conversion)}


def observe_assays(balance,rng,settings=None):
    settings=settings or observation_settings()
    assays=[]; details={}
    lod,loq=settings['simulated_icp_lod_wt_percent'],settings['simulated_icp_loq_wt_percent']
    if not 0<lod<loq:
        raise ValueError('Synthetic assay limits require 0 < LOD < LOQ.')
    for element,row in balance['elements'].items():
        latent=row['true_w_wt_percent']
        measured=min(100.,max(0.,latent*math.exp(float(rng.normal(0,settings['simulated_icp_log_sd'])))))
        flag='ND' if measured<lod else 'BLQ' if measured<loq else 'MEASURED'
        value=f'<{lod:g}' if flag=='ND' else f'<{loq:g}' if flag=='BLQ' else f'{measured:.8f}'
        assays.append({'element':element,'value':value,'flag':flag,'unit':'wt%','method_id':'',
            'source':'SIMULATED detected above LOD but below LOQ' if flag=='BLQ' else 'SIMULATED assay with declared noise/censoring; not an ICP instrument result'})
        details[element]={'latent_true_w_wt_percent':latent,'simulated_uncensored_reading_wt_percent':measured,
                         'lod_wt_percent':lod,'loq_wt_percent':loq,'reported_flag':flag,
                         'detected':measured>=lod,'quantified':measured>=loq}
    state='PASS' if all(a['flag']!='ND' for a in assays) else 'FAIL'
    return assays,state,details
