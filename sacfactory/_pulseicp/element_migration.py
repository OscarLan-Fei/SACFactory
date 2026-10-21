"""Descriptor-conditioned migration assumptions, not tabulated SAC diffusivities."""
from functools import lru_cache
import hashlib
import json
import math
import numpy as np
from .references import elements, thermal, positive


def _rng(seed, element):
    suffix=int.from_bytes(hashlib.sha256(element.encode()).digest()[:4],'little')
    return np.random.default_rng(np.random.SeedSequence([seed,2317,suffix]))


def validate_element_settings(cfg):
    if not isinstance(cfg.get('migration_element_specific',False),bool):
        raise ValueError('Element-specific migration must be enabled with a boolean.')
    for row in cfg.get('migration_descriptor_barrier_slopes',{}).values():
        if not all(math.isfinite(float(row[k])) for k in ('mean_eV','sd_eV')) or row['sd_eV']<0:
            raise ValueError('Descriptor barrier slopes need finite means and nonnegative SDs.')
    allowed={'melting_point_K','boiling_point_K','heat_of_formation_kJ_mol'}
    if set(cfg.get('migration_descriptor_barrier_slopes',{}))-allowed:
        raise ValueError('Unknown migration descriptor; add a sourced reference channel explicitly.')
    for e,row in cfg.get('migration_element_overrides',{}).items():
        if e not in elements():
            raise ValueError('Migration override needs a recognized element.')
        for key in ('barrier_eV','barrier_sd_eV','prefactor_ratio_to_Fe','prefactor_log_sd'):
            value=row.get(key)
            if isinstance(value,bool) or value is None or not math.isfinite(float(value)) or value<0:
                raise ValueError('Migration overrides need finite nonnegative barrier, SD and prefactor values.')
        if row['prefactor_ratio_to_Fe']<=0 or (e=='Fe' and row['prefactor_ratio_to_Fe']!=1.):
            raise ValueError('Prefactors must be positive; the Fe-relative reference prefactor is one.')
        basis=row.get('basis')
        if basis not in {'ELICITED_ASSUMPTION','EXPERIMENTAL','DFT'}:
            raise ValueError('Override basis must distinguish an assumption, experiment or DFT.')
        if basis!='ELICITED_ASSUMPTION' and (not row.get('source') or not row.get('support_state')):
            raise ValueError('External barriers require source and matching support/state metadata.')


def _override(element, cfg):
    row=cfg.get('migration_element_overrides',{}).get(element)
    if row is None:
        return None
    rng=_rng(cfg['seed']+919,element)
    n=cfg['draws']
    barrier=np.maximum(0.,rng.normal(row['barrier_eV'],row['barrier_sd_eV'],n))
    sd=row['prefactor_log_sd']
    prefactor=row['prefactor_ratio_to_Fe']*rng.lognormal(-.5*sd*sd,sd,n) if element!='Fe' else np.ones(n)
    return barrier,prefactor,{'parameter_origin':row['basis'],'source':row.get('source',''),
        'support_state':row.get('support_state','UNSPECIFIED_ASSUMPTION'),
        'available_descriptors':[],'missing_descriptors':[],
        'note':'Recorded external inputs still require matching support, state and uncertainty validation.'}


@lru_cache(maxsize=512)
def element_parameters(element, serialized, shared_barrier_tuple):
    """Stable per-element draws share descriptor slopes and the Fe anchor.

    Missing properties stay missing. A zero-centered discrepancy represents their
    unresolved effect; it is not an imputed melting point or measured barrier.
    """
    cfg=json.loads(serialized)
    base=np.asarray(shared_barrier_tuple,dtype=float)
    if not cfg.get('migration_element_specific',False):
        return base,np.ones_like(base),{'parameter_origin':'SHARED_BARRIER_ABLATION',
            'available_descriptors':[],'missing_descriptors':[]}
    explicit=_override(element,cfg)
    if explicit is not None:
        return explicit
    fe_override=_override('Fe',cfg)
    if fe_override is not None:
        base=fe_override[0]
    if element=='Fe':
        return base,np.ones_like(base),{'parameter_origin':'ELICITED_FE_REFERENCE',
            'available_descriptors':[],'missing_descriptors':[]}
    ref,row=thermal()['Fe'],thermal().get(element,{})
    barrier=base.copy()
    rng=_rng(cfg['seed'],element)
    slopes_rng=np.random.default_rng(cfg['seed']+4117)
    available,missing=[],[]
    for field,coefficient in sorted(cfg['migration_descriptor_barrier_slopes'].items()):
        slope=slopes_rng.normal(coefficient['mean_eV'],coefficient['sd_eV'],cfg['draws'])
        if positive(row.get(field)) and positive(ref.get(field)):
            barrier+=slope*math.log(row[field]/ref[field])
            available.append(field)
        else:
            missing.append(field)
    barrier+=rng.normal(0.,cfg['migration_element_residual_sd'],cfg['draws'])
    if missing:
        barrier+=rng.normal(0.,cfg['migration_missing_descriptor_sd']*math.sqrt(len(missing)),cfg['draws'])
    sd=cfg['migration_log_prefactor_sd']
    prefactor=rng.lognormal(-.5*sd*sd,sd,cfg['draws'])
    return np.maximum(0.,barrier),prefactor,{
        'parameter_origin':'ELICITED_DESCRIPTOR_PRIOR',
        'available_descriptors':available,'missing_descriptors':missing,
        'note':'Bulk thermal descriptors condition an elicited barrier distribution; no support-specific diffusion table is claimed.'}
