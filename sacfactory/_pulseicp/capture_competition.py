"""Conservative dispersed/captured/clustered competition for electrospun precursors.

Pathways are literature-inspired; all numerical kinetic parameters are elicited.
Latent fractions are not measured single-atom populations or calibrated XRD events.
"""
from functools import lru_cache
import json
import math
from threading import RLock
import numpy as np
from .composition import as_composition
from .migration import parameter_draws, initial_feed_dose
from .element_migration import element_parameters
from .thermal_profile import KB_EV_K, phase_durations, heating_rate_C_s
from .references import thermal, positive
from .metal_prior import retention
from .configuration import physics_settings

CHANNELS=('capture','detachment','redispersion','site_activation','site_damage')


def validate_settings(cfg):
    if cfg.get('process_model','LEGACY_ENCOUNTER') not in {'CAPTURE_AGGREGATION_COMPETITION','LEGACY_ENCOUNTER'}:
        raise ValueError('Unknown XRD process model.')
    if isinstance(cfg['kinetic_phase_steps'],bool) or not isinstance(cfg['kinetic_phase_steps'],int) or not 4<=cfg['kinetic_phase_steps']<=256:
        raise ValueError('Use 4 to 256 numerical steps per ramp.')
    if isinstance(cfg['kinetic_max_cycles'],bool) or not isinstance(cfg['kinetic_max_cycles'],int) or cfg['kinetic_max_cycles']<1:
        raise ValueError('Kinetic solver cycle budget must be a positive integer.')
    for field in ('kinetic_initial_captured_fraction','kinetic_initial_clustered_fraction','kinetic_initial_active_site_fraction','kinetic_uncaptured_risk_mean'):
        if not math.isfinite(cfg[field]) or not 0<=cfg[field]<=1:
            raise ValueError('Initial state/site fractions and risk weights must lie in [0,1].')
    if cfg['kinetic_initial_captured_fraction']+cfg['kinetic_initial_clustered_fraction']>1:
        raise ValueError('Initial captured and clustered fractions cannot exceed one together.')
    for field in ('kinetic_site_capacity_reference_mean','kinetic_barrier_transfer_weight','kinetic_aggregation_rate_mean_s','kinetic_score_weight_mean'):
        if not math.isfinite(cfg[field]) or cfg[field]<0:
            raise ValueError('Kinetic capacity, transfer weight and rate/score means must be nonnegative.')
    if set(cfg['kinetic_rates'])!=set(CHANNELS):
        raise ValueError('Record all five kinetic channels explicitly.')
    for row in cfg['kinetic_rates'].values():
        for field in ('rate_at_reference_mean_s','rate_log_sd','barrier_eV_mean','barrier_eV_sd'):
            value=row[field]
            if isinstance(value,bool) or not math.isfinite(value) or value<0:
                raise ValueError('Kinetic rates, barriers and SDs must be finite and nonnegative.')
    if not isinstance(cfg['kinetic_volatilization_enabled'],bool):
        raise ValueError('Volatilization must be enabled with a boolean.')
    if set(cfg['kinetic_volatilization_weights'])!={'dispersed_uncaptured','captured','clustered'} or not all(positive(v) for v in cfg['kinetic_volatilization_weights'].values()):
        raise ValueError('All three state volatility weights must be finite and positive.')


def _lognormal(rng,mean,sd,n):
    return mean*rng.lognormal(-.5*sd*sd,sd,n)


@lru_cache(maxsize=16)
def draws(serialized):
    cfg=json.loads(serialized)
    rng=np.random.default_rng(cfg['seed']+5147)
    n=cfg['draws']
    rates={}
    for name in CHANNELS:
        row=cfg['kinetic_rates'][name]
        rates[name]=(_lognormal(rng,row['rate_at_reference_mean_s'],row['rate_log_sd'],n),
            np.maximum(0.,rng.normal(row['barrier_eV_mean'],row['barrier_eV_sd'],n)))
    return {'rates':rates,
        'capacity':_lognormal(rng,cfg['kinetic_site_capacity_reference_mean'],cfg['kinetic_site_capacity_log_sd'],n),
        'aggregation':_lognormal(rng,cfg['kinetic_aggregation_rate_mean_s'],cfg['kinetic_aggregation_rate_log_sd'],n),
        'uncaptured_risk':np.clip(rng.normal(cfg['kinetic_uncaptured_risk_mean'],cfg['kinetic_uncaptured_risk_sd'],n),0.,1.),
        'score':np.maximum(0.,rng.normal(cfg['kinetic_score_weight_mean'],cfg['kinetic_score_weight_sd'],n)),
        'volatility_weights':np.array([_lognormal(rng,cfg['kinetic_volatilization_weights'][name],cfg['kinetic_volatilization_weight_log_sd'],n) for name in ('dispersed_uncaptured','captured','clustered')])}


def _exp(log_rate):
    if not np.isfinite(log_rate).all() or np.max(log_rate)>600:
        raise ValueError('Competitive kinetic rates exceed the numerical domain.')
    return np.exp(log_rate)


class _History:
    """Incremental cycle prefixes, locked across UI sessions and returned as copies."""
    def __init__(self, key, serialized, reference_profile, physics=None):
        self.cfg=json.loads(serialized)
        c=as_composition(key)
        self.composition=c
        self.physics=physics_settings() if physics is None else physics
        cfg=self.cfg
        self.fractions=np.array([c.concentrations_mM[e] for e in c.elements],dtype=float)
        self.fractions/=self.fractions.sum()
        self.elements=c.elements
        n=cfg['draws']
        self.initial=np.zeros((4,len(c.elements),n))
        self.initial[1]=cfg['kinetic_initial_captured_fraction']
        self.initial[2]=cfg['kinetic_initial_clustered_fraction']
        self.initial[0]=1-self.initial[1]-self.initial[2]
        self.parameters=draws(serialized)
        from .composition import Composition
        ref=Composition({cfg['reference_element']:cfg['reference_aggregation_concentration_mM']},
            cfg['reference_temperature_C'],cfg['reference_pulses'],cfg['reference_pan_mass_g'],
            cfg['reference_volume_mL'],cfg['reference_carbonization_loss_fraction'])
        self.dose_ratio=initial_feed_dose(c)/initial_feed_dose(ref)
        self.maximum_capacity=self.parameters['capacity']/self.dose_ratio
        profile=cfg['migration_reference_profile'] if reference_profile else cfg['pulse_profile']
        duration=phase_durations(c.temperature_C,profile)
        migration=parameter_draws(serialized)
        fe,_,_=element_parameters('Fe',serialized,tuple(migration['barrier']))
        barriers=[]; prefactors=[]
        for e in c.elements:
            barrier,prefactor,_=element_parameters(e,serialized,tuple(migration['barrier']))
            barriers.append(barrier); prefactors.append(prefactor)
        barrier=np.asarray(barriers); prefactor=np.asarray(prefactors)
        offset=cfg['kinetic_barrier_transfer_weight']*(barrier-fe)
        cooling=profile['cooling_rate_min_C_s']+migration['cooling_u']*(profile['cooling_rate_max_C_s']-profile['cooling_rate_min_C_s'])
        steps=cfg['kinetic_phase_steps']
        delta=c.temperature_C-profile['baseline_temperature_C']
        heating=delta/heating_rate_C_s(profile)
        hold_steps=max(1,steps//4)
        waveform=[(profile['baseline_temperature_C']+delta*(j+.5)/steps,np.full(n,heating/steps)) for j in range(steps)]
        waveform += [(c.temperature_C,np.full(n,duration['hold_seconds']/hold_steps)) for _ in range(hold_steps)]
        waveform += [(c.temperature_C-delta*(j+.5)/steps,delta/cooling/steps) for j in range(steps)]
        self.waveform=[]
        escape=[]
        self.escape_metadata=[]
        for element in c.elements:
            energy=thermal().get(element,{}).get('heat_of_formation_kJ_mol')
            known=positive(energy)
            escape.append(energy/96.4853321233 if known else 0.)
            self.escape_metadata.append({'element':element,'escape_timing_status':'GAS_ATOM_ENERGY_TIMING_PROXY' if known else 'UNIFORM_TIME_ALLOCATION_MISSING_ESCAPE_DESCRIPTOR','gas_atom_energy_proxy_eV':energy/96.4853321233 if known else None})
        escape=np.array(escape)[:,None]
        escape_weights=[]
        reference_K=cfg['reference_temperature_C']+273.15
        for temperature,dt in waveform:
            if not np.any(dt):
                continue
            tK=temperature+273.15
            relative_diffusion=prefactor*_exp(fe/(KB_EV_K*reference_K)-barrier/(KB_EV_K*tK))
            rates={}
            for name,(base,ea) in self.parameters['rates'].items():
                effective=np.maximum(0.,ea+offset) if name in ('capture','detachment','redispersion') else ea
                rates[name]=base*_exp(ea/(KB_EV_K*reference_K)-effective/(KB_EV_K*tK))
            self.waveform.append((dt,relative_diffusion,rates))
            escape_weights.append(dt*np.exp(escape/KB_EV_K*(1/(c.temperature_C+273.15)-1/tK)))
        if not self.waveform:
            raise ValueError('Competitive kinetics needs a nonzero thermal-cycle duration.')
        escape_weights=np.asarray(escape_weights)
        self.escape_shares=escape_weights/escape_weights.sum(axis=0)
        self.lock=RLock()
        self.snapshots={0:(self.initial.copy(),np.full(n,cfg['kinetic_initial_active_site_fraction']),np.ones(n))}
        self.last=0

    def _cycle(self,state,active,intact,cycle):
        x=self.fractions
        if self.cfg['kinetic_volatilization_enabled']:
            from dataclasses import replace
            ends=np.array([retention(replace(self.composition.component(e),pulse_count=cycle),self.physics)[0] for e in self.elements])[:,None]
            starts=np.ones_like(ends) if cycle==1 else np.array([retention(replace(self.composition.component(e),pulse_count=cycle-1),self.physics)[0] for e in self.elements])[:,None]
            if np.any(ends<=0) or np.any(ends>starts):
                raise ValueError('Volatilization allocation needs a positive nonincreasing retention budget.')
            total_hazard=-np.log(ends/starts)
        else:
            total_hazard=np.zeros((len(self.elements),1))
        for step,(dt,diffusion,r) in enumerate(self.waveform):
            active=1-(1-active)*np.exp(-r['site_activation']*dt)
            intact=intact*np.exp(-r['site_damage']*dt)
            capacity=self.maximum_capacity*active*intact
            mobile,captured,clustered,escaped=state
            occupied=x@captured
            survival=np.minimum(1.,np.divide(capacity,occupied,out=np.ones_like(capacity),where=occupied>0))
            released=captured*(1-survival)
            captured=captured*survival
            mobile=mobile+released
            occupied=x@captured
            free=np.maximum(0.,1-np.divide(occupied,capacity,out=np.ones_like(capacity),where=capacity>0))
            a=r['capture']*free
            # Relative pair diffusivity D_i + D_j; mobile populations and initial
            # feed density determine collision supply, not element count alone.
            b=self.parameters['aggregation']*self.dose_ratio*.5*(diffusion*(x@mobile)+(x@(diffusion*mobile)))
            d,e=r['detachment'],r['redispersion']
            adt=a*dt; bdt=b*dt; ddt=d*dt; edt=e*dt
            dn=1+ddt; en=1+edt
            new_mobile=(mobile+ddt/dn*captured+edt/en*clustered)/(1+adt/dn+bdt/en)
            retained=captured/dn
            capture=adt/dn*new_mobile
            total_capture=x@capture
            remaining=np.maximum(0.,capacity-x@retained)
            scale=np.minimum(1.,np.divide(remaining,total_capture,out=np.ones_like(remaining),where=total_capture>0))
            new_captured=retained+capture*scale
            new_clustered=(clustered+bdt*new_mobile)/en
            new_mobile=new_mobile+capture*(1-scale)
            retained=np.array([new_mobile,new_captured,new_clustered])
            # The phase kernel allocates the existing per-pulse metal-loss budget;
            # it does not subtract a second loss from the ICP mass decoder.
            target_loss=retained.sum(axis=0)*(-np.expm1(-total_hazard*self.escape_shares[step]))
            losses=np.zeros_like(retained)
            remaining=target_loss.copy()
            weights=self.parameters['volatility_weights'][:,None,:]
            for _ in range(3):
                available=retained-losses
                weighted=weights*available
                denominator=weighted.sum(axis=0)
                allocation=weighted*np.divide(remaining,denominator,out=np.zeros_like(remaining),where=denominator>0)
                take=np.minimum(available,allocation)
                losses+=take
                remaining=np.maximum(0.,remaining-take.sum(axis=0))
            if np.max(remaining)>1e-10:
                raise ValueError('Volatilization allocation could not close its metal-loss budget.')
            retained-=losses
            state=np.concatenate((retained,(escaped+losses.sum(axis=0))[None,:,:]),axis=0)
        return state,active,intact

    def at(self,count):
        with self.lock:
            if count in self.snapshots:
                return tuple(v.copy() for v in self.snapshots[count])
            start=self.last if self.last<count else 0
            state,active,intact=(v.copy() for v in self.snapshots[start])
            for cycle in range(start+1,count+1):
                state,active,intact=self._cycle(state,active,intact,cycle)
            if not np.isfinite(state).all() or state.min() < -1e-10 or not np.allclose(state.sum(axis=0),1.,atol=1e-9):
                raise ValueError('Competitive solver failed its nonnegative atom-budget check.')
            self.snapshots[count]=(state.copy(),active.copy(),intact.copy())
            self.last=count
            # Keep requested prefixes bounded in memory; zero and latest survive.
            for old in sorted(self.snapshots):
                if len(self.snapshots)<=26:
                    break
                if old not in (0,self.last):
                    del self.snapshots[old]
            return state,active,intact


@lru_cache(maxsize=8)
def _history(key,serialized,reference_profile,serialized_physics):
    from .composition import Composition
    values=json.loads(key)
    values['concentrations_mM']=dict(values['concentrations_mM'])
    return _History(Composition(**values),serialized,reference_profile,json.loads(serialized_physics))


def simulate(composition,settings,reference_profile=False,physics=None):
    from dataclasses import asdict
    c=as_composition(composition)
    if c.pulse_count>settings['kinetic_max_cycles']:
        raise ValueError('COMPETITIVE_SOLVER_CYCLE_BUDGET')
    values=asdict(c)
    values['pulse_count']=1  # Prefix cache keys must not depend on requested N.
    physics=physics_settings() if physics is None else physics
    context=_history(json.dumps(values,sort_keys=True),json.dumps(settings,sort_keys=True),reference_profile,json.dumps(physics,sort_keys=True))
    state,active,intact=context.at(c.pulse_count)
    feed_summary=context.fractions@state
    retained=feed_summary[:3].sum(axis=0)
    summary=feed_summary[:3]/retained
    risk=summary[2]+context.parameters['uncaptured_risk']*summary[0]
    rows=[{'element':e,'molar_fraction':float(context.fractions[i]),
        'dispersed_uncaptured_fraction':float(np.mean(state[0,i]/state[:3,i].sum(axis=0))),
        'captured_fraction':float(np.mean(state[1,i]/state[:3,i].sum(axis=0))),
        'clustered_fraction':float(np.mean(state[2,i]/state[:3,i].sum(axis=0))),
        'retained_fraction_of_feed':float(state[:3,i].sum(axis=0).mean()),
        'volatilized_fraction_of_feed':float(state[3,i].mean()),
        **context.escape_metadata[i]}
        for i,e in enumerate(c.elements)]
    return risk,{'state':state,'fractions':summary,'feed_fractions':feed_summary,'active':active,'intact':intact,
        'capacity':context.maximum_capacity*active*intact,'elements':rows}


def competition_score(composition,reference,settings,physics=None):
    risk,details=simulate(composition,settings,physics=physics)
    reference_risk,_=simulate(reference,settings,reference_profile=True,physics=physics)
    params=draws(json.dumps(settings,sort_keys=True))
    score=params['score']*(risk-reference_risk)
    fractions=details['fractions']
    columns={'xrd_process_model':'CAPTURE_AGGREGATION_COMPETITION',
        'xrd_competition_status':'ELICITED_ELECTROSPUN_PRECURSOR_COMPETITION',
        'xrd_competition_initial_state':'UNIFORMLY_DISPERSED_PRECURSOR_ASSUMPTION' if settings['kinetic_initial_captured_fraction']+settings['kinetic_initial_clustered_fraction']==0 else 'DECLARED_INITIAL_FRACTIONS_ASSUMPTION',
        'xrd_competition_initial_fractions_json':json.dumps({'dispersed_uncaptured':1-settings['kinetic_initial_captured_fraction']-settings['kinetic_initial_clustered_fraction'],'captured':settings['kinetic_initial_captured_fraction'],'clustered':settings['kinetic_initial_clustered_fraction']},sort_keys=True),
        'xrd_competition_element_states_json':json.dumps(details['elements'],sort_keys=True),
        'xrd_competition_risk_proxy':float(risk.mean()),
        'xrd_competition_score_mean':float(score.mean()),
        'xrd_competition_reference_risk_proxy':float(reference_risk.mean()),
        'xrd_competition_active_site_fraction':float(details['active'].mean()),
        'xrd_competition_intact_site_fraction':float(details['intact'].mean()),
        'xrd_competition_maximum_atom_budget_error':float(np.max(np.abs(details['state'].sum(axis=0)-1.))),
        'xrd_competition_fixed_temperature_penalty_applied':False,
        'xrd_competition_redispersion_enabled':settings['kinetic_rates']['redispersion']['rate_at_reference_mean_s']>0,
        'xrd_competition_volatilization_enabled':settings['kinetic_volatilization_enabled'],
        'xrd_competition_volatilized_fraction_of_feed':float(details['feed_fractions'][3].mean()),
        'xrd_competition_retained_fraction_of_feed':float(details['feed_fractions'][:3].sum(axis=0).mean()),
        'xrd_competition_loss_basis':'MATCHED_EXISTING_ICP_RETENTION_PRIOR_NO_DOUBLE_SUBTRACTION',
        'xrd_competition_parameter_status':'ELICITED_NOT_EXPERIMENTALLY_FITTED',
        'xrd_competition_config_json':json.dumps(settings,sort_keys=True)}
    for i,name in enumerate(('dispersed_uncaptured','captured','clustered')):
        columns['xrd_competition_'+name+'_fraction']=float(fractions[i].mean())
        columns['xrd_competition_'+name+'_lower']=float(np.quantile(fractions[i],.025))
        columns['xrd_competition_'+name+'_upper']=float(np.quantile(fractions[i],.975))
    return score,columns
