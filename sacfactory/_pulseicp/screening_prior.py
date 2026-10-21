"""Explicitly elicited XRD-pass prior, with assumption sensitivity rather than validation."""
import json
import math
from functools import lru_cache
import numpy as np
import pandas as pd
from .composition import as_composition
from .mass_decoder import prior
from .schema import Recipe
from .references import ROOT, digest, elements
from .thermodynamics import descriptors, reference_identity, R
from .ensemble import model_references
from .thermal_profile import validate_profile
from .migration import cumulative_migration
from .element_migration import validate_element_settings
from .capture_competition import competition_score, validate_settings as validate_kinetics


def prior_settings():
    return json.loads((ROOT/'configs/screening_prior.json').read_text(encoding='utf-8'))


def _dose(composition, physics):
    """Prior-retained metal mmol per gram matrix; no measured ICP or XRD input."""
    retained_mmol = 0.
    matrix = None
    for e in composition.elements:
        p = prior(composition.component(e), physics)
        retained_mmol += 1000*p['feed_metal_g']*p['retention_prior']/elements()[e]['atomic_weight']
        matrix = p['matrix_mass_g']
    return retained_mmol/matrix


@lru_cache(maxsize=16)
def _parameter_draws(serialized):
    cfg = json.loads(serialized)
    count = cfg['draws']
    if not isinstance(count,int) or not 32 <= count <= 8192:
        raise ValueError('Use 32 to 8192 assumption-sensitivity draws.')
    rng = np.random.default_rng(cfg['seed'])
    values = {}
    for field in ('dose','temperature','low_temperature','pulse'):
        values[field] = rng.normal(cfg[field+'_coefficient_mean'],cfg[field+'_coefficient_sd'],count)
    for field in ('dose','temperature','low_temperature'):
        values[field] = np.maximum(values[field],0.)
    values['intercept'] = rng.normal(0.,cfg['intercept_sd'],count)
    values['log_reference_shift'] = rng.normal(0.,cfg['reference_log_dose_sd'],count)
    # Lognormal multiplier has unit mean; a zero mixing weight stays exactly zero.
    sd = cfg['mixing_weight_relative_sd']
    values['mixing'] = cfg['mixing_weight_mean']*rng.lognormal(-.5*sd*sd,sd,count)
    values['missing_mixing'] = rng.normal(0.,cfg.get('missing_mixing_score_sd',1.5),count)
    return values


def predict_xrd_prior(compositions, physics, settings=None):
    cfg = prior_settings() if settings is None else settings.copy()
    if cfg.get('process_model','LEGACY_ENCOUNTER') not in {'CAPTURE_AGGREGATION_COMPETITION','LEGACY_ENCOUNTER'}:
        raise ValueError('Unknown XRD process model.')
    for key,value in cfg.items():
        if isinstance(value,bool):
            if key not in {'migration_enabled','migration_element_specific','kinetic_volatilization_enabled'}:
                raise ValueError('Boolean values are only valid for migration switches.')
            continue
        if isinstance(value,(float,int)) and not math.isfinite(value):
            raise ValueError('Prior parameters must be finite numeric values.')
        if (key.endswith('_sd') or key=='mixing_weight_mean') and value < 0:
            raise ValueError('Prior SDs and mixing weight must be nonnegative.')
    if cfg['pulse_saturation_tau'] <= 0:
        raise ValueError('Pulse saturation tau must be positive.')
    if cfg.get('migration_enabled',False):
        if cfg.get('process_model')=='CAPTURE_AGGREGATION_COMPETITION':
            validate_kinetics(cfg)
        validate_element_settings(cfg)
        validate_profile(cfg['pulse_profile'])
        validate_profile(cfg['migration_reference_profile'])
        if not 0 <= cfg['migration_mobile_survival_mean'] <= 1 or cfg['migration_reference_length_ratio_mean'] <= 0 or cfg['migration_barrier_eV_mean'] < 0 or cfg['migration_score_weight_mean'] < 0:
            raise ValueError('Use nonnegative migration parameters, positive length ratio and mobile survival in [0,1].')
    params = _parameter_draws(json.dumps(cfg,sort_keys=True))
    reference = Recipe(cfg['reference_element'],cfg['reference_aggregation_concentration_mM'],
        cfg['reference_temperature_C'],cfg['reference_pulses'],cfg['reference_pan_mass_g'],
        cfg['reference_volume_mL'],cfg['reference_carbonization_loss_fraction']).validate()
    ref_dose = _dose(as_composition(reference),physics)
    model_id = digest({'settings':cfg,'physics':physics,'references':model_references({'thermodynamic_features':True})})[:20]
    rows = []
    for i,raw in enumerate(compositions):
        c = as_composition(raw)
        d = descriptors(c.concentrations_mM,c.temperature_C)
        dose = _dose(c,physics)
        log_ratio = math.log(dose/ref_dose)
        hot = (c.temperature_C-cfg['reference_temperature_C'])/500.
        cold = (max(1000.-c.temperature_C,0.)/200.)**2
        pulse = 1-math.exp(-(c.pulse_count-1)/cfg['pulse_saturation_tau'])
        migration_score = None
        migration_details = {'xrd_migration_status':'DISABLED_LEGACY_PULSE_SCORE'}
        if cfg.get('migration_enabled',False):
            migration_score,migration_details = cumulative_migration(c,as_composition(reference),cfg)
            if migration_details['xrd_migration_status']=='PROFILE_UNAVAILABLE_BELOW_BASELINE':
                migration_score = None
        competitive=cfg.get('process_model')=='CAPTURE_AGGREGATION_COMPETITION' and cfg.get('migration_enabled',False)
        process_details={'xrd_process_model':'LEGACY_ENCOUNTER','xrd_competition_status':'NOT_REQUESTED'}
        kinetic_score=None
        if competitive:
            if c.temperature_C<cfg['pulse_profile']['baseline_temperature_C']:
                process_details['xrd_competition_status']='LEGACY_FALLBACK_PROFILE_BELOW_BASELINE'
            elif c.pulse_count>cfg['kinetic_max_cycles']:
                process_details['xrd_competition_status']='LEGACY_FALLBACK_COMPETITIVE_SOLVER_CYCLE_BUDGET'
            else:
                kinetic_score,process_details=competition_score(c,as_composition(reference),cfg,physics)
        legacy_temperature_score=params['temperature']*hot+params['low_temperature']*cold
        if kinetic_score is not None:
            process_score=kinetic_score
        else:
            process_score=legacy_temperature_score+(migration_score if migration_score is not None else params['pulse']*pulse)
        # A negative alloy G can favor a mixed cluster; it is not an SAC entropy bonus.
        mix = -1000*d['g_mix_proxy_kJ_mol_atoms']/(R*(c.temperature_C+273.15)) if d['complete'] else None
        # A missing free energy is not zero: explicitly omit its contribution and
        # add declared assumption sensitivity for the unmodeled pair coverage.
        missing_fraction = 1-(d['pair_weight_coverage'] or 0.) if not d['complete'] else 0.
        score = (params['intercept']+params['dose']*(log_ratio-params['log_reference_shift'])+
            process_score+(params['mixing']*mix if mix is not None else 0.)+
            params['missing_mixing']*missing_fraction*int(cfg['mixing_weight_mean']>0))
        p_pass = 1/(1+np.exp(np.clip(score,-700.,700.)))
        rows.append({'composition_index':i,'xrd_prior_pass_probability':float(p_pass.mean()),
            **migration_details,
            **process_details,
            'xrd_migration_score_applied':bool(kinetic_score is None and migration_score is not None),
            'xrd_fixed_temperature_penalty_applied':bool(kinetic_score is None),
            'xrd_prior_sensitivity_lower':float(np.quantile(p_pass,.025)),
            'xrd_prior_sensitivity_upper':float(np.quantile(p_pass,.975)),
            'xrd_prior_status':'UNVALIDATED_ELICITED_PRIOR' if d['complete'] else 'UNVALIDATED_PROCESS_PRIOR_INCOMPLETE_MIXING', 'xrd_prior_model_id':model_id,
            'xrd_prior_mixing_applied':bool(d['complete'] and cfg['mixing_weight_mean']>0),
            'xrd_prior_missing_pairs':json.dumps(d['missing_pairs']),
            'xrd_prior_effective_retained_dose_mmol_per_g_matrix':dose,
            'xrd_prior_reference_dose_mmol_per_g_matrix':ref_dose,
            'xrd_prior_mixing_drive_over_RT':mix,
            'xrd_prior_mixing_weight_mean':cfg['mixing_weight_mean'],
            'xrd_prior_table_sha256':reference_identity()['binary_table_sha256']})
    return pd.DataFrame(rows)
