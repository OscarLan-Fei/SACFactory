"""Prespecified synthesis-result utility and protected predictions of that utility."""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .outcome_validation import ENDPOINTS, normalized_weights
from .compute_runtime import jobs
from .result_utility import LEGACY,CONTINUOUS,DEFINITIONS,loading_utility,FAILURE_DISCOUNT

VERSION = 'operational-synthesis-system-score-v4-continuous-target'
DIRECT = 'Direct random forest'
PRIOR_RESIDUAL = 'Physical-prior residual'
PULSE_GAUSSIAN='PulseICP Gaussian score fusion'
PULSE_DIFFUSION='PulseICP diffusion score fusion'
PULSE_TRANSFORMER='PulseICP Transformer score fusion'
PULSE_DIT='PulseICP diffusion-transformer score fusion'
PULSE_MODELS={PULSE_GAUSSIAN:'Gaussian residual',PULSE_DIFFUSION:'Conditional diffusion',
    PULSE_TRANSFORMER:'Transformer residual',PULSE_DIT:'Diffusion Transformer'}
MODELS = [DIRECT, PRIOR_RESIDUAL,*PULSE_MODELS]
COMPONENTS = {**ENDPOINTS, 'icp': 'Whole-recipe ICP target attainment',
    'burnout': 'Sample recovery / no complete burnout'}


def configuration(weights=None, target_wt_percent=2.5):
    """The target is whole-recipe total W, never a per-element sum of targets."""
    if isinstance(target_wt_percent,bool) or not math.isfinite(float(target_wt_percent)) or not 0 < float(target_wt_percent) <= 100:
        raise ValueError('System-score total ICP target must be in (0, 100] wt%.')
    weights=normalized_weights({'screening':.7,'icp':.3} if weights is None else weights,.3)
    return weights,float(target_wt_percent)


def observed_score(record, weights, target,score_definition=LEGACY):
    """Missing evidence gives bounds; explicit complete burnout has zero utility.

    The burnout rule defines the synthesis-result utility only. It neither
    imputes W=0 nor changes unknown individual assay or structural labels.
    """
    if score_definition not in DEFINITIONS:
        raise ValueError('Select a supported synthesis-result score definition.')
    sample=record['sample']
    selected=sample['elements']
    quantified=bool(selected) and all(e in record['assays'] for e in selected)
    total=sum(record['assays'][e] for e in selected) if quantified else None
    components={k:record['outcomes'][k] for k in ENDPOINTS if k!='icp'}
    components['burnout']=1-components['burnout'] if components['burnout'] is not None else None
    loading=float(loading_utility(total,target,score_definition)) if total is not None and 0 <= total <= 100 else None
    components['icp']=loading
    if score_definition==CONTINUOUS and 'screening' in weights and weights.get('icp',0)>0:
        screening=components['screening']
        components['icp']=loading*(screening+FAILURE_DISCOUNT*(1-screening)) if loading is not None and screening is not None else None
    missing=[k for k in weights if components[k] is None]
    known=sum(weights[k]*components[k] for k in weights if components[k] is not None)
    lower=100*known
    upper=100*(known+sum(weights[k] for k in missing))
    score=lower if not missing else None
    status='COMPLETE' if not missing else 'INCOMPLETE_EVIDENCE'
    if sample.get('sample_condition')=='BURNED_OUT':
        score=lower=upper=0.
        status='EXPLICIT_BURNOUT_ZERO_UTILITY'
    return {'observed_system_score':score,'observed_system_score_lower':lower,
        'observed_system_score_upper':upper,'system_score_status':status,
        'system_missing_components':', '.join(missing),'system_total_target_wt_percent':total,
        'system_score_definition':score_definition,'system_loading_target_match':loading,
        **{f'system_component_{k}':v for k,v in components.items()}}


def prediction_settings(model, prior_strength):
    if model not in MODELS:
        raise ValueError('Select a supported synthesis system-score model.')
    if isinstance(prior_strength,bool) or not math.isfinite(float(prior_strength)) or not 0 <= float(prior_strength) <= 1:
        raise ValueError('System-score prior strength must be in [0, 1].')
    return model,float(prior_strength)


def prior_baseline(sample, weights, target, strength,score_definition=LEGACY):
    """Frozen controls-only loading contribution; never an observed utility."""
    from . import pulseicp_bridge as bridge
    from ._pulseicp.composition import decode_composition
    physics=bridge.configuration()['physics']
    composition=bridge.composition(sample,physics)
    values=decode_composition(composition,dict.fromkeys(composition.elements,0.),physics)
    total=float(sum(values.values()))
    if not math.isfinite(total) or not 0 <= total <= 100:
        raise ValueError('The physical prior did not produce a valid total mass percent.')
    return 100*weights.get('icp',0.)*strength*float(loading_utility(total,target,score_definition)),total


def fit_fold(train, test, weights, target, trees=100, seed=42, model=DIRECT, prior_strength=1.,pulse_settings=None,
             score_definition=LEGACY):
    """Predict E[recorded utility | controls] from outer-training samples only.

    This direct head learns the same fixed utility computed from observations.
    It is not F(mean W, marginal label probabilities), which in general differs
    from E[F(outcomes)] because attainment saturation and burnout are nonlinear.
    """
    model,prior_strength=prediction_settings(model,prior_strength)
    if score_definition not in DEFINITIONS:
        raise ValueError('Select a supported synthesis-result score definition.')
    def observed(row):
        return observed_score(row,weights,target) if score_definition==LEGACY else observed_score(row,weights,target,score_definition)
    uses_prior=model==PRIOR_RESIDUAL and weights.get('icp',0.)>0 and prior_strength>0
    uses_learned=model in PULSE_MODELS and weights.get('icp',0.)>0 and prior_strength>0
    learned,learned_info={},None
    if uses_learned:
        from .pulseicp_score import loading_baselines
        learned,learned_info=loading_baselines(train,test,train[0]['sample']['origin'],PULSE_MODELS[model],
            pulse_settings,weights['icp'],target,prior_strength,
            **({} if score_definition==LEGACY else {'score_definition':score_definition}))
    prior_values,prior_exclusions={},[]
    train_ids={r['sample']['sample_id'] for r in train}
    for r in [*train,*test]:
        ident=r['sample']['sample_id']
        if uses_learned:
            if ident in learned:
                prior_values[ident]=(learned[ident]['contribution'],learned[ident]['total_mean_wt_percent'])
            else:
                prior_exclusions.append({'sample_id':ident,'scope':'Training' if ident in train_ids else 'Prediction',
                    'reason':'No protected learned ICP baseline; inspect PulseICP fusion folds and exclusions.'})
        elif uses_prior:
            try:
                prior_values[ident]=prior_baseline(r['sample'],weights,target,prior_strength,score_definition)
            except (ValueError,TypeError,KeyError,OverflowError) as exc:
                prior_exclusions.append({'sample_id':ident,'reason':str(exc),
                    'scope':'Training' if ident in train_ids else 'Prediction'})
        else:
            prior_values[ident]=(0.,None)
    source=[(r,observed(r)) for r in train if r['sample']['sample_id'] in prior_values]
    source=[(r,v) for r,v in source if v['observed_system_score'] is not None]
    values=[v['observed_system_score'] for _,v in source]
    source_batches={r['sample']['batch_id'] for r,_ in source}
    source_recipes={r['recipe'] for r,_ in source}
    available=len(source)>=12 and len(source_batches)>=2 and len(source_recipes)>=3 and len(set(values))>=2
    predictions={}
    available_test=[r for r in test if r['sample']['sample_id'] in prior_values]
    if available:
        bases=np.array([prior_values[r['sample']['sample_id']][0] for r,_ in source])
        forest=RandomForestRegressor(n_estimators=trees,min_samples_leaf=3,max_depth=16,
            random_state=seed,n_jobs=jobs()).fit([r['x'] for r,_ in source],np.asarray(values)-bases)
        if available_test:
            bases=np.array([prior_values[r['sample']['sample_id']][0] for r in available_test])
            predicted=np.clip(bases+forest.predict([r['x'] for r in available_test]),0.,100.)
            predictions=dict(zip([r['sample']['sample_id'] for r in available_test],predicted.tolist()))
    baseline=float(np.mean(values)) if values else None
    rows=[]
    for r in test:
        ident=r['sample']['sample_id']
        predicted=predictions.get(ident)
        prior_contribution,prior_w=prior_values.get(ident,(None,None))
        result=observed(r)
        rows.append({'sample_id':r['sample']['sample_id'],'batch_id':r['sample']['batch_id'],
            'composition':'+'.join(r['sample']['elements']),
            'system_screening_observed':r['outcomes']['screening'],
            'predicted_system_score':predicted,'baseline_system_score':baseline,
            'system_prior_total_wt_percent':None if uses_learned else prior_w,
            'system_prior_loading_contribution':None if uses_learned else prior_contribution,
            'system_loading_baseline_contribution':prior_contribution,
            'system_learned_total_wt_percent':learned.get(ident,{}).get('total_mean_wt_percent'),
            'system_learned_expected_attainment':learned.get(ident,{}).get('expected_attainment'),
            'system_loading_outside_support':learned.get(ident,{}).get('outside_training_support'),
            'system_prediction_model':model,**result})
    from .pulseicp_bridge import identity
    info={'version':VERSION,'score_definition':score_definition,'family':model,'prior_strength':prior_strength,'prior_used':uses_prior,
        'prior_snapshot':identity() if uses_prior or uses_learned else None,'prior_exclusions':prior_exclusions,
        'learned_loading_used':uses_learned,'pulseicp_fusion':learned_info,
        'training_n':len(source),'training_sample_ids':[r['sample']['sample_id'] for r,_ in source],
        'training_batch_ids':sorted(source_batches),'distinct_training_scores':len(set(values)),
        'available':available,'source_mean':baseline,'weights':weights,'target_wt_percent':target,
        'reason':'' if available else 'Need 12 complete training scores, 2 batches, 3 recipes and 2 distinct scores'}
    return rows,info


def summary(rows):
    frame=pd.DataFrame(rows)
    paired=frame.dropna(subset=['predicted_system_score','observed_system_score']) if len(frame) else frame
    metrics={'evaluated_samples':len(paired),'predicted_samples':int(frame.predicted_system_score.notna().sum()) if len(frame) else 0,
        'observed_samples':int(frame.observed_system_score.notna().sum()) if len(frame) else 0,
        'mae_score_points':None,'rmse_score_points':None,'r_squared':None,'baseline_mae_score_points':None,
        'failed_samples_evaluated':0,'failed_sample_mae_score_points':None,'incomplete_observations':0,
        'linear_fit_slope':None,'linear_fit_intercept':None,'linear_fit_r_squared':None}
    if len(frame):
        metrics['incomplete_observations']=int(frame.observed_system_score.isna().sum())
    if len(paired):
        observed=paired.observed_system_score.to_numpy(dtype=float)
        predicted=paired.predicted_system_score.to_numpy(dtype=float)
        metrics.update(mae_score_points=float(mean_absolute_error(observed,predicted)),
            rmse_score_points=float(np.sqrt(mean_squared_error(observed,predicted))),
            baseline_mae_score_points=float(mean_absolute_error(observed,paired.baseline_system_score)))
        if len(paired)>1 and np.ptp(observed)>1e-12:
            metrics['r_squared']=float(r2_score(observed,predicted))
        if len(paired)>1 and np.ptp(predicted)>1e-12:
            slope,intercept=np.linalg.lstsq(np.column_stack([predicted,np.ones(len(predicted))]),observed,rcond=None)[0]
            metrics.update(linear_fit_slope=float(slope),linear_fit_intercept=float(intercept))
            if np.ptp(observed)>1e-12:
                metrics['linear_fit_r_squared']=float(r2_score(observed,intercept+slope*predicted))
        failed=paired.loc[paired.system_screening_observed==0]
        metrics['failed_samples_evaluated']=len(failed)
        if len(failed):
            metrics['failed_sample_mae_score_points']=float(mean_absolute_error(failed.observed_system_score,failed.predicted_system_score))
    return metrics
