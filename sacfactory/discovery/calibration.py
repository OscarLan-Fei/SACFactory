"""Held-out probability and recipe-group conformal residual calibration."""
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from .common import identity,outcomes,recipe_id,values


class Calibration:
    def fit(self,samples,predictions,training_samples=(),evaluation_samples=(),alpha=.1):
        if len(samples)!=len(predictions) or not samples:raise ValueError('Aligned calibration observations are required.')
        if not 0<float(alpha)<1:raise ValueError('Alpha must be between zero and one.')
        namespaces={s.get('origin') for s in [*samples,*training_samples,*evaluation_samples]}
        if len(namespaces)!=1:raise ValueError('Calibration cannot mix data spaces.')
        groups={recipe_id(s) for s in samples}
        for pool in (training_samples,evaluation_samples):
            if groups & {recipe_id(s) for s in pool} or {identity(s) for s in samples}&{identity(s) for s in pool}:
                raise ValueError('Calibration, training and evaluation recipes must be disjoint.')
        self.alpha=float(alpha);residuals={};probabilities=[];labels=[]
        self.calibration_keys=[identity(s) for s in samples]
        for sample,forecast in zip(samples,predictions):
            actual,label=outcomes(sample);mean=values(forecast)
            common=set(actual)&set(mean)
            if common:
                error=max(abs(actual[e]-mean[e]) for e in common)
                key=recipe_id(sample);residuals[key]=max(residuals.get(key,0.),error)
            if label is not None and forecast.get('p_xrd_pass') is not None:
                probabilities.append(forecast['p_xrd_pass']);labels.append(label)
        errors=np.sort(list(residuals.values()));rank=math.ceil((len(errors)+1)*(1-self.alpha))
        self.radius=float(errors[rank-1]) if 1<=rank<=len(errors) else None
        self.probability_model=None
        if len(labels)>=20 and min(sum(labels),len(labels)-sum(labels))>=5:
            logits=np.log(np.clip(probabilities,1e-6,1-1e-6)/np.clip(1-np.array(probabilities),1e-6,1))[:,None]
            model=LogisticRegression(C=.1,max_iter=1000).fit(logits,labels)
            if model.coef_[0,0]>0:self.probability_model=model
        self.info={'independent_recipe_groups':len(errors),'alpha':self.alpha,'absolute_radius_wt_percent':self.radius,
                   'interval_status':'CALIBRATED_RECIPE_GROUP_MAX_RESIDUAL' if self.radius is not None else 'INSUFFICIENT_CALIBRATION_GROUPS',
                   'probability_status':'HELD_OUT_PLATT_FIT' if self.probability_model else 'UNCHANGED_INSUFFICIENT_CLASS_SUPPORT',
                   'scope':'Marginal recipe-group exchangeability; no conditional or distribution-shift coverage guarantee'}
        return self

    def apply(self,predictions):
        import copy
        result=copy.deepcopy(predictions)
        for row in result:
            p=row.get('p_xrd_pass')
            if p is not None and self.probability_model is not None:
                logit=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))
                row['p_xrd_pass']=float(self.probability_model.predict_proba([[logit]])[0,1])
                row['p_xrd_fail']=1-row['p_xrd_pass']
                row['xrd_pass_threshold']=.5
                row['predicted_xrd_state']='PASS' if row['p_xrd_pass']>=.5 else 'FAIL'
                row['xrd_probability_status']='HELD_OUT_PLATT_FIT_REQUIRES_EVALUATION'
            row['calibration']=self.info
            for assay in row.get('icp_predictions',[]):
                mean=assay['predicted_mean_w_wt_percent']
                assay['calibrated_lower_wt_percent']=max(0.,mean-self.radius) if self.radius is not None else None
                assay['calibrated_upper_wt_percent']=min(100.,mean+self.radius) if self.radius is not None else None
        return result


def disagreement(samples,model_predictions):
    rows=[]
    for i,sample in enumerate(samples):
        vectors=[values(p[i]) for p in model_predictions.values()]
        common=set.intersection(*(set(v) for v in vectors)) if vectors else set()
        spread=float(np.mean([np.std([v[e] for v in vectors]) for e in common])) if common else None
        probs=[p[i]['p_xrd_pass'] for p in model_predictions.values() if p[i].get('p_xrd_pass') is not None]
        rows.append({'recipe_key':recipe_id(sample),'ICP_model_SD_wt_percent':spread,
                     'structural_probability_SD':float(np.std(probs)) if len(probs)>1 else None,
                     'models_compared':list(model_predictions),
                     'status':'MODEL_DISAGREEMENT_NOT_CALIBRATED_PROBABILITY'})
    return rows


def learn_weights(samples,predictions,forbidden_samples=()):
    """Learn a simplex on a separate weight-fit pool, never interval calibration/test."""
    from scipy.optimize import minimize
    forbidden={recipe_id(s) for s in forbidden_samples}
    if forbidden&{recipe_id(s) for s in samples}:raise ValueError('Weight fitting overlaps a protected recipe group.')
    names=list(predictions)
    if len(names)<2:raise ValueError('Compare at least two models.')
    x=[];y=[];weight=[]
    for i,sample in enumerate(samples):
        actual,_=outcomes(sample);available=set(actual)
        for name in names:available &= set(values(predictions[name][i]))
        for element in available:
            x.append([values(predictions[name][i])[element] for name in names]);y.append(actual[element]);weight.append(1/len(available))
    if len(y)<8:raise ValueError('Need at least eight common quantified weight-fit targets.')
    x=np.array(x);y=np.array(y);weight=np.array(weight)
    fit=minimize(lambda w:np.average((x@w-y)**2,weights=weight)+.001*np.sum(w*w),
                 np.ones(len(names))/len(names),bounds=[(0,1)]*len(names),constraints={'type':'eq','fun':lambda w:w.sum()-1})
    if not fit.success:raise ValueError('Ensemble weight fit did not converge.')
    return dict(zip(names,(fit.x/fit.x.sum()).tolist()))


def fuse(predictions,weights):
    if set(predictions)!=set(weights) or any(w<0 for w in weights.values()) or not np.isclose(sum(weights.values()),1):
        raise ValueError('Use nonnegative aligned ensemble weights summing to one.')
    result=[]
    for i in range(len(next(iter(predictions.values())))):
        elements=set.intersection(*(set(values(p[i])) for p in predictions.values()))
        if any(set(values(p[i]))!=elements for p in predictions.values()):raise ValueError('Fusion needs complete matching component forecasts.')
        assays=[{'element':e,'predicted_mean_w_wt_percent':sum(weights[n]*values(predictions[n][i])[e] for n in weights)} for e in sorted(elements)]
        probabilities=[(weights[n],predictions[n][i].get('p_xrd_pass')) for n in weights if weights[n]>0 and predictions[n][i].get('p_xrd_pass') is not None]
        p=sum(w*v for w,v in probabilities)/sum(w for w,_ in probabilities) if probabilities else None
        result.append({'model_name':'Calibrated model ensemble','model_id':'ensemble','icp_predictions':assays,
                       'predicted_total_wt_percent':sum(a['predicted_mean_w_wt_percent'] for a in assays),
                       'p_xrd_pass':p,'p_xrd_fail':None if p is None else 1-p,
                       'outside_training_support':any(row[i].get('outside_training_support',True) for row in predictions.values())})
    return result


def protected_weights(samples,predictions,forbidden_samples=()):
    targets=0
    for i,sample in enumerate(samples):
        actual,_=outcomes(sample);common=set(actual)
        for forecasts in predictions.values():common &= set(values(forecasts[i]))
        targets+=len(common)
    if targets<8:
        if {recipe_id(s) for s in samples}&{recipe_id(s) for s in forbidden_samples}:
            raise ValueError('Weight pool overlaps protected recipes.')
        return {n:1/len(predictions) for n in predictions},'NOT_FITTED_EQUAL_WEIGHTS_INSUFFICIENT_TARGETS'
    return learn_weights(samples,predictions,forbidden_samples),'FITTED_ON_SEPARATE_WEIGHT_POOL'
