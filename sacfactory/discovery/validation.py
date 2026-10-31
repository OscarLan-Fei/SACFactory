"""Protected model/calibration validation and finite-pool acquisition replay."""
import copy
import numpy as np
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from .common import digest,identity,outcomes,prediction_metrics,query,recipe_id,values
from .calibration import Calibration,protected_weights,fuse,disagreement
from .set_model import SetResidual


def usable(samples,origin):
    if any(s.get('origin')!=origin for s in samples):raise ValueError('Validation cannot mix namespaces.')
    result=[]
    for sample in samples:
        try:
            query(sample);actual,label=outcomes(sample)
            if actual or label is not None:result.append(sample)
        except (ValueError,TypeError):continue
    return result


def factories(seed=42,epochs=80,include_sac=True):
    from ..data_only import DataOnlyModel
    from ..science_only import SciencePriorModel
    from ..weighted_validation import collect
    from ..sac_gen import SACGen
    from .. import pulseicp_bridge as bridge
    def data(samples,origin):
        records,_=collect(samples,{},origin,'chemistry_pulseicp')
        return DataOnlyModel('Gradient boosting',seed).fit_records(records,origin)
    result={'Data GB':data,'Fixed prior':lambda s,o:SciencePriorModel(o),
            'SAC-Gen-Set':lambda s,o:SetResidual(epochs=epochs,seed=seed).fit(s,o)}
    if include_sac:
        settings={**bridge.configuration()['model'],'seed':seed,'crossfit_grouping':'recipe','refine_xrd':True}
        result['SAC-Gen-GB']=lambda s,o:SACGen('Gradient boosting residual',settings).fit(s,o,{})
    return result


def validate(samples,origin,folds=5,epochs=80,seed=42,alpha=.1,include_sac=True,progress=None):
    samples=usable(samples,origin)
    groups=np.array([recipe_id(s) for s in samples]);unique=len(set(groups))
    if unique<20 or len(samples)<30:raise ValueError('Protected validation needs 30 observations and 20 distinct recipes.')
    if not 2<=int(folds)<=min(10,unique):raise ValueError('Select 2-10 supported folds.')
    metrics=[];details=[];point_rows=[]
    for fold,(outer,test) in enumerate(GroupKFold(int(folds),shuffle=True,random_state=seed).split(samples,groups=groups),1):
        fit_idx,cal_idx=next(GroupShuffleSplit(test_size=.3,random_state=seed+fold).split(outer,groups=groups[outer]))
        fit=outer[fit_idx];cal=outer[cal_idx]
        weight_idx,interval_idx=next(GroupShuffleSplit(test_size=.5,random_state=seed+100+fold).split(cal,groups=groups[cal]))
        weight=cal[weight_idx];interval=cal[interval_idx]
        pools=[[samples[i] for i in index] for index in (fit,weight,interval,test)]
        training,weight_samples,calibration_samples,testing=pools
        sets=[{recipe_id(s) for s in pool} for pool in pools]
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
        models={};weight_preds={};cal_preds={};test_preds={}
        for name,factory in factories(seed+fold,epochs,include_sac).items():
            if progress:progress({'fold':fold,'folds':int(folds),'stage':'Fitting '+name})
            model=factory(training,origin);models[name]=model
            weight_preds[name]=model.predict([query(s) for s in weight_samples])
            cal_preds[name]=model.predict([query(s) for s in calibration_samples])
            test_preds[name]=model.predict([query(s) for s in testing])
            metrics.append({'fold':fold,'model':name,**prediction_metrics(testing,test_preds[name])})
        physical=[n for n in ['Fixed prior','SAC-Gen-Set','SAC-Gen-GB'] if n in models]
        weights,weight_status=protected_weights(weight_samples,{n:weight_preds[n] for n in physical},calibration_samples+testing)
        fused_cal=fuse({n:cal_preds[n] for n in physical},weights)
        fused_test=fuse({n:test_preds[n] for n in physical},weights)
        calibrator=Calibration().fit(calibration_samples,fused_cal,training+weight_samples,testing,alpha)
        output=calibrator.apply(fused_test)
        coverage=[];widths=[]
        for s,p in zip(testing,output):
            actual,_=outcomes(s)
            for assay in p['icp_predictions']:
                e=assay['element'];lo=assay.get('calibrated_lower_wt_percent');hi=assay.get('calibrated_upper_wt_percent')
                covered=lo<=actual[e]<=hi if lo is not None and e in actual else None
                if covered is not None:coverage.append((identity(s),covered))
                if lo is not None:widths.append(hi-lo)
                point_rows.append({'fold':fold,'record_key':identity(s),'sample_id':s['sample_id'],'element':e,
                                   'observed_wt_percent':actual.get(e),'predicted_wt_percent':assay['predicted_mean_w_wt_percent'],
                                   'lower_wt_percent':lo,'upper_wt_percent':hi,'covered':covered})
        by_specimen={}
        for ident,covered in coverage:by_specimen.setdefault(ident,[]).append(covered)
        metrics.append({'fold':fold,'model':'Calibrated physical ensemble',**prediction_metrics(testing,output),
                        'component_coverage':float(np.mean([v for _,v in coverage])) if coverage else None,
                        'mean_interval_width_wt_percent':float(np.mean(widths)) if widths else None,
                        'simultaneous_specimen_coverage':float(np.mean([all(v) for v in by_specimen.values()])) if by_specimen else None})
        details.append({'fold':fold,'fit_keys':[identity(s) for s in training],'weight_keys':[identity(s) for s in weight_samples],
                        'calibration_keys':[identity(s) for s in calibration_samples],'test_keys':[identity(s) for s in testing],
                        'fit_recipe_keys':sorted(sets[0]),'weight_recipe_keys':sorted(sets[1]),
                        'calibration_recipe_keys':sorted(sets[2]),'test_recipe_keys':sorted(sets[3]),
                        'ensemble_weights':weights,'weight_status':weight_status,'calibration':calibrator.info})
    return {'source_fingerprint':digest(samples),'origin':origin,'metrics':metrics,'folds':details,'predictions':point_rows,
            'settings':{'epochs':epochs,'folds':folds,'alpha':alpha,'seed':seed,'include_sac':include_sac},
            'note':'All preprocessing, model fitting, weight fitting and interval calibration use separate recipe pools. No deployment checkpoint is used. These are development results, not independent prospective gains.'}


def replay(samples,origin,budget=36,initial=16,step=10,epochs=30,seed=42,policies=None,progress=None):
    from scipy.spatial.distance import cdist
    from ..loading import controls
    from ..data_only import DataOnlyModel
    from ..science_only import SciencePriorModel
    from ..weighted_validation import collect
    pool=usable(samples,origin);groups=np.array([recipe_id(s) for s in pool])
    if len(set(groups))<20:raise ValueError('Replay needs at least twenty distinct recipes.')
    train,test=next(GroupShuffleSplit(test_size=.2,random_state=seed).split(pool,groups=groups))
    held=[pool[i] for i in test]
    # One representative per recipe, selected without reading its result.
    candidates={}
    for i in train:candidates.setdefault(groups[i],pool[i])
    pool=list(candidates.values())
    if not 12<=initial<=budget<=len(pool) or step<1:raise ValueError('Use 12 <= initial <= budget <= candidate recipes.')
    policies=policies or ['Random','Space filling','GP-BO','Model disagreement']
    if set(policies)-{'Random','Space filling','GP-BO','Model disagreement'}:raise ValueError('Unsupported acquisition policy.')
    features=np.array([controls(query(s),'chemistry_ion',s['elements'][0]) for s in pool])
    features=(features-features.mean(0))/np.maximum(features.std(0),1e-6)
    start=np.random.default_rng(seed).choice(len(pool),initial,replace=False).tolist()
    curves=[];traces=[]
    for policy in policies:
        selected=start.copy();rng=np.random.default_rng(seed)
        while True:
            revealed=[pool[i] for i in selected]
            if progress:progress({'stage':f'{policy}: {len(selected)} revealed recipes'})
            model=SetResidual(epochs=epochs,seed=seed).fit(revealed,origin)
            forecasts=model.predict([query(s) for s in held])
            curves.append({'policy':policy,'observations':len(selected),**prediction_metrics(held,forecasts),
                           'observed_elements':len({e for s in revealed for e in s['elements']})})
            if len(selected)>=budget:break
            remaining=[i for i in range(len(pool)) if i not in selected]
            distances=cdist(features[remaining],features[selected]).min(1)
            if policy=='Random':score=rng.random(len(remaining))
            elif policy=='Space filling':score=distances
            elif policy=='GP-BO':
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import Matern,WhiteKernel
                labeled=[i for i in selected if outcomes(pool[i])[1] is not None]
                if labeled:
                    gp=GaussianProcessRegressor(Matern(nu=2.5)+WhiteKernel(.05),optimizer=None,normalize_y=True,alpha=1e-5)
                    gp.fit(features[labeled],[outcomes(pool[i])[1] for i in labeled])
                    mean,sd=gp.predict(features[remaining],return_std=True)
                    score=mean+2*sd
                else:score=distances
            else:
                queries=[query(pool[i]) for i in remaining]
                outputs={'Set':model.predict(queries),'Prior':SciencePriorModel(origin).predict(queries)}
                score=np.array([r['ICP_model_SD_wt_percent'] or 0. for r in disagreement(queries,outputs)])
            chosen=[]
            for _ in range(min(step,budget-len(selected))):
                diversity=cdist(features[remaining],features[chosen]).min(1) if chosen else np.zeros(len(remaining))
                utility=score+.1*diversity/(1+diversity)
                utility[np.isin(remaining,chosen)]=-np.inf
                chosen.append(remaining[int(np.argmax(utility))])
            traces.append({'policy':policy,'before':len(selected),'revealed_keys':[identity(pool[i]) for i in selected],
                           'chosen_keys':[identity(pool[i]) for i in chosen]})
            selected+=chosen
    return {'origin':origin,'source_fingerprint':digest(samples),'curves':curves,'queries':traces,
            'held_out_keys':[identity(s) for s in held],'initial_keys':[identity(pool[i]) for i in start],
            'settings':{'seed':seed,'budget':budget,'initial':initial,'step':step,'epochs':epochs},
            'note':'Recorded-pool replay only; equal initial recipes and fixed held-out groups. GP-BO uses GPR upper confidence bounds on observed binary structural labels, not a calibrated classification probability. Curves do not establish prospective laboratory discovery efficiency.'}
