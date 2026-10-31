"""Matched outer-fold baselines for continuous ICP and XRD probability."""
import copy
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor,GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern,WhiteKernel
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,roc_auc_score
from threadpoolctl import threadpool_limits

from .weighted_validation import collect,split_records,element_features,ROTATING_CV,PROTOCOLS
from .validation_groups import RANDOM_RECIPE_CV, assert_mask
from .compute_runtime import jobs,emit
from .planning import fingerprint
from .prediction_agreement import component_score

MODELS=['Mean / prevalence','Ridge / logistic','Random forest','Gradient boosting','Gaussian process',
    'Data-only Gaussian process','Data-only Random forest','Data-only Gradient boosting',
    'Scientific descriptors Gradient boosting','Science prior (fixed)','SAC-Gen',
    'SAC-Gen-Diffusion','SAC-Gen-GB']
VERSION='matched-icp-xrd-baselines-v1'
PUBLIC_MODELS=['Data-only Gaussian process','Data-only Random forest','Data-only Gradient boosting',
    'Scientific descriptors Gradient boosting','Science prior (fixed)','SAC-Gen-Diffusion','SAC-Gen-GB']


def estimators(name,dimension,seed):
    if name=='Mean / prevalence':return DummyRegressor(),None
    if name=='Ridge / logistic':return make_pipeline(StandardScaler(),Ridge(alpha=10.)),make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=seed))
    if name=='Random forest':return (RandomForestRegressor(n_estimators=120,min_samples_leaf=3,random_state=seed,n_jobs=jobs()),
        RandomForestClassifier(n_estimators=120,min_samples_leaf=3,random_state=seed,n_jobs=jobs()))
    if name=='Gradient boosting':return (HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,min_samples_leaf=8,l2_regularization=1.,random_state=seed),
        HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=8,l2_regularization=1.,random_state=seed))
    if name=='Gaussian process':
        kernel=Matern(length_scale=np.sqrt(dimension),nu=2.5)
        return (make_pipeline(StandardScaler(),GaussianProcessRegressor(kernel=kernel+WhiteKernel(.05),optimizer=None,normalize_y=True,alpha=1e-5,random_state=seed)),
                make_pipeline(StandardScaler(),GaussianProcessClassifier(kernel=kernel,optimizer=None,max_iter_predict=50,random_state=seed)))
    raise ValueError('Select a baseline estimator.')


def fitted_baseline(train,test,model,profile,seed=42):
    if model.startswith(('Data-only ','Scientific descriptors ')) or model=='Science prior (fixed)':
        if model.startswith('Scientific descriptors '):
            from .scientific_descriptors_model import ScientificDescriptorModel
            fitted=ScientificDescriptorModel(model.removeprefix('Scientific descriptors '),seed).fit_records(train,train[0]['sample']['origin'])
            detail={**fitted.info,'model_id':fitted.model_id}
        elif model.startswith('Data-only '):
            from .data_only import DataOnlyModel
            fitted=DataOnlyModel(model.removeprefix('Data-only '),seed).fit_records(train,train[0]['sample']['origin'])
            detail={**fitted.info,'model_id':fitted.model_id}
        else:
            from .science_only import SciencePriorModel
            fitted=SciencePriorModel(train[0]['sample']['origin'])
            detail={'model_id':fitted.model_id,'training_ids':[],'fitted_to_current_experiments':False,
                'input_profile':'Fixed mass/thermal/mixing/kinetic priors; empirical anchors disclosed'}
        predictions=fitted.predict([r['sample'] for r in test]);values=[]
        for r,p in zip(test,predictions):
            members={v['element']:v['predicted_mean_w_wt_percent'] for v in p['icp_predictions']}
            symbols=r['sample']['elements'];actual=sum(r['assays'][e] for e in symbols) if all(e in r['assays'] for e in symbols) else None
            values.extend({'sample_id':r['sample']['sample_id'],'batch_id':r['sample']['batch_id'],'element':e,
                'observed_w_wt_percent':r['assays'].get(e),'predicted_w_wt_percent':members.get(e),
                'composition':'+'.join(symbols),'observed_total_w_wt_percent':actual,
                'predicted_total_w_wt_percent':p['predicted_total_wt_percent'],
                'xrd_pass_observed':r['outcomes']['xrd_pass'],'xrd_pass_predicted':p['p_xrd_pass'],
                'xrd_pass_threshold':p['xrd_pass_threshold'],'predicted_xrd_state':p['predicted_xrd_state']} for e in symbols)
        return values,detail
    from . import pulseicp_bridge as bridge
    # Match the neural loading learner's eligible quantitative training targets.
    prepared=bridge.prepare([r['sample'] for r in train],train[0]['sample']['origin'])
    eligible={(r['raw']['sample_id'],r['recipe'].element) for r in prepared.records}
    quantitative=[(r,e,w) for r in train for e,w in r['assays'].items() if (r['sample']['sample_id'],e) in eligible]
    xr=[r for r in train if r['outcomes']['xrd_pass'] is not None]
    y=np.asarray([r['outcomes']['xrd_pass'] for r in xr])
    probability=[None]*len(test);reg=None
    regressor,classifier=estimators(model,len(train[0]['x']),seed)
    if len(quantitative)>=8:
        xx=np.asarray([element_features(r['sample'],e,r['x'],profile) for r,e,_ in quantitative])
        # GPR kernel dimensional scale follows the regression feature schema.
        if model=='Gaussian process':regressor=estimators(model,xx.shape[1],seed)[0]
        with threadpool_limits(limits=jobs()):reg=regressor.fit(xx,[w for _,_,w in quantitative])
    if len(y):
        if model=='Mean / prevalence' or len(set(y))<2:probability=[float(y.mean())]*len(test)
        elif len(y)>=12:
            with threadpool_limits(limits=jobs()):
                classifier.fit(np.asarray([r['x'] for r in xr]),y)
                probability=classifier.predict_proba(np.asarray([r['x'] for r in test]))[:,list(classifier.classes_).index(1)].tolist()
    targets=[(r,e) for r in test for e in r['sample']['elements']]
    if reg is not None and targets:
        with threadpool_limits(limits=jobs()):
            predictions=reg.predict(np.asarray([element_features(r['sample'],e,r['x'],profile) for r,e in targets]))
        predicted={(r['sample']['sample_id'],e):float(p) for (r,e),p in zip(targets,predictions)}
    else:predicted={}
    values=[]
    for i,r in enumerate(test):
        symbols=r['sample']['elements']
        actual_total=sum(r['assays'][e] for e in symbols) if all(e in r['assays'] for e in symbols) else None
        member_values=[predicted.get((r['sample']['sample_id'],e)) for e in symbols]
        predicted_total=sum(member_values) if all(v is not None for v in member_values) else None
        for e in r['sample']['elements']:
            pred=predicted.get((r['sample']['sample_id'],e))
            values.append({'sample_id':r['sample']['sample_id'],'batch_id':r['sample']['batch_id'],'element':e,
                'observed_w_wt_percent':r['assays'].get(e),'predicted_w_wt_percent':pred,
                'composition':'+'.join(symbols),'observed_total_w_wt_percent':actual_total,
                'predicted_total_w_wt_percent':predicted_total,
                'xrd_pass_observed':r['outcomes']['xrd_pass'],'xrd_pass_predicted':probability[i]})
    return values,{'training_sample_ids':[r['sample']['sample_id'] for r in train],
        'ICP_training_pairs':len(quantitative),'XRD_training_n':len(y),'XRD_classes':sorted(set(map(int,y))),
        'probability_status':'UNCALIBRATED','regression_bounds':'Unclipped statistical predictions'}


def metrics(frame,scale=.03,weight=.6):
    numeric=frame.dropna(subset=['observed_w_wt_percent','predicted_w_wt_percent'])
    binary=frame.drop_duplicates(['fold','sample_id']).dropna(subset=['xrd_pass_observed','xrd_pass_predicted'])
    result={'ICP_n':len(numeric),'ICP_MAE':None,'ICP_RMSE':None,'ICP_R2':None,
            'XRD_n':len(binary),'XRD_Brier':None,'XRD_ROC_AUC':None,'accuracy_n':0,'mean_agreement':None}
    if len(numeric):
        # Each physical specimen has equal mass in pooled metrics.
        counts=numeric.groupby(['fold','sample_id']).sample_id.transform('size')
        weights=1/counts
        y=numeric.observed_w_wt_percent;p=numeric.predicted_w_wt_percent
        result.update(ICP_MAE=float(mean_absolute_error(y,p,sample_weight=weights)),
            ICP_RMSE=float(np.sqrt(mean_squared_error(y,p,sample_weight=weights))),
            ICP_R2=float(r2_score(y,p,sample_weight=weights)) if len(y)>1 and np.ptp(y)>1e-12 else None)
    if len(binary):
        y=binary.xrd_pass_observed.to_numpy();p=binary.xrd_pass_predicted.to_numpy()
        result.update(XRD_Brier=float(np.mean((y-p)**2)),XRD_ROC_AUC=float(roc_auc_score(y,p)) if len(set(y))==2 else None)
        from sklearn.metrics import balanced_accuracy_score
        threshold=binary.xrd_pass_threshold.fillna(.5).to_numpy() if 'xrd_pass_threshold' in binary else np.full(len(binary),.5)
        result.update(XRD_balanced_accuracy_selected_threshold=float(balanced_accuracy_score(y,p>=threshold)) if len(set(y))==2 else None,
            XRD_FAIL_recall_selected_threshold=float(np.mean(p[y==0]<threshold[y==0])) if (y==0).any() else None,
            XRD_balanced_accuracy_0_5=float(balanced_accuracy_score(y,p>=.5)) if len(set(y))==2 else None)
    scores=[]
    for _,part in frame.groupby(['fold','sample_id']):
        if part[['observed_w_wt_percent','predicted_w_wt_percent']].notna().all().all() and pd.notna(part.iloc[0].xrd_pass_predicted) and pd.notna(part.iloc[0].xrd_pass_observed):
            icp=float(np.mean(((part.predicted_w_wt_percent-part.observed_w_wt_percent)/scale)**2))
            xrd=float((part.iloc[0].xrd_pass_predicted-part.iloc[0].xrd_pass_observed)**2)
            scores.append(weight*component_score('icp',icp)+(1-weight)*component_score('xrd_pass',xrd))
    result.update(accuracy_n=len(scores),mean_agreement=float(np.mean(scores)) if scores else None)
    return result


def compare(samples,methods,origin,targets,models=None,profile='chemistry_pulseicp',batches_per_fold=1,
            seed=42,sac_settings=None,sac_family='Diffusion Transformer',reference=None,
            protocol=ROTATING_CV,max_folds=5,scale=.03,agreement_weight=.6):
    models=list(dict.fromkeys(MODELS if models is None else models))
    if set(models)-set(MODELS) or not models:raise ValueError('Choose comparison models.')
    if protocol not in PROTOCOLS:raise ValueError('Choose a supported validation protocol.')
    if not np.isfinite(scale) or scale<=0 or not np.isfinite(agreement_weight) or not 0<=agreement_weight<=1:
        raise ValueError('Agreement scale must be positive and its ICP weight must lie in [0,1].')
    records,audit=collect(samples,methods,origin,profile)
    splits=split_records(records,samples,origin,targets,protocol,max_folds,batches_per_fold,seed)
    rows=[];partitions=[]
    if reference is not None:
        _,info=reference
        if (info['dataset_fingerprint']!=fingerprint(samples) or info.get('methods_fingerprint')!=fingerprint(methods) or info['profile']!=profile or
            info['protocol']!=protocol or info['batches_per_test_fold']!=batches_per_fold or
            set(info['selected_elements'])!=set(targets) or info['icp_family']!=sac_family):
            raise ValueError('Saved SAC-Gen predictions do not match the comparison data/settings.')
    for index,(train,test,fold,fold_targets) in enumerate(splits,1):
        emit({'stage':'Matched model comparison','fold':index,'folds':len(splits)})
        assert_mask([r['sample'] for r in train], [r['sample'] for r in test], protocol == RANDOM_RECIPE_CV)
        partition={'fold':fold,'train_ids':[r['sample']['sample_id'] for r in train],
            'test_ids':[r['sample']['sample_id'] for r in test],
            'train_batches':sorted({r['sample']['batch_id'] for r in train}),
            'test_batches':sorted({r['sample']['batch_id'] for r in test}),'models':{}}
        if len(train)<8 or len(partition['train_batches'])<2 or len({r['recipe'] for r in train})<3:
            partition['models']={model:{'unavailable_reason':'Need 8 training specimens, 2 training batches and 3 recipes.'} for model in models}
            partitions.append(partition)
            continue
        for model in models:
            emit({'stage':f'Fitting {model}','fold':index,'folds':len(splits)})
            try:
                if model=='SAC-Gen' and reference is not None:
                    frame,info=reference
                    old=next(p for p in info['folds'] if p['fold']==fold)
                    if set(old['train_sample_ids'])!=set(partition['train_ids']) or set(old['test_sample_ids'])!=set(partition['test_ids']):
                        raise ValueError('Saved SAC-Gen fold membership differs.')
                    values=frame.loc[frame.fold==fold].to_dict('records')
                    detail={'reused_matching_OOF_predictions':True,'family':sac_family,'source_model_id':old['pulseicp'].get('model_id')}
                elif model in {'SAC-Gen','SAC-Gen-Diffusion','SAC-Gen-GB'}:
                    from .sac_gen import SACGen
                    from .pulseicp_bridge import configuration
                    settings = dict(sac_settings or configuration()['model'])
                    settings['crossfit_grouping'] = 'recipe' if protocol == RANDOM_RECIPE_CV else 'batch'
                    selected_family={'SAC-Gen-Diffusion':'Diffusion Transformer',
                        'SAC-Gen-GB':'Gradient boosting residual'}.get(model,sac_family)
                    fitted=SACGen(selected_family,settings,profile).fit([r['sample'] for r in train],origin,methods)
                    prediction=fitted.predict([r['sample'] for r in test]);values=[]
                    for r,p in zip(test,prediction):
                        measured={v['element']:v['predicted_mean_w_wt_percent'] for v in p['icp_predictions']}
                        values.extend({'sample_id':r['sample']['sample_id'],'batch_id':r['sample']['batch_id'],'element':e,
                            'observed_w_wt_percent':r['assays'].get(e),'predicted_w_wt_percent':measured.get(e),
                            'composition':'+'.join(r['sample']['elements']),
                            'observed_total_w_wt_percent':sum(r['assays'][v] for v in r['sample']['elements']) if all(v in r['assays'] for v in r['sample']['elements']) else None,
                            'predicted_total_w_wt_percent':p['predicted_total_wt_percent'],
                            'xrd_pass_observed':r['outcomes']['xrd_pass'],'xrd_pass_predicted':p['p_xrd_pass'],
                            'xrd_pass_threshold':p.get('xrd_pass_threshold',.5),'predicted_xrd_state':p.get('predicted_xrd_state')} for e in r['sample']['elements'])
                    detail={'model_id':fitted.model_id,'family':selected_family,'settings':fitted.settings,'XRD':fitted.xrd_info}
                else:values,detail=fitted_baseline(train,test,model,profile,seed)
                rows.extend({**v,'model':model,'fold':fold} for v in values if v['element'] in fold_targets)
                partition['models'][model]=detail
            except (ValueError,KeyError,TypeError,RuntimeError) as exc:
                partition['models'][model]={'unavailable_reason':str(exc)}
        partitions.append(partition)
    columns=['sample_id','batch_id','element','observed_w_wt_percent','predicted_w_wt_percent','xrd_pass_observed','xrd_pass_predicted','model','fold']
    points=pd.DataFrame(rows)
    if not len(points):points=pd.DataFrame(columns=columns)
    summaries=[]
    icp_sets=[];xrd_sets=[]
    for model in models:
        part=points.loc[points.model==model]
        icp_sets.append(set(zip(part.loc[part[['observed_w_wt_percent','predicted_w_wt_percent']].notna().all(axis=1),'fold'],
            part.loc[part[['observed_w_wt_percent','predicted_w_wt_percent']].notna().all(axis=1),'sample_id'],
            part.loc[part[['observed_w_wt_percent','predicted_w_wt_percent']].notna().all(axis=1),'element'])))
        valid=part.dropna(subset=['xrd_pass_observed','xrd_pass_predicted'])
        xrd_sets.append(set(zip(valid.fold,valid.sample_id)))
    common_icp=set.intersection(*icp_sets) if icp_sets else set();common_xrd=set.intersection(*xrd_sets) if xrd_sets else set()
    for model in models:
        part=points.loc[points.model==model].copy()
        all_metrics=metrics(part,scale,agreement_weight)
        for i,r in part.iterrows():
            if (r.fold,r.sample_id,r.element) not in common_icp:part.loc[i,['observed_w_wt_percent','predicted_w_wt_percent']]=np.nan
            if (r.fold,r.sample_id) not in common_xrd:part.loc[i,['xrd_pass_observed','xrd_pass_predicted']]=np.nan
        summaries.append({'model':model,**metrics(part,scale,agreement_weight),'available_ICP_n':all_metrics['ICP_n'],'available_XRD_n':all_metrics['XRD_n']})
    return points,pd.DataFrame(summaries),{'version':VERSION,'origin':origin,'profile':profile,
        'dataset_fingerprint':fingerprint(samples),'partitions':partitions,'seed':seed,'models':models,
        'batches_per_fold':batches_per_fold,'protocol':protocol,'max_folds':max_folds,'comparison_rows':'Common eligible endpoint intersection across all selected models',
        'regression_weighting':'Equal specimen, equal selected member within specimen','accuracy_scale_wt_percent':scale,'agreement_icp_weight':agreement_weight,
        'audit':audit.astype(object).where(pd.notna(audit),None).to_dict('records'),
        'note':'Predictive model comparison, not an acquisition-policy comparison. Models share declared outer masks, quantitative eligibility and pre-synthesis descriptor profile. Random recipe mode permits batch overlap but excludes identical recipes and specimens from the opposing pool. SAC-Gen adds cross-fitted loading summaries to its XRD head. Baseline hyperparameters are fixed, not tuned on outer tests. Kernel optimizers are disabled. These are development cross-validation estimates, not independent confirmation of superiority.'}
