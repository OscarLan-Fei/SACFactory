"""Training-only nested head selection, optional calibration and decision threshold."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def masks(y,groups,seed):
    counts=np.bincount(np.asarray(y,dtype=int),minlength=2)
    count=min(3,int(counts.min()),len(set(groups)))
    if count<2:return []
    return list(StratifiedGroupKFold(n_splits=count,shuffle=True,random_state=int(seed)).split(np.zeros(len(y)),y,groups))


def estimator(c,seed):
    return make_pipeline(StandardScaler(),LogisticRegression(C=float(c),max_iter=2000,random_state=int(seed)))


def choose_c(x,y,groups,seed):
    partitions=masks(y,groups,seed);scores=[]
    for c in (.03,.1,.3,1.):
        losses=[]
        for train,held in partitions:
            if len(set(y[train]))<2:continue
            model=estimator(c,seed).fit(x[train],y[train])
            losses.extend((model.predict_proba(x[held])[:,1]-y[held])**2)
        scores.append({'C':c,'Brier':float(np.mean(losses)) if losses else None})
    valid=[row for row in scores if row['Brier'] is not None]
    return (min(valid,key=lambda row:row['Brier'])['C'] if valid else .1),scores


def fit_head(x,y,groups,seed):
    """All arrays belong to the caller's training pool; outer outcomes are absent."""
    x=np.asarray(x);y=np.asarray(y,dtype=int);groups=np.asarray(groups,dtype=str)
    oof=np.full(len(y),np.nan);trace=[]
    for number,(train,held) in enumerate(masks(y,groups,seed),1):
        assert not set(groups[train])&set(groups[held])
        c,scores=choose_c(x[train],y[train],groups[train],seed+number)
        if len(set(y[train]))<2:p=np.full(len(held),float(y[train].mean()))
        else:p=estimator(c,seed).fit(x[train],y[train]).predict_proba(x[held])[:,1]
        oof[held]=p
        trace.append({'fold':number,'train_indices':train.tolist(),'held_indices':held.tolist(),'selected_C':c,'selection':scores})
    c,scores=choose_c(x,y,groups,seed)
    fitted=estimator(c,seed).fit(x,y);scaler=fitted.steps[0][1];classifier=fitted.steps[1][1]
    head={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
        'coefficient':classifier.coef_[0].tolist(),'intercept':float(classifier.intercept_[0]),'pass_threshold':.5}
    info={'C':float(c),'head_selection':{'candidates':scores,'nested_head_folds':trace,
        'calibration_scope':'Cross-fitted head scores conditional on training-pool generated ICP features; complete pipeline assessed by outer masks'}}
    valid=np.isfinite(oof)
    if not valid.all() or min(np.bincount(y,minlength=2))<6:
        info['refinement_status']='Insufficient head cross-fit coverage or fewer than six observations in one class.'
        return head,info
    logits=logit(np.clip(oof,1e-6,1-1e-6))[:,None]
    identity_loss=float(np.mean((oof-y)**2));choices=[{'method':'identity','Brier':identity_loss}]
    calibration_oof={}
    for strength in (.1,1.,10.):
        result=np.full(len(y),np.nan)
        for train,held in masks(y,groups,seed+101):
            if len(set(y[train]))<2:continue
            calibrator=LogisticRegression(C=strength,max_iter=2000).fit(logits[train],y[train])
            if calibrator.coef_[0,0]<=0:continue
            result[held]=calibrator.predict_proba(logits[held])[:,1]
        if np.isfinite(result).all():
            choices.append({'method':'sigmoid','C':strength,'Brier':float(np.mean((result-y)**2))})
            calibration_oof[strength]=result
    best=min(choices,key=lambda row:row['Brier']);probabilities=oof
    if best['method']=='sigmoid' and best['Brier']<identity_loss-.0001:
        calibrator=LogisticRegression(C=best['C'],max_iter=2000).fit(logits,y)
        if calibrator.coef_[0,0]>0:
            head['calibration']={'method':'sigmoid','coefficient':float(calibrator.coef_[0,0]),
                'intercept':float(calibrator.intercept_[0]),'C':float(best['C'])}
            probabilities=calibration_oof[best['C']]
    thresholds=np.unique(np.r_[np.linspace(.05,.95,91),.5])
    results=[(float(balanced_accuracy_score(y,probabilities>=threshold)),float(threshold)) for threshold in thresholds]
    value,threshold=max(results,key=lambda row:(row[0],-abs(row[1]-.5)))
    head['pass_threshold']=threshold
    info.update(refinement_status='TRAINING_ONLY_CROSS_FITTED_HEAD_REFINEMENT',pass_threshold=threshold,
        probability_status='CROSS_FITTED_SIGMOID_CALIBRATION; outer reliability still requires evaluation' if 'calibration' in head else 'UNCALIBRATED; identity selected by training-only calibration comparison',
        calibration_selection=choices,training_OOF_Brier=identity_loss,
        training_OOF_balanced_accuracy=value,training_OOF_fail_recall=float(np.mean(probabilities[y==0]<threshold)),
        training_OOF_pass_recall=float(np.mean(probabilities[y==1]>=threshold)))
    return head,info
