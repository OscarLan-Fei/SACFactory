"""Experimental-control baselines with no sourced chemistry or physical priors."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit,logit
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .domain import ELEMENTS,recipe_key
from .planning import fingerprint
from .compute_runtime import jobs

BACKENDS=['Gaussian process','Random forest','Gradient boosting']
VERSION='experimental-controls-only-v1'


def features(sample,element=None):
    """Identity is nominal; array positions have no numerical periodic meaning."""
    concentrations=sample['concentrations_mM']
    presence=[float(e in sample['elements']) for e in ELEMENTS]
    amounts=[np.log1p(float(concentrations[e])) if e in sample['elements'] else 0. for e in ELEMENTS]
    controls=[np.log1p(sum(float(concentrations[e]) for e in sample['elements'])),len(sample['elements']),
        float(sample['pulse_limit_C']),np.log1p(float(sample['pulse_count'])),
        np.log(float(sample['pan_mass_g'])),np.log(float(sample['solvent_volume_mL']))]
    value=presence+amounts+controls
    if element is not None:value += [float(e==element) for e in ELEMENTS]+[np.log1p(float(concentrations[element]))]
    result=np.asarray(value,dtype=float)
    if not np.isfinite(result).all():raise ValueError('Data-only models require finite recorded process controls.')
    return result


def classifier(backend,seed,dimension):
    if backend=='Gaussian process':model=GaussianProcessClassifier(Matern(length_scale=np.sqrt(dimension),nu=2.5),optimizer=None,max_iter_predict=40,random_state=seed)
    elif backend=='Random forest':model=RandomForestClassifier(n_estimators=120,min_samples_leaf=3,class_weight='balanced',random_state=seed,n_jobs=jobs())
    else:model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=8,l2_regularization=1.,random_state=seed)
    return make_pipeline(VarianceThreshold(),StandardScaler(),model)


def calibrate(y,probabilities,groups,seed):
    from .xrd_refinement import masks
    valid=np.isfinite(probabilities);calibration=None;chosen=probabilities.copy()
    if valid.all() and min(np.bincount(y.astype(int),minlength=2))>=6:
        x=logit(np.clip(probabilities,1e-6,1-1e-6))[:,None];out=np.full(len(y),np.nan)
        for train,held in masks(y,groups,seed):
            if len(set(y[train]))<2:continue
            model=LogisticRegression(C=1.,max_iter=2000).fit(x[train],y[train])
            if model.coef_[0,0]<=0:continue
            out[held]=model.predict_proba(x[held])[:,1]
        if np.isfinite(out).all() and np.mean((out-y)**2)<np.mean((probabilities-y)**2)-.0001:
            model=LogisticRegression(C=1.,max_iter=2000).fit(x,y)
            if model.coef_[0,0]>0:
                calibration={'coefficient':float(model.coef_[0,0]),'intercept':float(model.intercept_[0])};chosen=out
    threshold=.5
    if valid.all():
        choices=[(float(balanced_accuracy_score(y,chosen>=t)),float(t)) for t in np.linspace(.05,.95,91)]
        _,threshold=max(choices,key=lambda v:(v[0],-abs(v[1]-.5)))
    return calibration,threshold


class DataOnlyModel:
    def __init__(self,backend='Random forest',seed=42):
        if backend not in BACKENDS:raise ValueError('Select a supported experimental-only backend.')
        self.backend=backend;self.seed=int(seed);self.regressor=None;self.classifier=None

    def encode(self,sample,element=None):
        return features(sample,element)

    def fit_records(self,records,origin):
        """No measured target is excluded using a physical feed/matrix bound."""
        self.origin=origin;self.training_ids=[r['sample']['sample_id'] for r in records]
        self.training_fingerprint=fingerprint([r['sample'] for r in records]);self.compositions={'+'.join(r['sample']['elements']) for r in records}
        quantitative=[(r,e,w) for r in records for e,w in r['assays'].items()]
        if len(quantitative)<8:raise ValueError('Need eight recorded quantitative sample-element pairs.')
        x=np.asarray([self.encode(r['sample'],e) for r,e,_ in quantitative]);y=np.array([w for _,_,w in quantitative])
        weight=np.array([1/len(r['sample']['elements']) for r,_,_ in quantitative])
        self.target_elements={e for _,e,_ in quantitative};self.minimum=x.min(axis=0);self.maximum=x.max(axis=0)
        if self.backend=='Gaussian process':
            model=GaussianProcessRegressor(Matern(length_scale=np.sqrt(x.shape[1]),nu=2.5)+WhiteKernel(.05),
                optimizer=None,normalize_y=True,alpha=.005/weight,random_state=self.seed)
            self.regressor=make_pipeline(VarianceThreshold(),StandardScaler(),model)
            with threadpool_limits(limits=jobs()):self.regressor.fit(x,y)
        else:
            model=(RandomForestRegressor(n_estimators=120,min_samples_leaf=3,max_depth=18,n_jobs=jobs(),random_state=self.seed)
                if self.backend=='Random forest' else HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,min_samples_leaf=8,l2_regularization=1.,random_state=self.seed))
            self.regressor=make_pipeline(VarianceThreshold(),StandardScaler(),model)
            with threadpool_limits(limits=jobs()):self.regressor.fit(x,y,**{self.regressor.steps[-1][0]+'__sample_weight':weight})
        xr=[r for r in records if r['outcomes']['xrd_pass'] is not None]
        yy=np.asarray([r['outcomes']['xrd_pass'] for r in xr],dtype=int);self.xrd_mean=float(yy.mean()) if len(yy) else None
        self.calibration=None;self.threshold=.5;self.head_info={'n':len(yy),'pass_n':int(yy.sum()),'fail_n':int(len(yy)-yy.sum())}
        if len(set(yy))==2:
            xx=np.asarray([self.encode(r['sample']) for r in xr]);groups=np.asarray([repr(recipe_key(r['sample'])) for r in xr])
            from .xrd_refinement import masks
            oof=np.full(len(yy),np.nan);folds=[]
            with threadpool_limits(limits=jobs()):
                for train,held in masks(yy,groups,self.seed):
                    if len(set(yy[train]))<2:continue
                    model=classifier(self.backend,self.seed,xx.shape[1]).fit(xx[train],yy[train])
                    oof[held]=model.predict_proba(xx[held])[:,1]
                    folds.append({'train_ids':[xr[i]['sample']['sample_id'] for i in train],
                        'held_ids':[xr[i]['sample']['sample_id'] for i in held]})
                self.calibration,self.threshold=calibrate(yy,oof,groups,self.seed+100)
                self.classifier=classifier(self.backend,self.seed,xx.shape[1]).fit(xx,yy)
            self.head_info.update(folds=folds,calibration=self.calibration,pass_threshold=self.threshold)
        self.info={'version':VERSION,'backend':self.backend,'training_ids':self.training_ids,
            'ICP_training_pairs':len(quantitative),'XRD':self.head_info,'input_profile':'Recorded controls and nominal element identity only',
            'excluded_information':['elemental properties','oxidation/radius references','thermal references','retention prior','mixing H/S/G','mass conservation decoder'],
            'regression_bounds':'Unconstrained statistical outputs; negative predictions are retained and flagged',
            'eligibility':'All valid quantified experimental targets; no science-derived mass-bound exclusion'}
        return self

    @property
    def model_id(self):
        return fingerprint({'version':VERSION,'backend':self.backend,'seed':self.seed,'training':self.training_fingerprint,
            'calibration':self.calibration,'threshold':self.threshold,
            'source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def predict(self,samples):
        if any(s.get('origin',self.origin)!=self.origin for s in samples):raise ValueError('Do not mix REAL and DEMO data spaces.')
        pairs=[(i,e) for i,s in enumerate(samples) for e in s['elements']]
        x=np.asarray([self.encode(samples[i],e) for i,e in pairs])
        with threadpool_limits(limits=jobs()):values=self.regressor.predict(x)
        xx=np.asarray([self.encode(s) for s in samples])
        with threadpool_limits(limits=jobs()):p=self.classifier.predict_proba(xx)[:,1] if self.classifier is not None else np.full(len(samples),np.nan if self.xrd_mean is None else self.xrd_mean)
        if self.calibration:p=expit(self.calibration['coefficient']*logit(np.clip(p,1e-6,1-1e-6))+self.calibration['intercept'])
        result=[]
        for i,s in enumerate(samples):
            members=[{'element':e,'predicted_mean_w_wt_percent':float(values[k]),
                'negative_statistical_prediction':bool(values[k]<0)} for k,(j,e) in enumerate(pairs) if j==i]
            outside=any(e not in self.target_elements for e in s['elements']) or '+'.join(s['elements']) not in self.compositions
            outside=outside or any(np.any((x[k]<self.minimum-1e-10)|(x[k]>self.maximum+1e-10)) for k,(j,_) in enumerate(pairs) if j==i)
            result.append({'sample_id':s.get('sample_id'),'model_name':'Data-only '+self.backend,'model_id':self.model_id,
                'icp_predictions':members,'predicted_total_wt_percent':sum(v['predicted_mean_w_wt_percent'] for v in members),
                'p_xrd_pass':float(p[i]) if np.isfinite(p[i]) else None,'p_xrd_fail':1-float(p[i]) if np.isfinite(p[i]) else None,
                'xrd_pass_threshold':self.threshold,'predicted_xrd_state':('PASS' if p[i]>=self.threshold else 'FAIL') if np.isfinite(p[i]) else None,
                'outside_training_support':bool(outside),'xrd_probability_status':'TRAINING_ONLY_CALIBRATION' if self.calibration else 'UNCALIBRATED'})
        return result

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        joblib.dump(self,directory/'model.joblib')
        payload={'model_id':self.model_id,'origin':self.origin,'info':self.info,
            'weights_sha256':hashlib.sha256((directory/'model.joblib').read_bytes()).hexdigest()}
        (directory/'metadata.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')

    @classmethod
    def load(cls,directory):
        directory=Path(directory);meta=json.loads((directory/'metadata.json').read_text())
        if hashlib.sha256((directory/'model.joblib').read_bytes()).hexdigest()!=meta['weights_sha256']:raise ValueError('Data-only checkpoint integrity check failed.')
        obj=joblib.load(directory/'model.joblib')
        if obj.model_id!=meta['model_id']:raise ValueError('Data-only checkpoint source/identity changed.')
        return obj
