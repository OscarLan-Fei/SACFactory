"""SAC-Gen: learned mass-conserving ICP distributions and trained XRD probability.

The inherited PulseICP decoder and references retain their source attribution.
The XRD head is regularized logistic regression, not a renamed diffusion head.
Its generated ICP inputs are group-held-out within the training pool.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import pulseicp_bridge as bridge
from ._pulseicp.references import digest
from .compute_runtime import emit, batched_predict
from .validation_groups import recipe_folds, assert_mask

NAME='SAC-Gen'
VERSION='sac-gen-icp-xrd-v1'
FAMILIES=[*bridge.FAMILIES,'Transformer residual','Diffusion Transformer']


@lru_cache(maxsize=4096)
def _prior(serialized):
    """Controls only; an unavailable prior is masked, never a measured label."""
    from ._pulseicp.screening_prior import predict_xrd_prior
    try:
        s=json.loads(serialized)
        result=predict_xrd_prior([bridge.composition(s)],bridge.configuration()['physics'])
        return float(result.iloc[0].xrd_prior_pass_probability)
    except (ValueError,TypeError,KeyError,OverflowError):
        return None


def prior_probability(sample):
    keys=('elements','concentrations_mM','pulse_limit_C','pulse_count','pan_mass_g',
          'solvent_volume_mL','carbonization_loss_fraction','carbonization_loss')
    return _prior(json.dumps({k:sample.get(k) for k in keys},sort_keys=True))


def predict_loading(model,samples,origin):
    def forecast(m,rows,o):return bridge.predict(m,rows,o).to_dict('records')
    errors=[]
    rows=batched_predict(model,samples,origin,forecast,
        lambda s,e:errors.append({'sample_id':s.get('sample_id'),'reason':str(e)}))
    return pd.DataFrame(rows),errors


def loading_features(sample,frame):
    """All-member predictive ICP summaries; no observed ICP is ever accessed."""
    if len(frame) and set(frame.element)==set(sample['elements']):
        mean=float(frame.predicted_mean_w_wt_percent.sum())
        # Sum of marginal SDs is a conservative scale feature, not joint uncertainty.
        spread=float(frame.predicted_sd_w_wt_percent.sum())
        from ._pulseicp.references import elements
        table=elements()
        molar=sum(float(r.predicted_mean_w_wt_percent)/table[r.element]['atomic_weight']
                  for r in frame.itertuples())
        return [np.log1p(mean),np.log1p(spread),np.log1p(molar),0.]
    return [0.,0.,0.,1.]


def feature_matrix(samples,frame,profile):
    from .weighted_validation import recipe_features
    result=[]
    parts={ident:part for ident,part in frame.groupby('sample_id',sort=False)} if len(frame) else {}
    for s in samples:
        part=parts.get(s['sample_id'],pd.DataFrame())
        p=prior_probability(s)
        result.append(np.r_[recipe_features(s,profile),
            logit(np.clip(p,1e-6,1-1e-6)) if p is not None else 0.,float(p is None),
            loading_features(s,part)])
    return np.asarray(result,dtype=float)


class SACGen:
    def __init__(self,family='Gaussian residual',settings=None,profile='chemistry_pulseicp'):
        if family not in FAMILIES:raise ValueError('Select a SAC-Gen loading family.')
        self.family=family;self.settings=copy.deepcopy(settings or bridge.configuration()['model'])
        self.profile=profile;self.head=None;self.loading=None

    @property
    def variant_name(self):
        if self.family=='Gradient boosting residual':return 'SAC-Gen-GB'
        if self.family in {'Conditional diffusion','Diffusion Transformer'}:return 'SAC-Gen-Diffusion'
        return NAME

    def fit(self,samples,origin,methods=None):
        from .weighted_validation import collect
        records,audit=collect(samples,methods or {},origin,self.profile)
        self.origin=origin
        self.training_samples=copy.deepcopy([r['sample'] for r in records])
        self.training_ids=[s['sample_id'] for s in self.training_samples]
        self.training_fingerprint=digest(self.training_samples)
        groups=np.asarray([s['batch_id'] for s in self.training_samples])
        batches=set(groups)
        grouping=self.settings.get('crossfit_grouping','batch')
        if grouping not in ('batch','recipe'):raise ValueError('Select batch or recipe cross-fitting.')
        if grouping=='recipe':
            masks=recipe_folds(self.training_samples,3,int(self.settings['seed']))
            if len(masks)<2:raise ValueError('SAC-Gen needs at least two distinct recipes for nested ICP features.')
        else:
            if len(batches)<3:raise ValueError('SAC-Gen needs at least three actual training batches for nested ICP features.')
            masks=GroupKFold(n_splits=min(3,len(batches))).split(records,groups=groups)
        oof=[];folds=[]
        for index,(source,held) in enumerate(masks,1):
            emit({'stage':f'SAC-Gen inner ICP fold {index}/3','grouping':grouping})
            train=[records[i]['sample'] for i in source];test=[records[i]['sample'] for i in held]
            entry={'train_ids':[s['sample_id'] for s in train],'held_ids':[s['sample_id'] for s in test],
                   'train_batches':sorted(set(groups[source])),'held_batches':sorted(set(groups[held])),
                   'grouping':grouping}
            assert_mask(train,test,grouping=='recipe')
            try:
                model,_=bridge.fit(train,origin,self.family,self.settings)
                frame,errors=predict_loading(model,test,origin)
                if len(frame):oof.append(frame)
                entry.update(model_id=model.model_id,loading_training_ids=model.training_ids,errors=errors)
                if hasattr(model,'tuning'):entry['training_selection']=model.tuning
            except (ValueError,TypeError,KeyError) as exc:entry['unavailable_reason']=str(exc)
            folds.append(entry)
        generated=pd.concat(oof,ignore_index=True) if oof else pd.DataFrame()
        labels=[r for r in records if r['outcomes']['xrd_pass'] is not None]
        y=np.asarray([r['outcomes']['xrd_pass'] for r in labels],dtype=float)
        self.xrd_info={'target':'Recorded XRD/STEM structural-assessment PASS, independent of ICP judgment',
            'family':'L2 logistic regression over chemistry, mechanistic prior and cross-fitted ICP summaries',
            'training_ids':[r['sample']['sample_id'] for r in labels],
            'n':len(y),'pass_n':int(y.sum()),'fail_n':int(len(y)-y.sum()),
            'available':len(y)>=12 and len(set(y))==2,
            'probability_status':'TRAINED_UNCALIBRATED; empirical calibration must be assessed out of fold',
            'inner_folds':folds,'C':.1,'reason':''}
        self.xrd_mean=float(y.mean()) if len(y) else None
        if self.xrd_info['available']:
            x=feature_matrix([r['sample'] for r in labels],generated,self.profile)
            if self.settings.get('refine_xrd',False):
                from .xrd_refinement import fit_head
                self.head,detail=fit_head(x,y,[repr(r['recipe']) for r in labels],int(self.settings['seed']))
                self.xrd_info.update(detail)
                self.xrd_info['probability_status']=detail.get('probability_status',self.xrd_info['probability_status'])
            else:
                scaler=StandardScaler().fit(x)
                head=LogisticRegression(C=.1,max_iter=2000,random_state=int(self.settings['seed'])).fit(scaler.transform(x),y)
                self.head={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                           'coefficient':head.coef_[0].tolist(),'intercept':float(head.intercept_[0])}
        else:self.xrd_info['reason']='Need twelve recorded XRD labels and both classes; unknown labels are not failures.'
        emit({'stage':'SAC-Gen full source ICP fit'})
        self.loading,self.loading_audit=bridge.fit(self.training_samples,origin,self.family,self.settings)
        self.settings=copy.deepcopy(self.loading.settings)
        return self

    @property
    def model_id(self):
        return digest({'version':VERSION,'loading':self.loading.model_id,'profile':self.profile,
                       'head':self.head,'training':self.training_fingerprint})[:20]

    def predict(self,samples):
        if any(s.get('origin',self.origin)!=self.origin for s in samples):
            raise ValueError('SAC-Gen never pools REAL and DEMO predictions.')
        # Ephemeral identifiers make duplicate/unassigned proposed recipes unambiguous.
        rows=[{**s,'sample_id':f'SAC_GEN_QUERY_{i}'} for i,s in enumerate(samples)]
        loading,errors=predict_loading(self.loading,rows,self.origin)
        probability=[None]*len(rows)
        valid=[];features=[]
        try:
            features=feature_matrix(rows,loading,self.profile)
            valid=list(range(len(rows)))
        except (ValueError,TypeError,KeyError,OverflowError):
            for i,s in enumerate(rows):
                try:features.append(feature_matrix([s],loading,self.profile)[0]);valid.append(i)
                except (ValueError,TypeError,KeyError,OverflowError):continue
        if self.head is not None and valid:
            x=(np.asarray(features)-np.asarray(self.head['mean']))/np.asarray(self.head['scale'])
            values=expit(x@np.asarray(self.head['coefficient'])+self.head['intercept'])
            if self.head.get('calibration'):
                calibration=self.head['calibration']
                values=expit(calibration['coefficient']*logit(np.clip(values,1e-6,1-1e-6))+calibration['intercept'])
            for i,p in zip(valid,values):probability[i]=float(p)
        output=[];identity=self.model_id
        parts={ident:part for ident,part in loading.groupby('sample_id',sort=False)} if len(loading) else {}
        for i,s in enumerate(rows):
            p=probability[i]
            part=parts.get(s['sample_id'],pd.DataFrame())
            total=float(part.predicted_mean_w_wt_percent.sum()) if len(part)==len(s['elements']) else None
            output.append({'sample_id':samples[i].get('sample_id'), 'query_index':i,
                'model_name':self.variant_name,'model_id':identity,'p_xrd_pass':p,
                'p_xrd_fail':None if p is None else 1-p,'predicted_total_wt_percent':total,
                'icp_predictions':part.drop(columns='sample_id',errors='ignore').to_dict('records') if len(part) else [],
                'xrd_probability_status':self.xrd_info['probability_status'] if p is not None else self.xrd_info['reason'],
                'xrd_pass_threshold':self.head.get('pass_threshold',.5) if self.head else .5,
                'predicted_xrd_state':None if p is None else 'PASS' if p>=self.head.get('pass_threshold',.5) else 'FAIL',
                'outside_training_support':bool(part.outside_training_support.any()) if len(part) else True})
        return output

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        self.loading.save(path/'loading')
        metadata={'version':VERSION,'name':self.variant_name,'family':self.family,'settings':self.settings,
            'origin':self.origin,'profile':self.profile,'head':self.head,'xrd_info':self.xrd_info,
            'xrd_mean':self.xrd_mean,'training_ids':self.training_ids,
            'training_samples':self.training_samples,'training_fingerprint':self.training_fingerprint,
            'loading_id':self.loading.model_id,'model_id':self.model_id}
        # Exact training snapshots are local checkpoints, excluded from source packages.
        (path/'sac_gen.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')

    @classmethod
    def load(cls,path,current_samples=None):
        path=Path(path);meta=json.loads((path/'sac_gen.json').read_text(encoding='utf-8'))
        if meta['version']!=VERSION:raise ValueError('SAC-Gen checkpoint version changed; retrain.')
        obj=cls(meta['family'],meta['settings'],meta['profile'])
        for key in ('origin','head','xrd_info','xrd_mean','training_ids','training_samples','training_fingerprint'):setattr(obj,key,meta[key])
        from .accelerated_ensemble import ResidualEnsemble
        from .token_residual import TokenResidualEnsemble,FAMILIES as TOKEN_FAMILIES
        loader=TokenResidualEnsemble if obj.family in TOKEN_FAMILIES else ResidualEnsemble
        if obj.family=='Gradient boosting residual':
            from .gb_residual import GBResidual
            loader=GBResidual
        if obj.family in TOKEN_FAMILIES and obj.settings.get('training_variant')=='validated-token-v1':
            from .tuned_token_residual import TunedTokenResidualEnsemble
            loader=TunedTokenResidualEnsemble
        obj.loading=loader.load(path/'loading')
        from .composition import process_key
        obj.loading.sacfactory_process_key=process_key(obj.training_samples[0])
        if obj.model_id!=meta['model_id'] or digest(obj.training_samples)!=obj.training_fingerprint:
            raise ValueError('SAC-Gen checkpoint identity mismatch.')
        if current_samples is not None:
            by_id={s['sample_id']:s for s in current_samples}
            current=[by_id.get(i) for i in obj.training_ids]
            if digest(current)!=obj.training_fingerprint:raise ValueError('SAC-Gen source records changed or were deleted. Retrain before forecasting.')
        return obj


def latest(root,origin,current_samples=None):
    base=Path(root)/'sac_gen'/origin.lower();pointer=base/'latest.json'
    if not pointer.exists():return None
    name=json.loads(pointer.read_text())['run_id']
    if Path(name).name!=name or name in {'.','..'}:raise ValueError('Invalid SAC-Gen checkpoint reference.')
    model=SACGen.load(base/name,current_samples)
    if model.origin!=origin:raise ValueError('SAC-Gen checkpoint belongs to another data space.')
    return model


def forecast(samples,origin,root,current_samples):
    model=latest(root,origin,current_samples)
    return model.predict(samples) if model is not None else []


def activate_variant(root,origin,name,current_samples):
    """Activate a verified saved learner variant without refitting or editing data."""
    base=Path(root)/'sac_gen'/origin.lower()
    registry=json.loads((base/'variants.json').read_text())
    selected=registry['variants'].get(name)
    if selected is None:raise ValueError('Select a registered SAC-Gen variant.')
    run=selected['run_id']
    if Path(run).name!=run or run in {'','.','..'}:raise ValueError('Invalid saved variant path.')
    model=SACGen.load(base/run,current_samples)
    if model.origin!=origin or model.model_id!=selected['model_id']:
        raise ValueError('Saved variant origin or identity differs.')
    if not model.predict(model.training_samples[:1]):raise ValueError('Saved variant forecast check failed.')
    path=base/run/'training_run.json'
    pointer=json.loads(path.read_text()) if path.exists() else dict(selected)
    pointer.update(name=name,run_id=run,model_id=model.model_id,family=model.family,version=VERSION,
        training_samples=len(model.training_ids),validation_subdirectory='validation')
    temporary=base/'latest.pending.json';temporary.write_text(json.dumps(pointer,indent=2),encoding='utf-8')
    temporary.replace(base/'latest.json')
    registry['default']=name
    temporary=base/'variants.pending.json';temporary.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    temporary.replace(base/'variants.json')
    return pointer


def train_deployment(samples,origin,root,family,settings,methods=None):
    """Refit the selected algorithm; deployment is separate from OOF metrics."""
    from datetime import datetime,timezone
    from uuid import uuid4
    from contextlib import closing
    import sqlite3
    model=SACGen(family,settings).fit(samples,origin,methods)
    database=Path(root)/f'{origin.lower()}.sqlite3'
    if database.exists():
        with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            current={ident:json.loads(payload) for ident,payload in db.execute("SELECT id,payload FROM items WHERE kind='sample'")}
        if digest([current.get(i) for i in model.training_ids])!=model.training_fingerprint:
            raise ValueError('Source experiments changed during training. Retry on the current data.')
    run_id='sac_gen_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:6]
    base=Path(root)/'sac_gen'/origin.lower();base.mkdir(parents=True,exist_ok=True)
    model.save(base/run_id)
    # Check serialization and numerical forecasts before activating the checkpoint.
    restored=SACGen.load(base/run_id,samples)
    probe=restored.predict(model.training_samples[:1])
    if not probe or probe[0]['predicted_total_wt_percent'] is None:
        raise ValueError('Saved SAC-Gen checkpoint failed its forecast smoke test; previous deployment remains active.')
    pointer={'run_id':run_id,'model_id':model.model_id,'family':family,'version':VERSION,
             'training_samples':len(model.training_ids),'selection':'User-selected deployment; performance requires masked validation'}
    temp=base/'latest.pending.json';temp.write_text(json.dumps(pointer,indent=2))
    temp.replace(base/'latest.json')
    return pointer
