"""SAC-Gen gradient-boosting loading learner with the shared physical decoder."""
import copy
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits
from .compute_runtime import jobs
from ._pulseicp.references import digest
from ._pulseicp.features import TrainingScaler
from ._pulseicp.composition import component_features,decode_composition

FAMILY='Gradient boosting residual'
VERSION='sac-gen-gb-residual-v1'
CANDIDATES=[{'max_leaf_nodes':7,'max_iter':150},{'max_leaf_nodes':15,'max_iter':150},
    {'max_leaf_nodes':15,'max_iter':300}]


class GBResidual:
    def __init__(self,family,settings,physics):
        if family!=FAMILY:raise ValueError('Select the gradient-boosting residual family.')
        self.family=family;self.settings=copy.deepcopy(settings);self.physics=copy.deepcopy(physics)
        self.history=[];self.calibration={'status':'DETERMINISTIC_POINT_ESTIMATE','gap_wt_percent':None}

    def fit(self,data,indices):
        indices=np.asarray(indices);self.origin=data.records[indices[0]]['raw']['origin']
        self.protocols=sorted({data.records[i]['raw']['protocol_id'] for i in indices})
        self.training_ids=[data.records[i]['raw']['sample_id'] for i in indices]
        if len(set(self.training_ids))<16:raise ValueError('Sixteen quantified training specimens required.')
        raw=pd.DataFrame([data.records[i]['raw'] for i in indices]).astype(object)
        self.training_fingerprint=digest(raw.where(pd.notna(raw),None).to_dict('records'))
        self.elements={data.records[i]['recipe'].element for i in indices}
        self.compositions={data.records[i]['composition'].key for i in indices}
        weights=np.array([1/len(data.records[i]['composition'].elements) for i in indices])
        x=data.x[indices];self.scaler=TrainingScaler().fit(x,weights)
        groups=np.array([digest(vars(data.records[i]['composition'])) for i in indices])
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=int(self.settings['seed'])+37).split(x,groups=groups))
        def tree(cfg):return HistGradientBoostingRegressor(**cfg,min_samples_leaf=8,l2_regularization=1.,
            learning_rate=.1,early_stopping=False,random_state=int(self.settings['seed']))
        def decoded(fitted,local):
            delta=fitted.predict(x[local]);result=np.empty(len(local));blocks={}
            for j,k in enumerate(local):blocks.setdefault(data.records[indices[k]]['raw']['sample_id'],[]).append((j,k))
            for block in blocks.values():
                comp=data.records[indices[block[0][1]]]['composition']
                w=decode_composition(comp,{data.records[indices[k]]['raw']['element']:delta[j] for j,k in block},self.physics)
                for j,k in block:result[j]=w[data.records[indices[k]]['raw']['element']]
            return result
        with threadpool_limits(limits=jobs()):
            trials=[]
            for cfg in CANDIDATES:
                model=tree(cfg).fit(x[a],data.delta[indices[a]],sample_weight=weights[a])
                trials.append({**cfg,'inner_MAE':float(mean_absolute_error(data.w[indices[b]],decoded(model,b),sample_weight=weights[b]))})
            best=int(np.argmin([v['inner_MAE'] for v in trials]));self.regressor=tree(CANDIDATES[best]).fit(x,data.delta[indices],sample_weight=weights)
        self.selection={'trials':trials,'selected':trials[best],
            'inner_train_ids':sorted({self.training_ids[i] for i in a}),
            'inner_validation_ids':sorted({self.training_ids[i] for i in b})}
        self.history=trials;return self

    @property
    def model_id(self):
        return digest({'version':VERSION,'training':self.training_fingerprint,'physics':self.physics,
            'settings':self.settings,'selection':self.selection,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})[:20]

    def predict_compositions(self,compositions,draws=None,seed=None,origin=None,protocol=None):
        if origin is not None and origin!=self.origin:raise ValueError('Data namespaces differ.')
        if protocol is not None and protocol not in self.protocols:raise ValueError('Protocol differs from fitted context.')
        pairs=[(j,e) for j,c in enumerate(compositions) for e in c.elements]
        x=np.array([component_features(compositions[j],e,self.physics,self.settings.get('thermodynamic_features',True)) for j,e in pairs],dtype=np.float32)
        with threadpool_limits(limits=jobs()):delta=self.regressor.predict(x)
        values={};start=0
        for j,c in enumerate(compositions):
            values[j]=decode_composition(c,dict(zip(c.elements,delta[start:start+len(c.elements)])),self.physics);start+=len(c.elements)
        outside=self.scaler.outside(x);rows=[]
        for k,(j,e) in enumerate(pairs):
            value=float(values[j][e]);unsupported=bool(outside[k] or e not in self.elements or compositions[j].key not in self.compositions)
            rows.append({'composition_index':j,'element':e,'predicted_mean_w_wt_percent':value,
                'predicted_sd_w_wt_percent':0.,'lower_95_w_wt_percent':None,'upper_95_w_wt_percent':None,
                'outside_training_support':unsupported,'unseen_element':e not in self.elements,
                'model_id':self.model_id,'calibration_status':'DETERMINISTIC_POINT_ESTIMATE',
                'uncertainty_note':'No predictive distribution or interval from this deterministic GB learner.'})
        # Point-mass support keeps the common decoder API usable. Repeated
        # values are computational support, never reported as uncertainty draws.
        frame=pd.DataFrame(rows);support=np.repeat(frame.predicted_mean_w_wt_percent.to_numpy()[:,None],2,axis=1)
        return frame,support

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        joblib.dump(self,directory/'model.joblib')
        metadata={'version':VERSION,'family':self.family,'settings':self.settings,'physics':self.physics,
            'model_id':self.model_id,'calibration':self.calibration,'history':self.history,'selection':self.selection,
            'weights_sha256':hashlib.sha256((directory/'model.joblib').read_bytes()).hexdigest()}
        (directory/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')

    @classmethod
    def load(cls,directory):
        directory=Path(directory);metadata=json.loads((directory/'metadata.json').read_text())
        if hashlib.sha256((directory/'model.joblib').read_bytes()).hexdigest()!=metadata['weights_sha256']:
            raise ValueError('GB checkpoint integrity check failed.')
        obj=joblib.load(directory/'model.joblib')
        if obj.model_id!=metadata['model_id']:raise ValueError('GB checkpoint identity changed.')
        return obj
