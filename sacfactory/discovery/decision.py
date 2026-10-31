"""Deployable discovery ensemble with disjoint fit, weight and calibration pools."""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .common import digest,identity,query,recipe_id
from .calibration import Calibration,protected_weights,fuse
from .set_model import SetResidual
from ..science_only import SciencePriorModel


class DecisionEnsemble:
    variant_name='Calibrated discovery ensemble'
    def fit(self,samples,origin,epochs=100,seed=42):
        groups=np.array([recipe_id(s) for s in samples])
        if len(set(groups))<30:raise ValueError('A calibrated decision ensemble needs thirty distinct recipes.')
        fit,remaining=next(GroupShuffleSplit(test_size=.4,random_state=seed).split(samples,groups=groups))
        weight,cal=next(GroupShuffleSplit(test_size=.5,random_state=seed+1).split(remaining,groups=groups[remaining]))
        weight=remaining[weight];cal=remaining[cal]
        training=[samples[i] for i in fit];weight_samples=[samples[i] for i in weight];cal_samples=[samples[i] for i in cal]
        self.origin=origin;self.training_ids=[s['sample_id'] for s in samples];self.training_keys=[identity(s) for s in samples]
        self.training_fingerprint=digest(samples)
        self.models={'Set':SetResidual(epochs=epochs,seed=seed).fit(training,origin),'Prior':SciencePriorModel(origin)}
        def predict(pool):return {n:m.predict([query(s) for s in pool]) for n,m in self.models.items()}
        self.weights,self.weight_status=protected_weights(weight_samples,predict(weight_samples),training+cal_samples)
        self.calibration=Calibration().fit(cal_samples,fuse(predict(cal_samples),self.weights),training+weight_samples,alpha=.1)
        self.split={'fit_keys':[identity(s) for s in training],'weight_keys':[identity(s) for s in weight_samples],
                    'calibration_keys':[identity(s) for s in cal_samples]}
        self.model_id=digest({'data':self.training_fingerprint,'weights':self.weights,'epochs':epochs,'seed':seed})[:24]
        return self

    def predict(self,samples):
        result=self.calibration.apply(fuse({n:m.predict([query(s) for s in samples]) for n,m in self.models.items()},self.weights))
        for r in result:r.update(model_name=self.variant_name,model_id=self.model_id)
        return result

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        self.models['Set'].save(path/'set');self.models['Prior'].save(path/'prior')
        data={k:getattr(self,k) for k in ['origin','training_ids','training_keys','training_fingerprint','weights','weight_status','split','model_id']}
        calibration=self.calibration
        data['calibration']={'alpha':calibration.alpha,'radius':calibration.radius,'info':calibration.info,
                             'keys':calibration.calibration_keys,'platt':None if calibration.probability_model is None else
                             {'coefficient':calibration.probability_model.coef_[0,0],'intercept':calibration.probability_model.intercept_[0]}}
        (path/'metadata.json').write_text(json.dumps(data,indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path):
        from scipy.special import expit
        path=Path(path);data=json.loads((path/'metadata.json').read_text());model=cls()
        for k,v in data.items():
            if k!='calibration':setattr(model,k,v)
        model.models={'Set':SetResidual.load(path/'set'),'Prior':SciencePriorModel.load(path/'prior')}
        info=data['calibration'];cal=Calibration();cal.alpha=info['alpha'];cal.radius=info['radius'];cal.info=info['info'];cal.calibration_keys=info['keys']
        cal.probability_model=None
        if info['platt'] is not None:
            from sklearn.linear_model import LogisticRegression
            platt=LogisticRegression();platt.classes_=np.array([0,1]);platt.coef_=np.array([[info['platt']['coefficient']]])
            platt.intercept_=np.array([info['platt']['intercept']]);platt.n_features_in_=1;cal.probability_model=platt
        model.calibration=cal;return model
