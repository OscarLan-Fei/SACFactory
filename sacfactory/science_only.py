"""Fixed elicited physical/chemical prior without current-label fitting.

User-reported mass-loss and retention anchors are empirical assumptions; this
control is not a first-principles calculation or an experimentally calibrated model.
"""
import copy
import hashlib
import json
from pathlib import Path

from . import pulseicp_bridge as bridge
from ._pulseicp.references import digest,provenance
from ._pulseicp.composition import decode_composition
from ._pulseicp.screening_prior import predict_xrd_prior,prior_settings

VERSION='fixed-scientific-prior-v1'


class SciencePriorModel:
    def __init__(self,origin='REAL',physics=None,screening=None):
        self.origin=origin;self.physics=copy.deepcopy(physics or bridge.configuration()['physics'])
        self.screening=copy.deepcopy(screening or prior_settings());self.training_ids=[]

    @property
    def model_id(self):
        return digest({'version':VERSION,'physics':self.physics,'screening':self.screening,'references':provenance(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})[:20]

    def predict(self,samples):
        if any(s.get('origin',self.origin)!=self.origin for s in samples):raise ValueError('Do not mix REAL and DEMO controls.')
        compositions=[bridge.composition(s,self.physics) for s in samples]
        probabilities=predict_xrd_prior(compositions,self.physics,self.screening)
        result=[]
        for i,(s,c) in enumerate(zip(samples,compositions)):
            content=decode_composition(c,{e:0. for e in c.elements},self.physics)
            p=float(probabilities.iloc[i].xrd_prior_pass_probability)
            members=[{'element':e,'predicted_mean_w_wt_percent':float(content[e]),
                'predicted_sd_w_wt_percent':None,'calibration_status':'FIXED_PRIOR_POINT_ESTIMATE'} for e in s['elements']]
            result.append({'sample_id':s.get('sample_id'),'model_name':'Science prior (fixed)','model_id':self.model_id,
                'icp_predictions':members,'predicted_total_wt_percent':sum(v['predicted_mean_w_wt_percent'] for v in members),
                'p_xrd_pass':p,'p_xrd_fail':1-p,'xrd_pass_threshold':.5,'predicted_xrd_state':'PASS' if p>=.5 else 'FAIL',
                'xrd_probability_status':str(probabilities.iloc[i].xrd_prior_status),
                'outside_training_support':None,'fitted_to_current_experiments':False,
                'scope':'Fixed mass/thermal/mixing/kinetic assumptions; prior sensitivity is not calibrated predictive uncertainty'})
        return result

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        payload={'version':VERSION,'model_id':self.model_id,'origin':self.origin,'physics':self.physics,
            'screening':self.screening,'training_ids':[],'fitted_to_current_experiments':False,
            'empirical_anchor_disclosure':'Includes user-reported carbonization, pulse-loss and Fe-retention anchors. No current ICP/XRD labels are optimized.'}
        (directory/'metadata.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')

    @classmethod
    def load(cls,directory):
        meta=json.loads((Path(directory)/'metadata.json').read_text())
        model=cls(meta['origin'],meta['physics'],meta['screening'])
        if meta['version']!=VERSION or model.model_id!=meta['model_id']:raise ValueError('Science-prior reference/source identity changed.')
        return model
