"""Small permutation-invariant elemental set model with joint mass closure."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import digest,identity,outcomes,query
from ..chemical_descriptors import descriptor_matrix
from ..domain import ELEMENTS
from .. import pulseicp_bridge as bridge
from .._pulseicp.mass_decoder import prior


class SetNetwork(nn.Module):
    def __init__(self,dimension,width=24):
        super().__init__()
        self.phi=nn.Sequential(nn.Linear(dimension,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.delta=nn.Sequential(nn.Linear(3*width+4,width),nn.SiLU(),nn.Linear(width,1))
        self.structure=nn.Sequential(nn.Linear(2*width+4,width),nn.SiLU(),nn.Linear(width,1))

    def forward(self,tokens,mask,fractions,controls,feeds,matrix,prior_logits):
        hidden=self.phi(tokens)
        mean=(hidden*fractions[:,:,None]*mask[:,:,None]).sum(1)
        maximum=hidden.masked_fill(~mask[:,:,None],-1e6).max(1).values
        context=torch.cat([mean,maximum,controls],1)
        delta=self.delta(torch.cat([hidden,context[:,None,:].expand(-1,tokens.shape[1],-1)],2)).squeeze(-1)
        retained=feeds*torch.sigmoid(prior_logits+delta)*mask
        w=100*retained/(matrix[:,None]+retained.sum(1,keepdim=True))
        return w,self.structure(context).squeeze(-1)


class SetResidual:
    variant_name='SAC-Gen-Set'
    def __init__(self,width=24,epochs=100,seed=42):
        if not 4<=int(width)<=128 or not 1<=int(epochs)<=2000:raise ValueError('Invalid set-model budget.')
        self.width=int(width);self.epochs=int(epochs);self.seed=int(seed)
        self.physics=bridge.configuration()['physics'];self.descriptors=descriptor_matrix('chemistry')

    def encode(self,samples):
        maximum=max(len(s['elements']) for s in samples);dimension=self.descriptors.shape[1]+4
        count=len(samples);tokens=np.zeros((count,maximum,dimension),dtype=np.float32)
        mask=np.zeros((count,maximum),dtype=bool);fractions=np.zeros((count,maximum),dtype=np.float32)
        feeds=np.zeros_like(fractions);logits=np.zeros_like(fractions);matrix=[];controls=[];orders=[]
        for j,sample in enumerate(samples):
            s=query(sample);elements=s['elements'];orders.append(elements)
            comp=bridge.composition(s,self.physics);total=sum(s['concentrations_mM'].values())
            controls.append([s['pulse_limit_C']/1000,np.log1p(s['pulse_count']),np.log(s['pan_mass_g']),np.log(s['solvent_volume_mL'])])
            for k,e in enumerate(elements):
                concentration=s['concentrations_mM'][e];entry=s.get('precursors',{}).get(e,{})
                charge=entry.get('oxidation_state');source=entry.get('source','')
                known=charge is not None and bool(source) and 'ASSUMPTION' not in source.upper()
                tokens[j,k]=np.r_[self.descriptors[ELEMENTS.index(e)],np.log1p(concentration)/5,concentration/total,
                                  float(charge)/5 if known else 0.,0. if known else 1.]
                p=prior(comp.component(e),self.physics);feeds[j,k]=p['feed_metal_g']
                r=np.clip(p['retention_prior'],1e-6,1-1e-6);logits[j,k]=np.log(r/(1-r))
                fractions[j,k]=concentration/total;mask[j,k]=True
            matrix.append(p['matrix_mass_g'])
        data=[tokens,mask,fractions,np.array(controls,dtype=np.float32),feeds,np.array(matrix,dtype=np.float32),logits]
        return [torch.as_tensor(v) for v in data],orders

    def fit(self,samples,origin):
        torch.manual_seed(self.seed)
        if any(s.get('origin')!=origin for s in samples):raise ValueError('Set training cannot mix namespaces.')
        kept=[];self.training_audit=[]
        for s in samples:
            actual,label=outcomes(s)
            if actual or label is not None:kept.append(s)
            else:self.training_audit.append({'sample_id':s.get('sample_id'),'reason':'No valid learning outcome'})
        if len(kept)<12:raise ValueError('Set training needs at least twelve valid experimental specimens.')
        self.origin=origin;self.training_ids=[s['sample_id'] for s in kept];self.training_samples=kept
        self.training_fingerprint=digest(kept);self.training_keys=[identity(s) for s in kept]
        x,orders=self.encode(kept);count,maximum=x[1].shape
        y=torch.zeros((count,maximum));observed=torch.zeros((count,maximum),dtype=torch.bool)
        labels=torch.zeros(count);structural=torch.zeros(count,dtype=torch.bool)
        for i,(s,elements) in enumerate(zip(kept,orders)):
            actual,label=outcomes(s)
            for j,e in enumerate(elements):
                if e in actual:y[i,j]=actual[e];observed[i,j]=True
            if label is not None:labels[i]=label;structural[i]=True
        self.has_structure=bool(structural.any() and len(torch.unique(labels[structural]))==2)
        self.model=SetNetwork(x[0].shape[-1],self.width)
        optimizer=torch.optim.Adam(self.model.parameters(),lr=.003,weight_decay=1e-4)
        self.loss_history=[]
        for epoch in range(self.epochs):
            self.model.train();prediction,logit=self.model(*x)
            errors=((torch.log1p(prediction)-torch.log1p(y))**2)*observed
            counts=observed.sum(1);valid=counts>0
            loss=(errors.sum(1)[valid]/counts[valid]).mean() if valid.any() else logit.sum()*0
            if self.has_structure:loss=loss+.2*nn.functional.binary_cross_entropy_with_logits(logit[structural],labels[structural])
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),5.);optimizer.step()
            self.loss_history.append(float(loss.detach()))
        self.ranges={}
        for s in kept:
            for e in s['elements']:
                vector=[s['pulse_limit_C'],s['pulse_count'],s['concentrations_mM'][e]]
                self.ranges.setdefault(e,[]).append(vector)
        self.ranges={e:{'min':np.min(v,axis=0).tolist(),'max':np.max(v,axis=0).tolist()} for e,v in self.ranges.items()}
        self.model_id=digest({'schema':'element-set-mass-closure-v1','source':self.training_fingerprint,
                             'width':self.width,'epochs':self.epochs,'seed':self.seed,'physics':self.physics,
                             'descriptors':digest(self.descriptors.tolist())})[:24]
        self.descriptor_signature=digest(self.descriptors.tolist())
        return self

    def predict(self,samples):
        if not samples:return []
        if any(s.get('origin')!=self.origin for s in samples):raise ValueError('Set prediction namespace differs.')
        x,orders=self.encode(samples);self.model.eval()
        with torch.no_grad():w,logit=self.model(*x);probability=torch.sigmoid(logit).numpy()
        result=[]
        for i,(s,elements) in enumerate(zip(samples,orders)):
            outside=False
            for e in elements:
                vector=np.array([s['pulse_limit_C'],s['pulse_count'],s['concentrations_mM'][e]])
                outside |= e not in self.ranges or bool(np.any(vector<self.ranges[e]['min']) or np.any(vector>self.ranges[e]['max']))
            assays=[{'element':e,'predicted_mean_w_wt_percent':float(w[i,j])} for j,e in enumerate(elements)]
            p=float(probability[i]) if self.has_structure else None
            result.append({'model_name':self.variant_name,'model_id':self.model_id,'sample_id':s.get('sample_id'),
                           'icp_predictions':assays,'predicted_total_wt_percent':sum(a['predicted_mean_w_wt_percent'] for a in assays),
                           'p_xrd_pass':p,'p_xrd_fail':None if p is None else 1-p,'outside_training_support':outside,
                           'xrd_probability_status':'EXPERIMENTAL_UNCALIBRATED','architecture':'Permutation-invariant weighted-mean/max DeepSets with common mass denominator'})
        return result

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        torch.save(self.model.state_dict(),path/'weights.pt')
        data={k:getattr(self,k) for k in ['origin','width','epochs','seed','physics','training_ids','training_keys',
              'training_fingerprint','ranges','model_id','has_structure','loss_history','training_audit','descriptor_signature']}
        (path/'metadata.json').write_text(json.dumps(data,indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path):
        path=Path(path);data=json.loads((path/'metadata.json').read_text())
        model=cls(data['width'],data['epochs'],data['seed'])
        if digest(model.descriptors.tolist())!=data['descriptor_signature']:raise ValueError('Element reference schema changed; retrain the set model.')
        for key,value in data.items():setattr(model,key,value)
        model.model=SetNetwork(model.descriptors.shape[1]+4,model.width)
        model.model.load_state_dict(torch.load(path/'weights.pt',map_location='cpu',weights_only=True))
        model.model.eval();return model
