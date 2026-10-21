"""Compact experimental tabular attention networks for physical ICP residuals.

Numerical tokenization is inspired by Gorishniy et al. (2021, arXiv:2106.11959).
Transformer denoising is inspired by Peebles and Xie (2023, arXiv:2212.09748).
This is a small tabular adaptation, not the published image DiT implementation.
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

from .accelerated_ensemble import ResidualEnsemble,feature_names
from .compute_runtime import resolve, emit
from ._pulseicp.features import TrainingScaler
from ._pulseicp.references import digest
from .accelerated_diffusion import training_loss

TRANSFORMER='Transformer residual'
DIFFUSION_TRANSFORMER='Diffusion Transformer'
FAMILIES=[TRANSFORMER,DIFFUSION_TRANSFORMER]


class NumericTokens(nn.Module):
    def __init__(self,inputs,width):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(inputs,width)*.02)
        self.bias=nn.Parameter(torch.randn(inputs,width)*.02)

    def forward(self,x):
        return x[:,:,None]*self.weight[None,:,:]+self.bias[None,:,:]


def attention(width,heads,layers):
    layer=nn.TransformerEncoderLayer(width,heads,dim_feedforward=width*2,
        dropout=.1,activation='gelu',batch_first=True,norm_first=True)
    return nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)


class TransformerGaussian(nn.Module):
    def __init__(self,inputs,width,heads,layers):
        super().__init__()
        self.encoder=NumericTokens(inputs,width)
        self.cls=nn.Parameter(torch.zeros(1,1,width))
        self.blocks=attention(width,heads,layers)
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2))

    def forward(self,x):
        tokens=torch.cat([self.cls.expand(len(x),-1,-1),self.encoder(x)],dim=1)
        out=self.head(self.blocks(tokens)[:,0])
        return out[:,:1],out[:,1:].clamp(-7.,3.)


class TransformerDenoiser(nn.Module):
    def __init__(self,inputs,width,heads,layers):
        super().__init__()
        self.encoder=NumericTokens(inputs,width)
        self.noisy=nn.Linear(1,width)
        self.time=nn.Sequential(nn.Linear(3,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=attention(width,heads,layers)
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))

    def denoise(self,z,step,condition):
        t=torch.stack([step,torch.sin(step*torch.pi),torch.cos(step*torch.pi)],dim=1)
        query=(self.noisy(z)+self.time(t))[:,None,:]
        return self.head(self.blocks(torch.cat([query,condition],dim=1))[:,0])

    def forward(self,z,step,x):
        return self.denoise(z,step,self.encoder(x))


class TokenResidualEnsemble(ResidualEnsemble):
    """Reuse the frozen physical decoder, training scaler and draw accounting."""
    def __init__(self,family,settings,physics):
        if family not in FAMILIES:raise ValueError('Select a supported token residual family.')
        self.architecture_family=family
        settings={**settings,'architecture':family,'architecture_version':'tabular-token-residual-v1',
            'architecture_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        super().__init__('Gaussian residual' if family==TRANSFORMER else 'Conditional diffusion',settings,physics)

    def fit(self,data,indices,callback=None):
        settings=self.settings
        width=int(settings.get('token_width',32));heads=int(settings.get('attention_heads',4));layers=int(settings.get('attention_layers',2))
        count=int(settings['members'])
        epochs=int(settings['epochs_baseline'] if self.family=='Gaussian residual' else settings['epochs_diffusion'])
        if width not in (16,32,64) or heads not in (1,2,4,8) or width%heads or not 1<=layers<=3:
            raise ValueError('Use width 16/32/64, compatible heads 1/2/4/8 and 1-3 attention layers.')
        if not 1<=count<=5 or not 1<=epochs<=2000 or settings['learning_rate']<=0:
            raise ValueError('Use positive training budgets within the ensemble limits.')
        device=torch.device(resolve(settings)['device'])
        torch.set_num_threads(resolve(settings)['threads'])
        x,y=data.x[indices],data.delta[indices]
        if x.shape[1]!=len(feature_names(settings)):raise ValueError('Token feature schema differs from prepared data.')
        rows=[data.records[i] for i in indices]
        weight=np.array([1/len(r['composition'].elements) for r in rows])
        batches=np.array([str(r['raw']['batch_id']) for r in rows]);unique=sorted(set(batches))
        if len({r['raw']['sample_id'] for r in rows})<16 or len(unique)<2:
            raise ValueError('Token fitting needs 16 eligible specimens and at least 2 actual batches.')
        self.scaler=TrainingScaler().fit(x,weight)
        self.y_mean=float(np.average(y,weights=weight))
        self.y_scale=max(float(np.sqrt(np.average((y-self.y_mean)**2,weights=weight))),.05)
        z=(y-self.y_mean)/self.y_scale
        self.clean_limit=max(6.,float(np.max(np.abs(z)))+2.)
        self.compositions=sorted({r['composition'].key for r in rows})
        self.has_mixture_training=any(len(r['composition'].elements)>1 for r in rows)
        self.elements=sorted({r['recipe'].element for r in rows})
        self.origin=rows[0]['raw']['origin'];self.protocols=sorted({r['raw']['protocol_id'] for r in rows})
        self.training_ids=[str(r['raw']['sample_id']) for r in rows]
        raw=pd.DataFrame([r['raw'] for r in rows]).astype(object)
        self.training_fingerprint=digest(raw.where(pd.notna(raw),None).to_dict('records'))
        for member in range(count):
            seed=int(settings['seed'])+member*101;torch.manual_seed(seed)
            blocks=np.random.default_rng(seed).choice(unique,len(unique),replace=True)
            chosen=np.concatenate([np.flatnonzero(batches==b) for b in blocks])
            self.bootstrap.append({'member':member,'batch_ids_with_replacement':blocks.tolist(),
                'sample_ids_with_replacement':[self.training_ids[i] for i in chosen]})
            xx=torch.from_numpy(self.scaler.transform(x[chosen])).to(device)
            zz=torch.tensor(z[chosen,None],dtype=torch.float32,device=device)
            ww=torch.tensor(weight[chosen]/weight[chosen].mean(),dtype=torch.float32,device=device)
            cls=TransformerGaussian if self.family=='Gaussian residual' else TransformerDenoiser
            model=cls(x.shape[1],width,heads,layers).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=float(settings['learning_rate']),weight_decay=1e-4)
            generator=torch.Generator(device=device).manual_seed(seed+1)
            for epoch in range(epochs):
                model.train();losses=[]
                for ids in torch.randperm(len(xx),generator=generator,device=device).split(int(settings['batch_size'])):
                    optimizer.zero_grad()
                    if self.family=='Gaussian residual':
                        mean,logvar=model(xx[ids])
                        loss_row=.5*((zz[ids]-mean)**2*torch.exp(-logvar)+logvar).mean(dim=1)
                    else:
                        loss_row=training_loss(model,xx[ids],zz[ids],int(settings['diffusion_steps']),generator,reduction=False)
                    loss=(loss_row*ww[ids]).mean()
                    if not torch.isfinite(loss):raise ValueError('Token training became nonfinite.')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    losses.append(float(loss.detach()))
                event={'family':self.architecture_family,'member':member+1,'epoch':epoch+1,'epochs':epochs,
                    'training_loss':float(np.mean(losses))}
                self.history.append(event)
                if epoch==0 or (epoch+1)%20==0 or epoch+1==epochs:emit(event)
                if callback:callback(event)
            model.eval();self.members.append(model)
        return self

    def predict_compositions(self,*args,**kwargs):
        frame,draws=super().predict_compositions(*args,**kwargs)
        frame['model']=self.architecture_family
        return frame,draws

    def save(self,directory):
        super().save(directory)

    @classmethod
    def load(cls,directory):
        import json
        from ._pulseicp import __version__
        from .accelerated_ensemble import model_references
        from .compute_runtime import resolve
        directory=Path(directory)
        meta=json.loads((directory/'metadata.json').read_text())
        if meta['version']!=__version__ or meta['feature_names']!=feature_names(meta['settings']):
            raise ValueError('Token checkpoint schema changed; retrain.')
        if meta['references']!=model_references(meta['settings']):raise ValueError('Token reference snapshot changed.')
        if hashlib.sha256((directory/'weights.pt').read_bytes()).hexdigest()!=meta['weights_sha256']:
            raise ValueError('Token weights failed integrity verification.')
        obj=cls(meta['settings']['architecture'],meta['settings'],meta['physics'])
        obj.scaler=TrainingScaler.from_dict(meta['scaler'])
        for key in ('y_mean','y_scale','clean_limit','elements','origin','protocols','training_ids',
                    'training_fingerprint','bootstrap','history','calibration','compositions','has_mixture_training'):
            setattr(obj,key,meta[key])
        weights=torch.load(directory/'weights.pt',map_location='cpu',weights_only=True)
        settings=obj.settings
        net=TransformerGaussian if obj.family=='Gaussian residual' else TransformerDenoiser
        runtime=resolve({**settings,'device':'auto'})
        for i in range(int(settings['members'])):
            model=net(len(meta['feature_names']),int(settings.get('token_width',32)),
                int(settings.get('attention_heads',4)),int(settings.get('attention_layers',2)))
            model.load_state_dict(weights[f'member_{i}'])
            obj.members.append(model.to(runtime['device']).eval())
        if obj.model_id!=meta['model_id']:raise ValueError('Token checkpoint identity changed; retrain.')
        return obj
