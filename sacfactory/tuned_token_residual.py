"""Training-only recipe holdouts, early stopping and bounded learning-rate search.

The network and decoder are unchanged. Legacy token checkpoints remain loadable.
"""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from .token_residual import TokenResidualEnsemble, TransformerGaussian, TransformerDenoiser
from .accelerated_diffusion import training_loss, sample
from .compute_runtime import emit, resolve
from ._pulseicp.features import TrainingScaler
from ._pulseicp.composition import decode_composition

VARIANT='validated-token-v1'


def coverage_seed(batches,members,seed):
    """Select on batch identifiers alone; never inspect target values."""
    if members<3:return int(seed)
    unique=sorted(set(batches))
    for candidate in range(int(seed),int(seed)+10000):
        draws=[set(np.random.default_rng(candidate+member*101).choice(unique,len(unique),replace=True))
               for member in range(members)]
        if set.union(*draws)==set(unique):return candidate
    raise ValueError('No complete ensemble batch coverage seed found within the declared search budget.')


class TunedTokenResidualEnsemble(TokenResidualEnsemble):
    def __init__(self,family,settings,physics):
        settings={**settings,'training_variant':VARIANT,
            'validated_training_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        super().__init__(family,settings,physics)

    def fit(self,data,indices,callback=None):
        settings=self.settings
        original_seed=int(settings['seed']);device=torch.device(resolve(settings)['device'])
        torch.set_num_threads(resolve(settings)['threads'])
        indices=np.asarray(indices)
        rows=[data.records[i] for i in indices]
        groups=[json.dumps(asdict(r['composition']),sort_keys=True) for r in rows]
        if len(set(groups))<5:raise ValueError('Validated fitting needs at least five distinct recipes.')
        source,held=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=original_seed).split(indices,groups=groups))
        train_indices,validation_indices=indices[source],indices[held]
        assert not set(groups[i] for i in source)&set(groups[i] for i in held)
        weight=np.array([1/len(r['composition'].elements) for r in rows])
        scaler=TrainingScaler().fit(data.x[train_indices],weight[source])
        center=float(np.average(data.delta[train_indices],weights=weight[source]))
        spread=max(float(np.sqrt(np.average((data.delta[train_indices]-center)**2,weights=weight[source]))),.05)
        limit=max(6.,float(np.max(np.abs((data.delta[train_indices]-center)/spread)))+2.)
        xx=torch.from_numpy(scaler.transform(data.x[train_indices])).to(device)
        vx=torch.from_numpy(scaler.transform(data.x[validation_indices])).to(device)
        zz=torch.tensor((data.delta[train_indices,None]-center)/spread,dtype=torch.float32,device=device)
        vz=torch.tensor((data.delta[validation_indices,None]-center)/spread,dtype=torch.float32,device=device)
        ww=torch.tensor(weight[source]/weight[source].mean(),dtype=torch.float32,device=device)
        vw=torch.tensor(weight[held]/weight[held].mean(),dtype=torch.float32,device=device)
        maximum=int(settings.get('tuning_max_epochs',400));minimum=int(settings.get('tuning_min_epochs',100))
        interval=int(settings.get('tuning_interval',20));patience=int(settings.get('tuning_patience',3))
        rates=list(settings.get('candidate_learning_rates',[.0003,.001]))
        if not 1<=minimum<=maximum<=2000 or interval<1 or patience<1 or not rates or any(rate<=0 for rate in rates):
            raise ValueError('Use valid bounded early-stopping settings and positive learning rates.')
        steps=int(settings['diffusion_steps']);trials=[]
        for rate in rates:
            torch.manual_seed(original_seed)
            network=(TransformerGaussian if self.family=='Gaussian residual' else TransformerDenoiser)(
                data.x.shape[1],int(settings.get('token_width',32)),int(settings.get('attention_heads',4)),int(settings.get('attention_layers',2))).to(device)
            optimizer=torch.optim.AdamW(network.parameters(),lr=float(rate),weight_decay=1e-4)
            generator=torch.Generator(device=device).manual_seed(original_seed+1)
            best=float('inf');best_epoch=minimum;best_state=None;stale=0;history=[]
            for epoch in range(1,maximum+1):
                network.train();losses=[]
                for ids in torch.randperm(len(xx),generator=generator,device=device).split(int(settings['batch_size'])):
                    optimizer.zero_grad()
                    if self.family=='Gaussian residual':
                        mean,logvar=network(xx[ids]);per=.5*((zz[ids]-mean)**2*torch.exp(-logvar)+logvar).mean(dim=1)
                    else:per=training_loss(network,xx[ids],zz[ids],steps,generator,reduction=False)
                    loss=(per*ww[ids]).mean()
                    if not torch.isfinite(loss):raise ValueError('Early-stopping training became nonfinite.')
                    loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
                if epoch%interval and epoch!=maximum and epoch!=minimum:continue
                network.eval()
                with torch.no_grad():
                    if self.family=='Gaussian residual':
                        mean,logvar=network(vx);per=.5*((vz-mean)**2*torch.exp(-logvar)+logvar).mean(dim=1)
                    else:
                        fixed=torch.Generator(device=device).manual_seed(original_seed+7777)
                        per=training_loss(network,vx,vz,steps,fixed,reduction=False)
                    validation=float((per*vw).mean())
                event={'stage':'Inner early-stopping search','learning_rate':rate,'epoch':epoch,'epochs':maximum,
                    'training_loss':float(np.mean(losses)),'validation_loss':validation}
                history.append(event);emit(event)
                if epoch<minimum:continue
                if validation<best-float(settings.get('tuning_min_delta',.001)):
                    best=validation;best_epoch=epoch;best_state=copy.deepcopy(network.state_dict());stale=0
                else:stale+=1
                if stale>=patience:break
            network.load_state_dict(best_state);network.eval()
            draws=int(settings.get('tuning_draws',16))
            with torch.no_grad():
                if self.family=='Gaussian residual':
                    mean,logvar=network(vx);rng=torch.Generator(device=device).manual_seed(original_seed+8888)
                    values=(mean+torch.exp(.5*logvar)*torch.randn((len(vx),draws),generator=rng,device=device)).cpu().numpy()
                else:values,_=sample(network,vx,draws,steps,original_seed+8888,limit)
            residuals=values*spread+center;pred=np.zeros(len(held));blocks={}
            for k,i in enumerate(held):blocks.setdefault(rows[i]['raw']['sample_id'],[]).append(k)
            for block in blocks.values():
                c=rows[held[block[0]]]['composition']
                decoded=decode_composition(c,{rows[held[k]]['recipe'].element:residuals[k] for k in block},self.physics)
                for k in block:pred[k]=float(np.mean(decoded[rows[held[k]]['recipe'].element]))
            mae=float(np.average(np.abs(pred-data.w[validation_indices]),weights=weight[held]))
            trials.append({'learning_rate':float(rate),'selected_epochs':best_epoch,'validation_MAE_wt_percent':mae,
                'validation_loss':best,'epochs_run':epoch,'history':history})
            emit({'stage':'Inner learning-rate trial complete','learning_rate':rate,'selected_epochs':best_epoch,'validation_MAE_wt_percent':mae})
        selected=min(trials,key=lambda trial:trial['validation_MAE_wt_percent'])
        fit_seed=coverage_seed([r['raw']['batch_id'] for r in rows],int(settings['members']),original_seed)
        self.tuning={'selection':'Training-only grouped holdout; fixed-noise validation early stopping, decoded ICP MAE rate selection',
            'requested_seed':original_seed,'fit_seed':fit_seed,'selection_train_ids':sorted({rows[i]['raw']['sample_id'] for i in source}),
            'selection_validation_ids':sorted({rows[i]['raw']['sample_id'] for i in held}),
            'scaler_training_ids':sorted({rows[i]['raw']['sample_id'] for i in source}),
            'trials':trials,'selected':selected}
        settings.update(learning_rate=selected['learning_rate'],epochs_baseline=selected['selected_epochs'],
            epochs_diffusion=selected['selected_epochs'],seed=fit_seed,
            tuning_selection={'requested_seed':original_seed,'selected_epochs':selected['selected_epochs'],
                'selected_learning_rate':selected['learning_rate'],'validation_MAE_wt_percent':selected['validation_MAE_wt_percent']})
        emit({'stage':'Refitting all supplied training specimens at selected budget','epochs':selected['selected_epochs'],'learning_rate':selected['learning_rate']})
        return super().fit(data,indices,callback)

    def save(self,directory):
        super().save(directory)
        (Path(directory)/'training_selection.json').write_text(json.dumps(self.tuning,indent=2,allow_nan=False),encoding='utf-8')

    @classmethod
    def load(cls,directory):
        obj=super().load(directory)
        obj.tuning=json.loads((Path(directory)/'training_selection.json').read_text(encoding='utf-8'))
        return obj
