"""Discovery model lifecycle and idempotent links to the existing Pending UI."""
import copy
import json
from pathlib import Path
from uuid import uuid4
from .common import digest,identity,query,recipe_id,stamp
from .set_model import SetResidual


def available_models(root,origin):
    root=Path(root);names=['Fixed prior']
    path=root/'sac_gen'/origin.lower()/'variants.json'
    if path.exists():names+=list(json.loads(path.read_text())['variants'])
    index=root/'discovery_models'/origin.lower()/'index.json'
    if index.exists():names+=list(json.loads(index.read_text()))
    if origin=='REAL':
        from ..pretrained import available
        names+=['Pretrained '+name for name in available()]
    return list(dict.fromkeys(names))


def load_model(root,origin,name,samples):
    root=Path(root)
    if name.startswith('Pretrained '):
        if origin!='REAL':raise ValueError('Public pretrained models use the REAL prediction workspace.')
        from ..pretrained import load
        return load(name)
    if name=='Fixed prior':
        from ..science_only import SciencePriorModel
        return SciencePriorModel(origin)
    index=root/'discovery_models'/origin.lower()/'index.json'
    entries=json.loads(index.read_text()) if index.exists() else {}
    if name in entries:
        item=entries[name]
        if Path(item['run_id']).name!=item['run_id']:raise ValueError('Invalid candidate path.')
        path=index.parent/item['run_id']
        if name in {'SAC-Gen-Set','Calibrated discovery ensemble'}:
            from .decision import DecisionEnsemble
            model=(SetResidual if name=='SAC-Gen-Set' else DecisionEnsemble).load(path)
            current={identity(s):s for s in samples if s.get('origin')==origin}
            if digest([current.get(k) for k in model.training_keys])!=model.training_fingerprint:
                raise ValueError('Set-model training records changed; train a new candidate.')
            return model
        from ..sac_gen import SACGen
        return SACGen.load(path,samples)
    path=root/'sac_gen'/origin.lower()/'variants.json'
    if not path.exists():raise ValueError('Train or register this model first.')
    item=json.loads(path.read_text())['variants'].get(name)
    if item is None or Path(item['run_id']).name!=item['run_id']:raise ValueError('Unknown saved model.')
    from ..sac_gen import SACGen
    return SACGen.load(path.parent/item['run_id'],samples)


def train_candidate(root,samples,origin,name='SAC-Gen-Set',epochs=100,seed=42):
    from .validation import usable
    sources=usable(samples,origin)
    if name=='Calibrated discovery ensemble':
        from .decision import DecisionEnsemble
        model=DecisionEnsemble().fit(sources,origin,epochs,seed)
    elif name=='SAC-Gen-Set':model=SetResidual(epochs=epochs,seed=seed).fit(sources,origin)
    elif name in {'SAC-Gen-GB','SAC-Gen-Diffusion'}:
        from ..sac_gen import SACGen
        from .. import pulseicp_bridge as bridge
        settings={**bridge.configuration()['model'],'seed':seed,'crossfit_grouping':'recipe',
                  'epochs_baseline':epochs,'epochs_diffusion':epochs,'refine_xrd':True}
        model=SACGen('Gradient boosting residual' if name.endswith('GB') else 'Diffusion Transformer',settings).fit(sources,origin,{})
    else:raise ValueError('Select a supported candidate family.')
    base=Path(root)/'discovery_models'/origin.lower();base.mkdir(parents=True,exist_ok=True)
    run_id=stamp().replace(':','').replace('.','')+'_'+uuid4().hex[:6]
    model.save(base/run_id)
    reloaded=type(model).load(base/run_id) if name in {'SAC-Gen-Set','Calibrated discovery ensemble'} else type(model).load(base/run_id,samples)
    first=model.predict([query(sources[0])])[0];second=reloaded.predict([query(sources[0])])[0]
    import numpy as np
    if not np.isclose(first['predicted_total_wt_percent'],second['predicted_total_wt_percent']):raise ValueError('Saved-model forecast differs.')
    path=base/'index.json';index=json.loads(path.read_text()) if path.exists() else {}
    index[name]={'name':name,'run_id':run_id,'model_id':model.model_id,'training_fingerprint':digest(sources),
                 'created':stamp(),'activation':'Discovery candidate only; primary deployment unchanged'}
    temp=base/'index.pending.json';temp.write_text(json.dumps(index,indent=2),encoding='utf-8');temp.replace(path)
    return index[name]


def pending_from_proposal(store,ledger,proposal_id):
    from ..domain import validate_sample
    proposal=next((p for p in ledger.records('proposal') if p['id']==proposal_id),None)
    if proposal is None:raise ValueError('Unknown discovery proposal.')
    samples=store.all('sample')
    links=[r for r in ledger.records('link') if r['proposal_id']==proposal_id]
    if links:
        existing={identity(s):s for s in samples}
        if any(k not in existing for k in links[0]['record_keys']):raise ValueError('A linked pending record was deleted; create a new proposal.')
        return [existing[k]['sample_id'] for k in links[0]['record_keys']]
    recovered=sorted([s for s in samples if s.get('discovery_proposal_id')==proposal_id],key=lambda s:s['discovery_slot'])
    if recovered:
        ledger.link(proposal_id,recovered);return [s['sample_id'] for s in recovered]
    frozen=next(r for r in ledger.records('round') if r['id']==proposal['round_id'])
    if digest(samples)!=frozen['dataset_fingerprint']:raise ValueError('Experiment records changed after the round was frozen; generate a new proposal.')
    ids=store.next_sample_ids(len(proposal['recipes']));created=[]
    for slot,(ident,recipe,forecast) in enumerate(zip(ids,proposal['recipes'],proposal['predictions'])):
        sample=copy.deepcopy(recipe)
        sample.update(sample_id=ident,record_uid=uuid4().hex,batch_id='UNKNOWN',temperature_C=None,electrical_input_kind='SETTING',
                      sample_condition='NOT_RECORDED',icp=[],icp_review={'state':'PENDING'},xrd={'state':'PENDING','mode':'MANUAL'},
                      stem={'state':'NOT_TESTED'},result_reported_at=None,discovery_proposal_id=proposal_id,discovery_slot=slot)
        sample['manual_prediction']={'created':proposal['created'],'model_name':forecast.get('model_name'),
                                     'model_id':forecast.get('model_id'),'sac_gen_prediction':forecast,
                                     'p_xrd_pass':forecast.get('p_xrd_pass'),'p_screen_pass':None,'prediction_stale':False,
                                     'recipe':recipe,'prediction_mode':'DISCOVERY_PRE_RESULT_FORECAST'}
        created.append(validate_sample(sample))
    store.import_bundle(created,[])
    persisted=[store.get('sample',s['sample_id']) for s in created]
    ledger.link(proposal_id,persisted)
    return ids
