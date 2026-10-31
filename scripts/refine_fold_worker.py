"""Isolated CUDA worker for one declared outer refinement fold."""
import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_current_sac_gen import read_source,dump
from scripts.refine_sac_gen import point_rows
from sacfactory.compute_runtime import execution
from sacfactory.planning import fingerprint
from sacfactory.sac_gen import SACGen
from sacfactory.validation_groups import RANDOM_RECIPE_CV,assert_mask
from sacfactory.weighted_validation import collect,split_records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeat',type=int,required=True)
    parser.add_argument('--fold',type=int,required=True)
    args=parser.parse_args();output=args.output.resolve()
    manifest=json.loads((output/'refinement_configuration.json').read_text())
    app=Path(__file__).resolve().parents[1]
    samples,methods=read_source(app/'data')
    if fingerprint(samples)!=manifest['source_fingerprint'] or fingerprint(methods)!=manifest['methods_fingerprint']:
        raise ValueError('Worker inputs differ from the declared refinement snapshot.')
    import hashlib
    for name,expected in manifest['source_hashes'].items():
        if hashlib.sha256((app/'sacfactory'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Refinement source changed: '+name)
    records,_=collect(samples,methods,'REAL','chemistry_pulseicp')
    targets=sorted({e for r in records for e in r['sample']['elements']})
    seed=manifest['outer_seeds'][args.repeat-1]
    splits=split_records(records,samples,'REAL',targets,RANDOM_RECIPE_CV,5,seed=seed)
    train,test,fold,_=splits[args.fold-1]
    assert_mask([r['sample'] for r in train],[r['sample'] for r in test],True)
    prefix=output/f'repeat_{args.repeat}_fold_{args.fold}';started=time.perf_counter()
    def progress(event):
        value={**event,'repeat':args.repeat,'outer_fold':args.fold,'elapsed_seconds':round(time.perf_counter()-started,1)}
        dump(prefix.with_suffix('.progress.json'),value)
        if 'epoch' not in event or event['epoch']%100==0:print(json.dumps(value),flush=True)
    with execution({'device':'cuda','threads':4,'cpu_jobs':4},progress):
        model=SACGen('Diffusion Transformer',{**manifest['settings'],'seed':seed,'crossfit_grouping':'recipe'}).fit([r['sample'] for r in train],'REAL',methods)
        predictions=model.predict([r['sample'] for r in test])
    part=pd.DataFrame(point_rows(test,predictions));part['fold']=fold;part['repeat']=args.repeat-1;part['model']='SAC-Gen refined'
    if part.sample_id.nunique()!=len(test) or part.predicted_w_wt_percent.isna().any():
        raise ValueError('Incomplete outer predictions.')
    current,_=read_source(app/'data')
    if fingerprint(current)!=manifest['source_fingerprint']:raise ValueError('Experiments changed during the fold.')
    detail={'repeat':args.repeat-1,'fold':fold,'seed':seed,'train_ids':[r['sample']['sample_id'] for r in train],
        'test_ids':[r['sample']['sample_id'] for r in test],'model_id':model.model_id,
        'settings':model.settings,'XRD':model.xrd_info,'loading_tuning':model.loading.tuning,
        'execution':'Isolated single-process CUDA worker','elapsed_seconds':time.perf_counter()-started}
    part.to_csv(prefix.with_suffix('.csv'),index=False);dump(prefix.with_suffix('.json'),detail)
    progress({'stage':'Outer fold complete','completed':True})


if __name__=='__main__':main()
