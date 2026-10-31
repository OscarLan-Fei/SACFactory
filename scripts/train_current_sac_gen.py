"""Validate current REAL records and atomically activate a full-data SAC-Gen fit."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sacfactory import pulseicp_bridge as bridge
from sacfactory.compute_runtime import execution, hardware
from sacfactory.model_benchmarks import compare
from sacfactory.planning import fingerprint
from sacfactory.sac_gen import latest, train_deployment
from sacfactory.validation_groups import RANDOM_RECIPE_CV, assert_mask
from sacfactory.weighted_validation import collect


def clean(value):
    if isinstance(value, dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [clean(v) for v in value]
    if isinstance(value, np.generic):return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):return None
    return value


def dump(path, value):
    path.write_text(json.dumps(clean(value),indent=2,allow_nan=False),encoding='utf-8')


def markdown_table(frame):
    """Render small audit tables without optional pandas formatting packages."""
    escape=lambda v:str(v).replace('|','\\|').replace('\n',' ')
    rows=['| '+' | '.join(map(escape,frame.columns))+' |',
          '| '+' | '.join(['---']*len(frame.columns))+' |']
    rows.extend('| '+' | '.join(map(escape,row))+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(rows)


def read_source(root):
    with closing(sqlite3.connect((root/'real.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)) as db:
        samples=[json.loads(r[0]) for r in db.execute("SELECT payload FROM items WHERE kind='sample'")]
        methods={r[0]:json.loads(r[1]) for r in db.execute("SELECT id,payload FROM items WHERE kind='method'")}
    return samples,methods


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--members',type=int,default=3)
    parser.add_argument('--epochs',type=int,default=100)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/'data'
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();phase={'name':'validation','fold':0,'folds':5}
    progress_path=output/'progress.json'
    def progress(event):
        if 'fold' in event:phase.update(fold=event['fold'],folds=event['folds'])
        state={**phase,**event,'elapsed_seconds':round(time.perf_counter()-started,1),
               'updated_utc':datetime.now(timezone.utc).isoformat()}
        dump(progress_path,state)
        if 'epoch' not in event or event['epoch']%25==0:
            print(json.dumps(clean(state)),flush=True)
    samples,methods=read_source(root)
    source_fingerprint=fingerprint(samples)
    records,audit=collect(samples,methods,'REAL','chemistry_pulseicp')
    audit.to_csv(output/'eligibility.csv',index=False)
    settings={**bridge.configuration()['model'],'members':args.members,
        'epochs_baseline':args.epochs,'epochs_diffusion':args.epochs,'diffusion_steps':30,
        'draws_per_member':32,'token_width':32,'attention_heads':4,'attention_layers':2,
        'seed':42,'crossfit_grouping':'recipe',
        'split':'Five random recipe-grouped outer folds; three recipe-grouped inner folds',
        'status':'Fixed training budget; not tuned using outer outcomes'}
    config={'origin':'REAL','family':'Diffusion Transformer','profile':'chemistry_pulseicp',
        'protocol':RANDOM_RECIPE_CV,'folds':5,'settings':settings,'hardware':hardware(),
        'source_fingerprint':source_fingerprint,'source_specimens':len(samples),
        'eligible_control_specimens':len(records),'created_utc':datetime.now(timezone.utc).isoformat()}
    dump(output/'training_configuration.json',config)
    targets=sorted({e for r in records for e in r['sample']['elements']})
    with execution({'device':'cuda','threads':4,'cpu_jobs':8},progress):
        points,summary,info=compare(samples,methods,'REAL',targets,models=['SAC-Gen'],
            sac_family='Diffusion Transformer',sac_settings=settings,protocol=RANDOM_RECIPE_CV,
            max_folds=5,seed=42,scale=.25,agreement_weight=.6)
        points.to_csv(output/'out_of_fold_predictions.csv',index=False)
        summary.to_csv(output/'validation_metrics.csv',index=False)
        dump(output/'validation_methods.json',info)
        by_id={s['sample_id']:s for s in samples};held=[]
        for fold in info['partitions']:
            assert_mask([by_id[i] for i in fold['train_ids']], [by_id[i] for i in fold['test_ids']],True)
            if fold['models']['SAC-Gen'].get('unavailable_reason'):
                raise RuntimeError(f"Validation fold failed: {fold['models']['SAC-Gen']['unavailable_reason']}")
            held.extend(fold['test_ids'])
        if sorted(held)!=sorted(r['sample']['sample_id'] for r in records):
            raise RuntimeError('Not every eligible control specimen was held out exactly once.')
        if set(points.sample_id)!=set(held):
            raise RuntimeError('Missing specimen predictions. Previous deployment remains active.')
        current,_=read_source(root)
        if fingerprint(current)!=source_fingerprint:
            raise RuntimeError('Source data changed during validation; previous deployment remains active.')
        phase.update(name='deployment',fold=0,folds=1)
        progress({'stage':'Training full-data deployment model'})
        pointer=train_deployment(samples,'REAL',root,'Diffusion Transformer',settings,methods)
        deployed=latest(root,'REAL',samples)
        smoke=[next(r['sample'] for r in records if len(r['sample']['elements'])==n)
               for n in (1,max(len(r['sample']['elements']) for r in records))]
        smoke_predictions=deployed.predict(smoke)
        dump(output/'deployment_smoke_predictions.json',smoke_predictions)
    model_dir=root/'sac_gen'/'real'/pointer['run_id']
    validation_dir=model_dir/'validation';validation_dir.mkdir()
    loading_audit=bridge.prepare(deployed.training_samples,'REAL',deployed.settings,deployed.loading.physics).audit
    loading_audit.to_csv(output/'ICP_decoder_eligibility.csv',index=False)
    for name in ('out_of_fold_predictions.csv','validation_metrics.csv','validation_methods.json','training_configuration.json','eligibility.csv'):
        shutil.copy2(output/name,validation_dir/name)
    shutil.copy2(output/'ICP_decoder_eligibility.csv',validation_dir/'ICP_decoder_eligibility.csv')
    pointer.update(validation_subdirectory='validation',validation_protocol=RANDOM_RECIPE_CV,
        validation_specimens=len(set(held)),source_specimens=len(samples),
        validation_settings={'folds':5,'seed':42,'icp_weight':.6,'icp_error_scale_wt_percent':.25},
        created_utc=datetime.now(timezone.utc).isoformat())
    dump(model_dir/'training_run.json',pointer)
    pointer_path=model_dir.parent/'latest.json'
    # Do not replace a deployment selected concurrently by another app session.
    if json.loads(pointer_path.read_text())['model_id']!=pointer['model_id']:
        raise RuntimeError('Another deployment became active; this training result was saved but not reactivated.')
    temporary=pointer_path.with_suffix('.pending.json');dump(temporary,pointer);temporary.replace(pointer_path)
    dump(output/'deployment.json',pointer)
    current,_=read_source(root)
    if fingerprint(current)!=source_fingerprint:
        raise RuntimeError('Source data changed; deployment will be rejected by the app source-integrity guard.')
    icp_specimens=len(set(deployed.loading.training_ids))
    excluded=audit.loc[(audit.scope=='Controls')&~audit.eligible,['sample_id','reason']]
    excluded_icp=loading_audit.loc[~loading_audit.eligible,['sample_id','exclusion_reason']].drop_duplicates()
    report=[
        '# SAC-Gen training and deployment', '',
        f"Active model: `{pointer['model_id']}`. Architecture: Diffusion Transformer.",
        f"Checkpoint: `{model_dir}`.", '',
        f'{len(samples)} source specimens; {len(records)} eligible control specimens; '
        f'{icp_specimens} quantified specimens used by the ICP decoder; '
        f"{deployed.xrd_info['n']} recorded XRD labels ({deployed.xrd_info['pass_n']} PASS, {deployed.xrd_info['fail_n']} FAIL).", '',
        '## Validation', '',
        'Five deterministic random recipe-grouped outer folds. Each eligible specimen is hidden once. '
        'All element measurements and XRD outcomes of a specimen remain together. Exact recipe repeats remain together. '
        'Administrative batches may overlap between pools. This measures recipe interpolation within the observed '
        'process domain, not independent unseen-element, mixture-transfer or prospective performance.', '',
        markdown_table(summary), '',
        'ICP metrics give equal weight to physical specimens, then equal weight to members within a specimen. '
        'Missing ICP and unrecorded XRD are excluded only from the corresponding metric. '
        'XRD probabilities are trained but not independently calibrated. Agreement uses 60% continuous ICP '
        'and 40% XRD with a fixed 0.25 wt% ICP error scale; it is not percent classification accuracy.', '',
        '## Training', '',
        f'{args.members} members, {args.epochs} epochs per member, 30 diffusion steps, 32 draws per member, '
        'token width 32, four attention heads, two attention layers, seed 42, CUDA. '
        'No settings were selected by maximizing outer-fold fit. The XRD logistic head uses chemistry, '
        'a mechanistic prior and three-fold cross-fitted ICP summaries. Fixed physics references are modeling assumptions. '
        'The full-data deployment checkpoint is distinct from the validation fits.', '',
        '## Excluded controls', '',markdown_table(excluded), '',
        '## ICP decoder exclusions', '',markdown_table(excluded_icp), '',
        'The decoder requires complete quantified compositions. A conflict with an assumed feed/matrix '
        'mass bound is not proof of an erroneous measurement; check preparation metadata and the physical '
        'assumptions before relaxing the bound. Recorded values are never clipped. Quantified held-out '
        'observations remain in the validation metrics even if the decoder would exclude them from fitting.', '',
        'No measurements or batch identifiers were changed. The five missing controlled cutoff values were not imputed. '
        'SAC-Gen predicts elemental ICP loading and XRD PASS probability; these outputs do not establish atomic dispersion.', '',
        '## Application integration', '',
        'Next batch and pending-sample reforecasting read the active checkpoint via latest.json. '
        'Saved validation predictions and metrics are bundled with the checkpoint. Source edits or deletions '
        'invalidate the checkpoint for forecasting until retraining. New specimens can be added without overwriting the checkpoint.', '',
        f'Elapsed time: {time.perf_counter()-started:.1f} seconds.']
    (output/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    dump(progress_path,{'completed':True,'model_id':pointer['model_id'],'elapsed_seconds':time.perf_counter()-started})
    print(json.dumps({'completed':True,'deployment':pointer,'summary':clean(summary.to_dict('records'))}),flush=True)


if __name__=='__main__':main()
