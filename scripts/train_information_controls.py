"""Matched recipe-mask comparison and full-data training of information controls."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_current_sac_gen import read_source,dump,markdown_table
from scripts.refine_sac_gen import point_rows,summarize
from sacfactory.compute_runtime import execution
from sacfactory.data_only import DataOnlyModel,BACKENDS
from sacfactory.planning import fingerprint
from sacfactory.validation_groups import RANDOM_RECIPE_CV,assert_mask
from sacfactory.weighted_validation import collect,split_records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hybrid-output',type=Path,required=True)
    parser.add_argument('--science',choices=['none','fixed'],default='none')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    hybrid=args.hybrid_output.resolve();app=Path(__file__).resolve().parents[1];root=app/'data'
    config=json.loads((hybrid/'refinement_configuration.json').read_text());seeds=config['outer_seeds']
    samples,methods=read_source(root);source_fingerprint=fingerprint(samples)
    if source_fingerprint!=config['source_fingerprint']:raise ValueError('Control and hybrid data snapshots differ.')
    records,audit=collect(samples,methods,'REAL','periodic',use_thermal=False)
    targets=sorted({e for r in records for e in r['sample']['elements']})
    manifest={'source_fingerprint':source_fingerprint,'methods_fingerprint':fingerprint(methods),'seeds':seeds,
        'backends':BACKENDS,'science':args.science,'folds_per_repeat':5,
        'data_only_source_sha256':hashlib.sha256((app/'sacfactory/data_only.py').read_bytes()).hexdigest(),
        'selection':'Lowest sum of mean ICP MAE rank and XRD Brier rank; ties prefer balanced accuracy',
        'eligibility':'All valid quantified experimental ICP; no science-derived mass-bound filter',
        'scope':'Development model comparison; not a BO acquisition-policy comparison'}
    path=output/'configuration.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Control resume settings differ; use a separate directory.')
    dump(path,manifest);started=time.perf_counter();state={}
    def progress(stage,**extra):
        value={**state,'stage':stage,'elapsed_seconds':round(time.perf_counter()-started,1),**extra}
        dump(output/'progress.json',value);print(json.dumps(value),flush=True)
    points=[];partitions=[]
    with execution({'device':'cpu','threads':4,'cpu_jobs':4}):
        for backend in BACKENDS:
            for repeat,seed in enumerate(seeds):
                splits=split_records(records,samples,'REAL',targets,RANDOM_RECIPE_CV,5,seed=seed)
                for index,(train,test,fold,_) in enumerate(splits,1):
                    state.update(model='Data-only '+backend,repeat=repeat+1,fold=index)
                    prefix=output/f'{backend.lower().replace(" ","_")}_repeat_{repeat+1}_fold_{index}'
                    if prefix.with_suffix('.csv').exists() and prefix.with_suffix('.json').exists():
                        frame=pd.read_csv(prefix.with_suffix('.csv'));detail=json.loads(prefix.with_suffix('.json').read_text())
                    else:
                        progress('Fitting experimental-only model')
                        assert_mask([r['sample'] for r in train],[r['sample'] for r in test],True)
                        model=DataOnlyModel(backend,seed).fit_records(train,'REAL')
                        predictions=model.predict([r['sample'] for r in test])
                        frame=pd.DataFrame(point_rows(test,predictions));frame['model']='Data-only '+backend;frame['repeat']=repeat;frame['fold']=fold
                        detail={'model':backend,'repeat':repeat,'fold':fold,'train_ids':[r['sample']['sample_id'] for r in train],
                            'test_ids':[r['sample']['sample_id'] for r in test],'model_id':model.model_id,'info':model.info}
                        frame.to_csv(prefix.with_suffix('.csv'),index=False);dump(prefix.with_suffix('.json'),detail)
                    points.append(frame);partitions.append(detail)
                    current,_=read_source(root)
                    if fingerprint(current)!=source_fingerprint:raise RuntimeError('Source experiments changed during control training.')
                    progress('Control outer fold complete')
        all_points=pd.concat(points,ignore_index=True)
        all_points.to_csv(output/'data_only_out_of_fold_predictions.csv',index=False)
        summaries=pd.DataFrame([{'model':name,'repeat':repeat,**summarize(p)} for (name,repeat),p in all_points.groupby(['model','repeat'])])
        summaries.to_csv(output/'data_only_repeat_metrics.csv',index=False)
        means=summaries.groupby('model').mean(numeric_only=True)
        means['selection_rank']=means.ICP_MAE.rank()+means.XRD_Brier.rank()
        chosen=means.sort_values(['selection_rank','balanced_accuracy'],ascending=[True,False]).index[0]
        means.reset_index().to_csv(output/'data_only_model_comparison.csv',index=False)
        backend=chosen.removeprefix('Data-only ');progress('Fitting selected full-data experimental-only model',selected_backend=backend)
        selected=DataOnlyModel(backend,42).fit_records(records,'REAL')
        name='controls_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        base=root/'comparison_controls'/'real';folder=base/name;folder.mkdir(parents=True,exist_ok=False)
        selected.save(folder/'data_only');loaded=DataOnlyModel.load(folder/'data_only')
        if loaded.model_id!=selected.model_id:raise RuntimeError('Data-only reload identity mismatch.')
        probe=loaded.predict([samples[0],samples[-1]])
        dump(output/'data_only_deployment_smoke.json',probe)
    registry={'run_id':name,'source_fingerprint':source_fingerprint,'data_only':{'backend':backend,'model_id':selected.model_id,
        'checkpoint':'data_only','training_specimens':len(selected.training_ids)},'science':None}
    if args.science=='fixed':
        from sacfactory.science_only import SciencePriorModel
        progress('Evaluating fixed science prior; no experimental-label fitting')
        scientific=SciencePriorModel('REAL');predictions=scientific.predict([r['sample'] for r in records])
        frame=pd.DataFrame(point_rows(records,predictions));frame['fold']='FIXED_PRIOR';frame['repeat']=0;frame['model']='Science prior (fixed)'
        frame.to_csv(output/'science_prior_predictions.csv',index=False)
        scientific.save(folder/'science_prior')
        registry['science']={'model_id':scientific.model_id,'checkpoint':'science_prior','fitted_to_current_experiments':False}
        pd.DataFrame([summarize(frame)]).to_csv(output/'science_prior_metrics.csv',index=False)
    dump(output/'validation_methods.json',{'configuration':manifest,'partitions':partitions})
    current,_=read_source(root)
    if fingerprint(current)!=source_fingerprint:raise RuntimeError('Experiments changed before control registration.')
    dump(folder/'registry.json',registry);temporary=base/'latest.pending.json';dump(temporary,registry);temporary.replace(base/'latest.json')
    dump(output/'trained_controls.json',registry)
    comparisons=[{'model':name,'scope':'Mean of three matched repeats',**row.to_dict()} for name,row in means.iterrows()]
    if (hybrid/'repeat_metrics.csv').exists():
        h=pd.read_csv(hybrid/'repeat_metrics.csv').mean(numeric_only=True)
        comparisons.append({'model':'SAC-Gen refined','scope':'Mean of three matched repeats',**h.to_dict()})
    if args.science=='fixed':comparisons.append({'model':'Science prior (fixed)','scope':'Fixed evaluation; no training',**summarize(frame)})
    comparison=pd.DataFrame(comparisons);comparison.to_csv(output/'information_model_comparison.csv',index=False)
    report=['# SAC-Gen information-control models','',markdown_table(comparison),'',
        f'Selected experimental-only model: **{backend}**. Checkpoint: `{folder / "data_only"}`.','',
        'The data-only inputs are nominal element identity, concentrations, controlled cutoff, pulse count, PAN mass and DMF volume. '
        'No elemental property tables, ionic states, thermal references, mixing descriptors, retention prior or mass-conserving decoder enter its predictions. '
        'Every selected backend receives the same three recipe-grouped five-fold outer masks. Classifier calibration and thresholds '
        'are fitted using training-only cross-fitted raw-control scores. Predictions are not clipped to physical bounds.','',
        'The data-only regression uses all valid measured targets, including the composition excluded by the SAC-Gen decoder mass-bound assumption. '
        'This difference is disclosed: the comparison evaluates complete operational information/constraint packages, '
        'not an otherwise identical feature-only ablation. Each repeat counts physical specimens once; element-level rows are not independent experiments.','',
        'Backend choice uses development cross-validation. The reported selected-backend score is not an independent confirmation '
        'after model selection. Gaussian-process prediction is evaluated here; Bayesian-optimization acquisition efficiency requires '
        'a separate sequential replay or prospective experimental comparison.','']
    if args.science=='fixed':report += ['The science control is an unfitted mass/thermal/mixing/kinetic prior. It retains user-reported '
        'empirical carbonization/pulse-loss/Fe-retention anchors. It is not a first-principles predictor, and its '
        'probabilities and sensitivity ranges are not calibrated using current ICP/XRD outcomes.','']
    report += ['Original observations and the active SAC-Gen deployment were not changed by control-model training.']
    (output/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    progress('Information controls complete',completed=True,registry=registry)


if __name__=='__main__':main()
