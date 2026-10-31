"""Run bounded nested refinement, repeated grouped validation and audited selection."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,average_precision_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_current_sac_gen import read_source,dump,markdown_table
from sacfactory.compute_runtime import execution
from sacfactory.model_benchmarks import metrics
from sacfactory.planning import fingerprint
from sacfactory.sac_gen import SACGen
from sacfactory.validation_groups import RANDOM_RECIPE_CV,assert_mask
from sacfactory.weighted_validation import collect,split_records


def point_rows(records,predictions):
    rows=[]
    for r,p in zip(records,predictions):
        sample=r['sample'];observed=r['assays'];predicted={v['element']:v['predicted_mean_w_wt_percent'] for v in p['icp_predictions']}
        total=sum(observed[e] for e in sample['elements']) if all(e in observed for e in sample['elements']) else None
        for e in sample['elements']:
            rows.append({'sample_id':sample['sample_id'],'batch_id':sample['batch_id'],'element':e,
                'composition':'+'.join(sample['elements']),'observed_w_wt_percent':observed.get(e),
                'predicted_w_wt_percent':predicted.get(e),'observed_total_w_wt_percent':total,
                'predicted_total_w_wt_percent':p['predicted_total_wt_percent'],
                'xrd_pass_observed':r['outcomes']['xrd_pass'],'xrd_pass_predicted':p['p_xrd_pass'],
                'xrd_pass_threshold':p['xrd_pass_threshold'],'predicted_xrd_state':p['predicted_xrd_state']})
    return rows


def summarize(points):
    result=metrics(points,.25,.6)
    b=points.drop_duplicates(['repeat','fold','sample_id']).dropna(subset=['xrd_pass_observed','xrd_pass_predicted'])
    if len(b):
        y=b.xrd_pass_observed.to_numpy();threshold=b.xrd_pass_threshold.to_numpy();p=b.xrd_pass_predicted.to_numpy()
        result.update(balanced_accuracy=float(balanced_accuracy_score(y,p>=threshold)) if len(set(y))==2 else None,
            FAIL_recall=float(np.mean(p[y==0]<threshold[y==0])) if (y==0).any() else None,
            PASS_recall=float(np.mean(p[y==1]>=threshold[y==1])) if (y==1).any() else None,
            FAIL_average_precision=float(average_precision_score(1-y,1-p)) if len(set(y))==2 else None)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seeds',default='42,1042,2042')
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    app=Path(__file__).resolve().parents[1];root=app/'data';base=root/'sac_gen'/'real'
    samples,methods=read_source(root);source_fingerprint=fingerprint(samples)
    records,audit=collect(samples,methods,'REAL','chemistry_pulseicp')
    targets=sorted({e for r in records for e in r['sample']['elements']})
    seeds=[int(v) for v in args.seeds.split(',')]
    baseline_pointer=json.loads((base/'latest.json').read_text())
    baseline_dir=base/baseline_pointer['run_id']/'validation'
    original=json.loads((baseline_dir/'training_configuration.json').read_text())
    if original['source_fingerprint']!=source_fingerprint:raise ValueError('Baseline validation snapshot does not match current records.')
    baseline=pd.read_csv(baseline_dir/'out_of_fold_predictions.csv');baseline['repeat']=0
    baseline['xrd_pass_threshold']=.5;baseline['model']='SAC-Gen baseline'
    settings={**original['settings'],'training_variant':'validated-token-v1','refine_xrd':True,
        'tuning_max_epochs':400,'tuning_min_epochs':100,'tuning_interval':20,'tuning_patience':3,
        'tuning_draws':16,'candidate_learning_rates':[.0003,.001]}
    source_files=['tuned_token_residual.py','xrd_refinement.py','sac_gen.py']
    manifest={'source_fingerprint':source_fingerprint,'methods_fingerprint':fingerprint(methods),
        'settings':settings,'outer_seeds':seeds,'folds_per_repeat':5,
        'source_hashes':{name:hashlib.sha256((app/'sacfactory'/name).read_bytes()).hexdigest() for name in source_files},
        'acceptance':'Matched repeat 42 and repeated mean: ICP MAE and XRD Brier at most 5% above baseline; balanced accuracy improves at least 0.02',
        'baseline_pointer':baseline_pointer,'selection_status':'Development model selection, not independent confirmation'}
    manifest_path=output/'refinement_configuration.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Resume inputs/settings differ. Use a separate output directory.')
    dump(manifest_path,manifest);audit.to_csv(output/'eligibility.csv',index=False)
    baseline.to_csv(output/'baseline_out_of_fold_predictions.csv',index=False)
    started=time.perf_counter();state={'repeat':0,'outer_fold':0,'outer_folds':5,'repeats':len(seeds)}
    def progress(event):
        value={**state,**event,'elapsed_seconds':round(time.perf_counter()-started,1),'updated_utc':datetime.now(timezone.utc).isoformat()}
        dump(output/'progress.json',value)
        if 'epoch' not in event or event['epoch']%100==0:
            print(json.dumps(value),flush=True)
    all_points=[];partitions=[]
    with execution({'device':'cuda','threads':4,'cpu_jobs':8},progress):
        for repeat,seed in enumerate(seeds):
            splits=split_records(records,samples,'REAL',targets,RANDOM_RECIPE_CV,5,seed=seed)
            for index,(train,test,fold,_) in enumerate(splits,1):
                state.update(repeat=repeat+1,outer_fold=index)
                prefix=output/f'repeat_{repeat+1}_fold_{index}'
                saved_points=prefix.with_suffix('.csv');saved_info=prefix.with_suffix('.json')
                if saved_points.exists() and saved_info.exists():
                    part=pd.read_csv(saved_points);detail=json.loads(saved_info.read_text())
                    progress({'stage':'Reusing completed, input-matched outer fold'})
                else:
                    assert_mask([r['sample'] for r in train],[r['sample'] for r in test],True)
                    progress({'stage':'Fitting refined outer model'})
                    model=SACGen('Diffusion Transformer',{**settings,'seed':seed,'crossfit_grouping':'recipe'}).fit([r['sample'] for r in train],'REAL',methods)
                    predictions=model.predict([r['sample'] for r in test])
                    part=pd.DataFrame(point_rows(test,predictions));part['fold']=fold;part['repeat']=repeat;part['model']='SAC-Gen refined'
                    if part.sample_id.nunique()!=len(test) or part.predicted_w_wt_percent.isna().any():
                        raise RuntimeError('Incomplete outer predictions; baseline remains active.')
                    detail={'repeat':repeat,'fold':fold,'seed':seed,'train_ids':[r['sample']['sample_id'] for r in train],
                        'test_ids':[r['sample']['sample_id'] for r in test],'model_id':model.model_id,
                        'settings':model.settings,'XRD':model.xrd_info,'loading_tuning':model.loading.tuning}
                    part.to_csv(saved_points,index=False);dump(saved_info,detail)
                all_points.append(part);partitions.append(detail)
                current,_=read_source(root)
                if fingerprint(current)!=source_fingerprint:raise RuntimeError('Records changed during refinement; baseline remains active.')
                progress({'stage':'Outer fold complete'})
            frame=pd.concat(all_points,ignore_index=True)
            frame.to_csv(output/'refined_out_of_fold_predictions.csv',index=False)
            summaries=[{'repeat':r,'seed':seeds[r],**summarize(p)} for r,p in frame.groupby('repeat')]
            pd.DataFrame(summaries).to_csv(output/'repeat_metrics.csv',index=False)
        refined=pd.concat(all_points,ignore_index=True)
        baseline_summary=summarize(baseline)
        repeat_summary=pd.read_csv(output/'repeat_metrics.csv')
        first=summarize(refined.loc[refined.repeat==0])
        repeated=repeat_summary.select_dtypes(include='number').mean().to_dict()
        for r,p in refined.groupby('repeat'):
            if p.sample_id.nunique()!=len(records) or (p.groupby('sample_id').fold.nunique()!=1).any():
                raise RuntimeError('Outer coverage check failed.')
        accepted=all(v['ICP_MAE']<=baseline_summary['ICP_MAE']*1.05 and
            v['XRD_Brier']<=baseline_summary['XRD_Brier']*1.05 and
            v['balanced_accuracy']>=baseline_summary['balanced_accuracy']+.02 for v in (first,repeated))
        summary=pd.DataFrame([{'model':'SAC-Gen baseline','scope':'Seed 42',**baseline_summary},
            {'model':'SAC-Gen refined','scope':'Matched seed 42',**first},
            {'model':'SAC-Gen refined','scope':f'Mean of {len(seeds)} repeats',**repeated}])
        summary.to_csv(output/'model_comparison.csv',index=False)
        dump(output/'validation_methods.json',{'manifest':manifest,'partitions':partitions,
            'repeated_metrics_counting':'Each specimen counted once per repeat; repeats are dependent, not additional experiments'})
        state.update(repeat=0,outer_fold=0)
        progress({'stage':'Training full-data refined candidate'})
        candidate=SACGen('Diffusion Transformer',{**settings,'seed':42,'crossfit_grouping':'recipe'}).fit(samples,'REAL',methods)
        name='sac_gen_refined_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:6]
        candidate_dir=base/name;candidate.save(candidate_dir)
        loaded=SACGen.load(candidate_dir,samples)
        probe=loaded.predict([samples[0],samples[-1]])
        if not all(p['predicted_total_wt_percent'] is not None and p['p_xrd_pass'] is not None for p in probe):
            raise RuntimeError('Refined checkpoint reload/forecast failed; baseline remains active.')
        dump(output/'candidate_smoke_predictions.json',probe)
    current,_=read_source(root)
    if fingerprint(current)!=source_fingerprint:raise RuntimeError('Records changed; candidate saved but not activated.')
    pointer={'run_id':name,'model_id':candidate.model_id,'family':'Diffusion Transformer','version':'sac-gen-icp-xrd-v1',
        'training_samples':len(candidate.training_ids),'source_specimens':len(samples),
        'selection':'Declared development-CV multi-objective gate','refinement_variant':'validated-token-v1',
        'validation_protocol':RANDOM_RECIPE_CV,'validation_specimens':len(records),
        'validation_settings':{'folds':5,'seed':42,'icp_weight':.6,'icp_error_scale_wt_percent':.25}}
    # The existing single-repeat viewer displays the matched repeat; all repeats remain downloadable.
    validation=candidate_dir/'validation';validation.mkdir()
    matched=refined.loc[refined.repeat==0].copy();matched['model']='SAC-Gen'
    matched.to_csv(validation/'out_of_fold_predictions.csv',index=False)
    pd.DataFrame([{'model':'SAC-Gen',**first}]).to_csv(validation/'validation_metrics.csv',index=False)
    dump(validation/'validation_methods.json',{'models':['SAC-Gen'],'dataset_fingerprint':source_fingerprint,
        'protocol':RANDOM_RECIPE_CV,'partitions':partitions[:5]})
    dump(validation/'training_configuration.json',{'source_fingerprint':source_fingerprint,'source_specimens':len(samples),
        'eligible_control_specimens':len(records),'settings':candidate.settings,'refinement':manifest})
    shutil.copy2(output/'repeat_metrics.csv',validation/'repeat_metrics.csv')
    pointer['validation_subdirectory']='validation'
    dump(candidate_dir/'training_run.json',pointer)
    if accepted:
        if json.loads((base/'latest.json').read_text())['model_id']!=baseline_pointer['model_id']:
            raise RuntimeError('Active model changed concurrently; candidate saved but not activated.')
        temporary=base/'latest.pending.json';dump(temporary,pointer);temporary.replace(base/'latest.json')
    decision={'accepted_for_deployment':bool(accepted),'candidate':pointer,'baseline_model_id':baseline_pointer['model_id'],
        'active_model_id':json.loads((base/'latest.json').read_text())['model_id'],'elapsed_seconds':time.perf_counter()-started}
    dump(output/'selection.json',decision)
    report=['# SAC-Gen controlled refinement','',markdown_table(summary),'',
        '## Method','',
        'Three five-fold random recipe-grouped repeats. Learning-rate candidates 0.0003 and 0.001; '
        '100-400 epochs with deterministic validation-noise early stopping. Each loading fit makes its own '
        'training-only grouped selection and then refits all supplied training specimens. Scalers and residual '
        'normalizers use only the selection-training subset during parameter search. Final ensemble bootstrap '
        'coverage is selected on batch identifiers alone. Architecture and result-score weights are unchanged.','',
        'XRD regularization is selected within nested head folds. Optional sigmoid calibration and the decision '
        'threshold use cross-fitted training-head scores. Calibration is conditional on training-pool generated '
        'ICP features; the complete pipeline is assessed by the outer folds. ICP prediction intervals remain '
        'uncalibrated. Repeated predictions do not increase the number of physical specimens.','',
        'The previous baseline is compared on the matched seed-42 outer masks. Additional repeats estimate '
        'candidate split sensitivity; they are not additional independent datasets. Acceptance gates are '
        'development model-selection rules, not confirmation on a new independent experiment set.','',
        '## Deployment','',f'Candidate accepted: {accepted}. Active model: `{decision["active_model_id"]}`.',
        f'Candidate checkpoint: `{candidate_dir}`.', '',
        'No experimental observations, XRD labels, batch identifiers or score weights were changed.', '',
        f'Elapsed seconds: {decision["elapsed_seconds"]:.1f}.']
    (output/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    progress({'stage':'Refinement complete','completed':True,**decision})
    print(json.dumps(decision),flush=True)


if __name__=='__main__':main()
