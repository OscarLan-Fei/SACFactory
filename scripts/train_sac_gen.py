"""Train SAC-Gen with unchanged REAL records and protected nested comparisons."""
from pathlib import Path
from datetime import datetime,timezone
import json,sqlite3,sys,time
APP=Path(__file__).resolve().parents[1]
import argparse
parser=argparse.ArgumentParser(description='Nested batch SAC-Gen comparison and deployment refit')
parser.add_argument('--data-root',type=Path,default=APP/'data')
parser.add_argument('--output',type=Path,default=APP/'outputs/sac_gen_training')
parser.add_argument('--epochs',type=int,default=100)
args=parser.parse_args()
sys.path.insert(0,str(APP))
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sacfactory import pulseicp_bridge as bridge
from sacfactory.sac_gen import SACGen,FAMILIES,VERSION
from sacfactory.weighted_validation import weighted_validation,ROTATING_CV
from sacfactory.compute_runtime import execution,hardware

def read():
    with sqlite3.connect((args.data_root.resolve()/'real.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        return db.execute('SELECT kind,id,payload FROM items ORDER BY kind,id').fetchall()

before=read();samples=[json.loads(v) for k,_,v in before if k=='sample'];methods={i:json.loads(v) for k,i,v in before if k=='method'}
targets=list(dict.fromkeys(e for s in samples for e in s['elements']))
OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
settings={**bridge.configuration()['model'],'members':1,'epochs_baseline':args.epochs,'epochs_diffusion':args.epochs,
          'draws_per_member':32,'diffusion_steps':30,'token_width':32,'attention_heads':4,'attention_layers':2}
groups=max(1,(len({s['batch_id'] for s in samples})+4)//5)
comparisons=[];started=time.perf_counter()
for family in FAMILIES:
    def progress(event):
        if 'fold' in event:print(f"{family}: outer fold {event['fold']}/{event['folds']}",flush=True)
    with execution({'device':'auto','threads':4,'cpu_jobs':8},progress):
        points,scores,_,audit,info=weighted_validation(samples,methods,'REAL',targets,protocol=ROTATING_CV,
            batches_per_fold=groups,profile='chemistry_pulseicp',icp_family=family,icp_settings=settings,
            endpoint_weights={'icp':.6,'xrd_pass':.4},delta=.03,sac_gen=True)
    name=family.lower().replace(' ','_')
    points.to_csv(OUT/f'{name}_prediction_pairs.csv',index=False)
    scores.to_csv(OUT/f'{name}_accuracy_scores.csv',index=False)
    (OUT/f'{name}_provenance.json').write_text(json.dumps(info,indent=2,allow_nan=False))
    pairs=points.dropna(subset=['observed_w_wt_percent','predicted_w_wt_percent'])
    xrd=next(r for r in info['outcome_metrics'] if r['endpoint']=='xrd_pass')
    metric={'family':family,'ICP_pairs':len(pairs),'XRD_pairs':xrd['evaluated_samples'],
            'ICP_MAE_wt_percent':mean_absolute_error(pairs.observed_w_wt_percent,pairs.predicted_w_wt_percent),
            'ICP_RMSE_wt_percent':float(mean_squared_error(pairs.observed_w_wt_percent,pairs.predicted_w_wt_percent)**.5),
            'ICP_R_squared':r2_score(pairs.observed_w_wt_percent,pairs.predicted_w_wt_percent),
            'XRD_Brier':xrd['brier'],'XRD_ROC_AUC':xrd['roc_auc'],
            'mean_prediction_accuracy':info['sample_weighted_score'],'complete_accuracy_pairs':info['n_scored_samples']}
    comparisons.append(metric)
    pd.DataFrame(comparisons).to_csv(OUT/'model_comparison.csv',index=False)
    print(json.dumps(metric),flush=True)
    assert before==read(),'Experimental records changed during training; stop deployment.'
best=max(comparisons,key=lambda r:r['mean_prediction_accuracy'] if r['mean_prediction_accuracy'] is not None else -1)
print('Refitting deployment: '+best['family'],flush=True)
with execution({'device':'auto','threads':4,'cpu_jobs':8}):
    model=SACGen(best['family'],settings).fit(samples,'REAL',methods)
    run_id='sac_gen_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base=args.data_root/'sac_gen/real';base.mkdir(parents=True,exist_ok=True)
    destination=base/run_id;model.save(destination)
    loaded=SACGen.load(destination,samples)
    assert len(loaded.predict(samples[:3]))==3
assert before==read()
pointer={'run_id':run_id,'model_id':model.model_id,'version':VERSION,'family':best['family'],
         'selection':'Highest mean nested batch OOF ICP/XRD prediction agreement; development selection, not independent test',
         'training_samples':len(model.training_ids),'comparison':str(OUT/'model_comparison.csv')}
(base/'latest.json').write_text(json.dumps(pointer,indent=2))
summary={'source_unchanged':True,'hardware':hardware(),'source_samples':len(samples),'origin':'REAL',
         'elapsed_seconds':time.perf_counter()-started,'deployment':pointer,'models':comparisons}
(OUT/'training_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2),flush=True)
