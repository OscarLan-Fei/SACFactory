"""Read-only real-data evaluation of the six Discovery lab workflows."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import pandas as pd

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP))
from sacfactory.discovery.common import digest
from sacfactory.discovery.validation import validate,replay
from sacfactory.discovery.workflows import train_candidate
from sacfactory.compute_runtime import execution


def source():
    with closing(sqlite3.connect((APP/'data/real.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('BEGIN')
        rows=db.execute('SELECT kind,id,payload FROM items ORDER BY kind,id').fetchall()
    return [json.loads(p) for k,_,p in rows if k=='sample'],digest(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--skip-candidates',action='store_true');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    samples,before=source();started=time.perf_counter()
    primary=APP/'data/sac_gen/real/variants.json';pointer=APP/'data/sac_gen/real/latest.json'
    primary_hash={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [primary,pointer] if p.exists()}
    def progress(event):
        row={**event,'elapsed_seconds':round(time.perf_counter()-started,1)}
        print(json.dumps(row),flush=True)
        (out/'progress.json').write_text(json.dumps(row,indent=2))
    with execution({'device':'cpu','threads':4,'cpu_jobs':4}):
        validation=validate(samples,'REAL',args.folds,args.epochs,alpha=.05,include_sac=True,progress=progress)
        (out/'grouped_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
        frame=pd.DataFrame(validation['metrics']);frame.to_csv(out/'fold_metrics.csv',index=False)
        frame.groupby('model').mean(numeric_only=True).drop(columns='fold').to_csv(out/'model_summary.csv')
        pd.DataFrame(validation['predictions']).to_csv(out/'calibrated_predictions.csv',index=False)
        replay_result=replay(samples,'REAL',budget=48,initial=16,step=8,epochs=40,progress=progress)
        (out/'acquisition_replay.json').write_text(json.dumps(replay_result,indent=2),encoding='utf-8')
        pd.DataFrame(replay_result['curves']).to_csv(out/'replay_learning_curves.csv',index=False)
        candidates=[]
        if not args.skip_candidates:
            for name in ['SAC-Gen-Set','Calibrated discovery ensemble']:
                progress({'stage':'Saving verified discovery candidate: '+name})
                candidates.append(train_candidate(APP/'data',samples,'REAL',name,args.epochs))
    after_samples,after=source()
    assert before==after,'Source records changed during validation; do not interpret this as a frozen-source run.'
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in primary_hash.items())
    report={'status':'PASS','source_samples':len(samples),'source_fingerprint':digest(samples),
            'experiment_items_unchanged':before==after,'primary_model_registry_unchanged':True,
            'new_candidates':candidates,'elapsed_seconds':round(time.perf_counter()-started,1),
            'software_version':'discovery-lab-v1','scope':'Development validation and recorded-pool replay, not prospective discovery evidence',
            'settings':{'folds':args.folds,'epochs':args.epochs,'nominal_interval_coverage':.95}}
    (out/'validation_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
