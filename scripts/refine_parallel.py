"""Resume outer folds in isolated processes, then run selection and deployment."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_current_sac_gen import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();output=args.output.resolve()
    manifest=json.loads((output/'refinement_configuration.json').read_text())
    app=Path(__file__).resolve().parents[1];started=time.perf_counter()
    pending=[(r,f) for r in range(1,len(manifest['outer_seeds'])+1) for f in range(1,6)
        if not ((output/f'repeat_{r}_fold_{f}.csv').exists() and (output/f'repeat_{r}_fold_{f}.json').exists())]
    total=len(manifest['outer_seeds'])*5
    def run(pair):
        r,f=pair;log=output/f'repeat_{r}_fold_{f}.worker.log'
        with log.open('w',encoding='utf-8') as handle:
            result=subprocess.run([sys.executable,'-X','utf8',str(app/'scripts/refine_fold_worker.py'),
                '--output',str(output),'--repeat',str(r),'--fold',str(f)],cwd=app,stdout=handle,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
        return pair,result.returncode
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,pair):pair for pair in pending};failed=[]
        while futures:
            done,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                pair,code=future.result();del futures[future]
                if code:failed.append(pair)
            active=[]
            for pair in futures.values():
                path=output/f'repeat_{pair[0]}_fold_{pair[1]}.progress.json'
                try:state=json.loads(path.read_text())
                except (OSError,ValueError):continue
                if not state.get('completed'):active.append(state)
            finished=sum((output/f'repeat_{r}_fold_{f}.csv').exists() and (output/f'repeat_{r}_fold_{f}.json').exists()
                for r in range(1,len(manifest['outer_seeds'])+1) for f in range(1,6))
            status={'stage':'Parallel grouped validation','completed_folds':finished,'total_folds':total,
                'workers':args.workers,'active':active,'failed_folds':failed,'elapsed_seconds':round(time.perf_counter()-started,1)}
            dump(output/'parallel_progress.json',status)
            print(json.dumps({'completed_folds':finished,'total_folds':total,'failed_folds':failed}),flush=True)
        if failed:
            raise RuntimeError('Some isolated folds failed; inspect their worker logs and resume with fewer workers if memory-limited: '+str(failed))
    print('All outer folds complete; collecting results and fitting the final candidate.',flush=True)
    code=subprocess.call([sys.executable,'-X','utf8',str(app/'scripts/refine_sac_gen.py'),'--output',str(output),
        '--seeds',','.join(map(str,manifest['outer_seeds']))],cwd=app,creationflags=subprocess.CREATE_NO_WINDOW)
    if code:raise RuntimeError('Final candidate/selection run failed; completed validation folds were preserved.')
    dump(output/'parallel_progress.json',{'completed':True,'stage':'Refinement complete','elapsed_seconds':time.perf_counter()-started})


if __name__=='__main__':main()
