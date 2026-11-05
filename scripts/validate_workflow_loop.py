"""Persist a reproducible simulator loop and verify live REAL data protection."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP))
from sacfactory.storage import Store
from sacfactory.workflow_service import WorkflowService,simulation_template
from sacfactory.discovery.common import digest,outcomes
from sacfactory import __version__


def protected_source():
    path=APP/'data/real.sqlite3'
    if path.exists():
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            fingerprint=digest(db.execute('SELECT kind,id,payload FROM items ORDER BY kind,id').fetchall())
    else:fingerprint=None
    pointers={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [APP/'data/sac_gen/real/latest.json',APP/'data/sac_gen/real/variants.json'] if p.exists()}
    return {'real_items_fingerprint':fingerprint,'primary_pointer_hashes':pointers}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);before=protected_source();start=time.perf_counter()
    root=out/'simulation_workspace';service=WorkflowService(Store(root,'DEMO'));service.devices.defaults()
    state_path=out/'validation_state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
    if not state.get('run_id'):
        definition=service.db.save(simulation_template());state['run_id']=service.start(definition['id'])['id'];state_path.write_text(json.dumps(state))
        (out/'simulation_workflow.json').write_text(json.dumps(definition['spec'],indent=2),encoding='utf-8')
    run_id=state['run_id']
    def progress(summary):
        print(f"{summary['completed_steps']}/{summary['total_steps']} {summary['state']}",flush=True)
    run=service.db.run_ready(run_id,service.execute,progress=progress)
    if run['state']!='COMPLETED':raise RuntimeError(json.dumps(service.db.summary(run_id)))
    steps=service.db.steps(run_id);by_id={s['id']:s for s in steps};samples=service.store.all('sample')
    assert len(samples)==32 and all(s['origin']=='DEMO' and s['simulation']['synthetic'] for s in samples)
    assert all(outcomes(s)[0] and outcomes(s)[1] is not None for s in samples)
    assert by_id['next_design']['output']['model_id']==by_id['update']['output']['candidate']['model_id']
    acquisition=by_id['measurement']['output']['acquisition']
    for measurement in acquisition['measurements']:
        balance=measurement['mass_balance']
        assert all(0<=mass<=balance['feed_metal_g'][e] for e,mass in balance['retained_metal_g'].items())
        assert abs(sum(float(a['value']) for a in measurement['icp'])-100*sum(balance['retained_metal_g'].values())/balance['total_product_g'])<1e-8
    # Explicit pre-execution failure and retry, using one job ID.
    recovery=WorkflowService(service.store,selected_ids=[samples[0]['sample_id']])
    if not recovery.db.get('devices','recovery_heater'):
        recovery.devices.register({'id':'recovery_heater','name':'Recovery validation simulator','mode':'SIMULATOR','capabilities':['synthesize'],'fail_once':True})
    if not state.get('recovery_run_id'):
        spec={'name':'Explicit device recovery validation','steps':[{'id':'synthesis','operation':'device.synthesize',
            'parameters':{'device_id':'recovery_heater','sample_ids':[samples[0]['sample_id']]}}]}
        state['recovery_run_id']=recovery.start(recovery.db.save(spec)['id'])['id'];state_path.write_text(json.dumps(state))
    recovery_id=state['recovery_run_id'];recovery.db.run_ready(recovery_id,recovery.execute)
    if recovery.db.get('runs',recovery_id)['state']=='FAILED':
        (out/'recovery_before_retry.json').write_text(json.dumps(recovery.db.summary(recovery_id),indent=2))
        recovery.db.control(recovery_id,'retry');recovery.db.run_ready(recovery_id,recovery.execute)
    assert recovery.db.get('runs',recovery_id)['state']=='COMPLETED'
    jobs=[j for j in recovery.db.all('jobs') if j['device_id']=='recovery_heater']
    assert len(jobs)==1 and jobs[0]['attempts']==2
    after=protected_source();assert before==after,'Live REAL data or primary pointers changed during validation.'
    report={'status':'PASS','software_version':__version__,'completed_steps':len(steps),'simulated_specimens':len(samples),
        'synthetic_PASS':sum(outcomes(s)[1]==1 for s in samples),'synthetic_FAIL':sum(outcomes(s)[1]==0 for s in samples),
        'characterization_files':len(service.ledger.records('characterization')),'reviewed_assessments':len(service.ledger.records('characterization_assessment')),
        'main_run_id':run_id,'recovery_run_id':recovery_id,'recovery_job_id':jobs[0]['id'],'recovery_attempts':2,
        'validation_folds':len(by_id['validation']['output']['folds']),'updated_model_id':by_id['update']['output']['candidate']['model_id'],
        'next_design_uses_updated_model':True,'mass_balance_checks':'PASS','live_REAL_items_unchanged':True,'primary_pointers_unchanged':True,
        'protected_source':before,'elapsed_seconds':round(time.perf_counter()-start,2),
        'interpretation':'Simulator integration and recovery evidence only; not real scientific performance.'}
    (out/'loop_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'simulation_run.json').write_text(json.dumps({'run':service.db.get('runs',run_id),'steps':steps},indent=2),encoding='utf-8')
    (out/'recovery_after_retry.json').write_text(json.dumps(recovery.db.summary(recovery_id),indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
