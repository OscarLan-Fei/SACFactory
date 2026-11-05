"""Persistent, declarative workflows with dependency references and step receipts."""
import copy
from contextlib import closing
from datetime import datetime,timezone,timedelta
import json
import math
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from .discovery.common import digest,stamp

OPERATIONS={
    'knowledge.search':('Research memory',['query'],['limit','include_history']),
    'inverse.design':('Target-conditioned recipes',[],['elements','bounds_mmol','temperature','pulses','count','batch_size','model','target_total','min_probability','seed','allow_extrapolation','composition_mode']),
    'records.training_pool':('Assemble historical and returned training records',[],['sample_ids','include_history']),
    'campaign.plan':('Freeze forecasts and create pending specimens',['design'],[]),
    'campaign.evaluate':('Evaluate immutable forecasts',[],[]),
    'model.compare':('Compare model opinions',[],['sample_ids','models','recipes']),
    'model.train':('Train a versioned candidate',[],['sample_ids','family','epochs','seed']),
    'validation.run':('Protected recipe-group validation',[],['sample_ids','folds','epochs','include_sac']),
    'hypothesis.register':('Register a one-factor contrast',['sample_id','factor','values','statement'],['model']),
    'characterization.analyze':('Analyze registered raw exports',['measurements'],[]),
    'characterization.fixture_review':('Review DEMO fixture observations',['evidence'],[]),
    'characterization.commit':('Commit previously reviewed evidence',['sample_ids','state','reviewer','criteria'],['override']),
    'device.synthesize':('Submit simulated synthesis',['device_id','sample_ids'],[]),
    'device.characterize':('Acquire simulated measurements',['device_id','synthesis'],['modalities']),
    'manual.checkpoint':('Wait for a documented manual action',['instruction'],['device_id','capability','modalities','equipment_type','sample_ids','approval_required','return_fields','plan']),
}
IDENTIFIER=re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,63}$')


def validate_parameters(operation,parameters,allow_references=True):
    required,optional=OPERATIONS[operation][1:]
    if not isinstance(parameters,dict) or set(required)-set(parameters) or set(parameters)-set(required+optional):raise ValueError('Unexpected or missing parameters for '+operation)
    integers={'epochs':(1,500),'folds':(2,10),'count':(1,512),'batch_size':(1,48),'limit':(1,12),'seed':(0,2**31-1)}
    strings={'query','model','family','factor','statement','instruction','device_id','sample_id','state','reviewer','criteria','capability','equipment_type'}
    for key,value in parameters.items():
        if allow_references and isinstance(value,dict) and set(value)=={'$ref'}:continue
        if key in integers:
            lo,hi=integers[key]
            if type(value) is not int or not lo<=value<=hi:raise ValueError(f'{key} requires an integer in [{lo},{hi}].')
        if key in {'include_sac','include_history','allow_extrapolation','override','approval_required'} and type(value) is not bool:raise ValueError(key+' requires a boolean.')
        if key=='composition_mode' and value not in {'independent','equimolar'}:raise ValueError('Choose independent or equimolar precursor feeds.')
        if key=='return_fields' and (not isinstance(value,list) or any(not isinstance(v,str) or not v for v in value) or len(set(value))!=len(value)):raise ValueError('Return fields must be distinct text identifiers.')
        if key in strings and (not isinstance(value,str) or not value.strip()):raise ValueError(key+' requires nonempty text.')
        if key in {'sample_ids','elements','models','modalities'} and (not isinstance(value,list) or any(not isinstance(v,str) or not v for v in value) or len(set(value))!=len(value)):raise ValueError(key+' requires distinct text identifiers.')
        if key=='modalities' and (not value or operation=='device.characterize' and set(value)-{'ICP','XRD','STEM','SEM'}):raise ValueError('Use supported acquisition modalities: ICP, XRD, STEM or SEM.')
        if key=='capability':
            from .device_catalog import CAPABILITIES
            if value not in CAPABILITIES:raise ValueError('Choose a registered equipment capability.')
        if key in {'temperature','target_total'} and (not isinstance(value,list) or len(value)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in value) or value[0]>value[1]):raise ValueError(key+' requires increasing finite bounds.')
        if key=='min_probability' and (type(value) not in (int,float) or not 0<=value<=1):raise ValueError('min_probability requires a number in [0,1].')
        if key in {'pulses','values'} and (not isinstance(value,list) or not value or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in value)):raise ValueError(key+' requires positive finite values.')
    if operation=='manual.checkpoint' and 'device_id' in parameters and 'capability' not in parameters:raise ValueError('A device-bound manual action needs a capability.')
    return parameters


def references(value):
    if isinstance(value,dict):
        if '$ref' in value:
            if set(value)!={'$ref'} or not isinstance(value['$ref'],str):raise ValueError('A reference is exactly {"$ref":"step.output.path"}.')
            yield value['$ref']
        else:
            for child in value.values():yield from references(child)
    elif isinstance(value,list):
        for child in value:yield from references(child)


def validate_spec(spec):
    if not isinstance(spec,dict) or set(spec)-{'name','version','steps','description'}:raise ValueError('Use name, version, description and steps only.')
    if not isinstance(spec.get('name'),str) or not 1<=len(spec['name'].strip())<=120:raise ValueError('Name the workflow (1-120 characters).')
    steps=spec.get('steps')
    if not isinstance(steps,list) or not 1<=len(steps)<=48:raise ValueError('A workflow needs 1-48 steps.')
    if len(json.dumps(spec,allow_nan=False))>120000:raise ValueError('Workflow definition exceeds 120 KB.')
    seen=set();ancestors={}
    for node in steps:
        if not isinstance(node,dict) or set(node)-{'id','operation','parameters','depends_on','schedule'}:raise ValueError('Invalid step fields.')
        if 'schedule' in node:
            from .workflow_schedule import validate_schedule
            validate_schedule(node['schedule'])
        key=node.get('id','');op=node.get('operation');params=node.get('parameters',{});deps=node.get('depends_on',[])
        if not isinstance(key,str) or not IDENTIFIER.fullmatch(key) or key in seen:raise ValueError('Use unique identifier-like step IDs.')
        if op not in OPERATIONS:raise ValueError('Unsupported workflow operation: '+str(op))
        if not isinstance(deps,list) or any(not isinstance(d,str) or d not in seen for d in deps) or len(set(deps))!=len(deps):raise ValueError('Dependencies must name earlier steps, without duplicates.')
        validate_parameters(op,params)
        ancestry=set(deps)
        for dep in deps:ancestry|=ancestors[dep]
        for reference in references(params):
            head,*path=reference.split('.')
            if head not in ancestry or not path or any(not re.fullmatch(r'[A-Za-z0-9_-]+',p) for p in path):raise ValueError('References must point to declared dependency outputs.')
        seen.add(key);ancestors[key]=ancestry
    result=copy.deepcopy(spec);result.setdefault('version',1)
    return result


def resolve(value,outputs):
    if isinstance(value,dict):
        if '$ref' in value:
            parts=value['$ref'].split('.');result=outputs[parts[0]]
            for part in parts[1:]:result=result[int(part)] if isinstance(result,list) else result[part]
            return copy.deepcopy(result)
        return {k:resolve(v,outputs) for k,v in value.items()}
    if isinstance(value,list):return [resolve(v,outputs) for v in value]
    return value


class WorkflowDB:
    def __init__(self,root,origin):
        if origin not in {'REAL','DEMO'}:raise ValueError('Select a data space.')
        self.origin=origin;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/f'workflows_{origin.lower()}.sqlite3'
        with closing(self.connect()) as db,db:
            db.executescript('''CREATE TABLE IF NOT EXISTS definitions(id TEXT PRIMARY KEY, payload TEXT);
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,payload TEXT);
            CREATE TABLE IF NOT EXISTS steps(run_id TEXT,id TEXT,state TEXT,payload TEXT,PRIMARY KEY(run_id,id));
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,created TEXT,payload TEXT);
            CREATE TABLE IF NOT EXISTS devices(id TEXT PRIMARY KEY,payload TEXT);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,payload TEXT);
            CREATE TABLE IF NOT EXISTS locks(device_id TEXT PRIMARY KEY,job_id TEXT);
            CREATE TABLE IF NOT EXISTS receipts(id TEXT PRIMARY KEY,payload TEXT);''')

    def connect(self):return sqlite3.connect(self.path,timeout=20)

    def get(self,table,key):
        if table not in {'definitions','runs','devices','jobs','receipts'}:raise ValueError('Invalid workflow table.')
        with closing(self.connect()) as db:
            row=db.execute(f'SELECT payload FROM {table} WHERE id=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self,table):
        if table not in {'definitions','runs','devices','jobs'}:raise ValueError('Invalid workflow table.')
        with closing(self.connect()) as db:return [json.loads(row[0]) for row in db.execute(f'SELECT payload FROM {table} ORDER BY id')]

    def event(self,db,**event):db.execute('INSERT INTO events(created,payload) VALUES(?,?)',(stamp(),json.dumps(event,allow_nan=False)))

    def save(self,spec,template=None):
        spec=validate_spec(spec);key='workflow_'+uuid4().hex[:16]
        row={'id':key,'origin':self.origin,'created':stamp(),'spec':spec,'spec_hash':digest(spec)}
        if template is not None:row['template']=copy.deepcopy(template)
        with closing(self.connect()) as db,db:
            db.execute('INSERT INTO definitions VALUES(?,?)',(key,json.dumps(row,allow_nan=False)));self.event(db,action='SAVE_DEFINITION',workflow_id=key)
        return row

    def start(self,workflow_id,context,selected_ids,selected_keys=None,source_fingerprint=None):
        definition=self.get('definitions',workflow_id)
        if definition is None:raise ValueError('Unknown saved workflow.')
        row={'id':'run_'+uuid4().hex[:16],'workflow_id':workflow_id,'origin':self.origin,'created':stamp(),
             'state':'READY','spec':copy.deepcopy(definition['spec']),'spec_hash':definition['spec_hash'],
             'context':copy.deepcopy(context),'selected_ids':list(selected_ids),'selected_keys':selected_keys or {},
             'source_fingerprint':source_fingerprint,'error':None}
        with closing(self.connect()) as db,db:
            db.execute('INSERT INTO runs VALUES(?,?)',(row['id'],json.dumps(row,allow_nan=False)))
            for node in row['spec']['steps']:
                payload={'id':node['id'],'operation':node['operation'],'attempts':0,'output':None,'error':None,'lease_until':None}
                db.execute('INSERT INTO steps VALUES(?,?,?,?)',(row['id'],node['id'],'PENDING',json.dumps(payload)))
            self.event(db,action='START_RUN',run_id=row['id'])
        return row

    def steps(self,run_id):
        run=self.get('runs',run_id)
        if not run:raise ValueError('Unknown workflow run.')
        with closing(self.connect()) as db:
            rows={ident:{**json.loads(raw),'state':state} for ident,state,raw in db.execute('SELECT id,state,payload FROM steps WHERE run_id=?',(run_id,))}
        return [rows[n['id']] for n in run['spec']['steps']]

    def control(self,run_id,action):
        if action not in {'pause','resume','cancel','retry','recover'}:raise ValueError('Unsupported workflow control.')
        with closing(self.connect()) as db,db:
            db.execute('BEGIN IMMEDIATE');encoded=db.execute('SELECT payload FROM runs WHERE id=?',(run_id,)).fetchone()
            if not encoded:raise ValueError('Unknown workflow run.')
            run=json.loads(encoded[0]);active=db.execute("SELECT id,payload FROM steps WHERE run_id=? AND state='RUNNING'",(run_id,)).fetchall()
            if action=='recover':
                if not active:raise ValueError('No interrupted step to reconcile.')
                for ident,raw in active:
                    step=json.loads(raw)
                    if step['lease_until']>stamp():raise ValueError('The step lease is still active; wait before reconciling.')
                    step['error']='Interrupted worker; retry uses operation receipts and device job IDs.'
                    held=step.get('manual_lock')
                    if held:db.execute('DELETE FROM locks WHERE device_id=? AND job_id=?',(held['device_id'],held['owner']))
                    db.execute("UPDATE steps SET state='FAILED',payload=? WHERE run_id=? AND id=?",(json.dumps(step),run_id,ident))
                run['state']='FAILED'
            elif action=='cancel':
                if run['state']=='COMPLETED':raise ValueError('A completed run cannot be cancelled.')
                run['state']='CANCELLED'
                for ident,raw in db.execute("SELECT id,payload FROM steps WHERE run_id=?",(run_id,)).fetchall():
                    held=json.loads(raw).get('manual_lock')
                    if held:db.execute('DELETE FROM locks WHERE device_id=? AND job_id=?',(held['device_id'],held['owner']))
            elif action=='pause':
                if run['state'] not in {'READY','RUNNING','WAITING','QUEUED'}:raise ValueError('This run cannot be paused.')
                run['state']='PAUSED'
            elif action=='resume':
                if run['state']!='PAUSED' or active:raise ValueError('Resume a paused, inactive workflow.')
                waiting=db.execute("SELECT 1 FROM steps WHERE run_id=? AND state='WAITING'",(run_id,)).fetchone()
                run['state']='WAITING' if waiting else 'READY'
            elif action=='retry':
                if run['state']!='FAILED' or active:raise ValueError('Retry an inactive failed workflow.')
                db.execute("UPDATE steps SET state='PENDING' WHERE run_id=? AND state='FAILED'",(run_id,));run['state']='READY';run['error']=None
            db.execute('UPDATE runs SET payload=? WHERE id=?',(json.dumps(run),run_id));self.event(db,action=action.upper(),run_id=run_id)
        return run

    def acknowledge(self,run_id,node_id,note,returned_data=None):
        if not isinstance(note,str) or not note.strip():raise ValueError('Document the completed manual action.')
        from .workflow_returns import validate_return
        with closing(self.connect()) as db,db:
            db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM runs WHERE id=?',(run_id,)).fetchone()
            if not row:raise ValueError('Unknown workflow run.')
            run=json.loads(row[0]);step=db.execute("SELECT payload FROM steps WHERE run_id=? AND id=? AND state='WAITING'",(run_id,node_id)).fetchone()
            if run['state']!='WAITING' or not step:raise ValueError('Acknowledge a waiting manual checkpoint.')
            payload=json.loads(step[0]);output=payload.get('output') or {}
            data=validate_return(output,returned_data)
            payload['output']={**output,'note':note,'returned_data':data,'acknowledged_at':stamp(),'awaiting_manual_action':False}
            payload['finished']=payload['output']['acknowledged_at']
            held=payload.get('manual_lock')
            if held:
                db.execute('DELETE FROM locks WHERE device_id=? AND job_id=?',(held['device_id'],held['owner']))
                payload['output']['device_id']=held['device_id']
            db.execute("UPDATE steps SET state='COMPLETED',payload=? WHERE run_id=? AND id=?",(json.dumps(payload),run_id,node_id))
            run['state']='READY' if db.execute("SELECT 1 FROM steps WHERE run_id=? AND state!='COMPLETED'",(run_id,)).fetchone() else 'COMPLETED'
            db.execute('UPDATE runs SET payload=? WHERE id=?',(json.dumps(run),run_id));self.event(db,action='MANUAL_ACKNOWLEDGEMENT',run_id=run_id,node_id=node_id)

    def advance(self,run_id,executor):
        with closing(self.connect()) as db,db:
            db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT payload FROM runs WHERE id=?',(run_id,)).fetchone()
            if not row:raise ValueError('Unknown workflow run.')
            run=json.loads(row[0])
            if run['state'] not in {'READY','RUNNING','QUEUED'}:raise ValueError('Run is '+run['state']+'.')
            steps={ident:{**json.loads(raw),'state':state} for ident,state,raw in db.execute('SELECT id,state,payload FROM steps WHERE run_id=?',(run_id,))}
            if any(s['state']=='RUNNING' for s in steps.values()):raise ValueError('Another worker already owns a running step.')
            from .workflow_schedule import choose_ready
            devices=[json.loads(r[0]) for r in db.execute('SELECT payload FROM devices')]
            locks=dict(db.execute('SELECT device_id,job_id FROM locks'))
            node=choose_ready(run['spec'],list(steps.values()),devices,locks,self.origin)
            if node is None:
                if all(s['state']=='COMPLETED' for s in steps.values()):run['state']='COMPLETED';db.execute('UPDATE runs SET payload=? WHERE id=?',(json.dumps(run),run_id));return run
                run['state']='QUEUED';run['scheduling_note']='Waiting for dependencies or an available compatible device.'
                db.execute('UPDATE runs SET payload=? WHERE id=?',(json.dumps(run),run_id));return run
            step=steps[node['id']];step['attempts']+=1;step['started']=stamp();step['lease_until']=(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat()
            if node['operation']=='manual.checkpoint' and node.get('parameters',{}).get('device_id'):
                from .workflow_schedule import device_requirement
                outputs={key:s['output'] for key,s in steps.items() if s['state']=='COMPLETED'}
                device_id,_=device_requirement(node,outputs);owner='manual:'+run_id+':'+node['id']
                db.execute('INSERT INTO locks VALUES(?,?)',(device_id,owner))
                step['manual_lock']={'device_id':device_id,'owner':owner}
            db.execute("UPDATE steps SET state='RUNNING',payload=? WHERE run_id=? AND id=?",(json.dumps(step),run_id,node['id']))
            run['state']='RUNNING';db.execute('UPDATE runs SET payload=? WHERE id=?',(json.dumps(run),run_id));self.event(db,action='CLAIM_STEP',run_id=run_id,node_id=node['id'])
        try:
            outputs={key:s['output'] for key,s in steps.items() if s['state']=='COMPLETED'}
            params=resolve(node.get('parameters',{}),outputs)
            output=executor(node['operation'],params,run,run_id+':'+node['id'])
            json.dumps(output,allow_nan=False)
            waiting=node['operation']=='manual.checkpoint'
            status='WAITING' if waiting else 'COMPLETED';error=None
        except Exception as exc:
            status='FAILED';output=None;error=f'{type(exc).__name__}: {exc}'[:2000]
        with closing(self.connect()) as db,db:
            db.execute('BEGIN IMMEDIATE');current=json.loads(db.execute('SELECT payload FROM runs WHERE id=?',(run_id,)).fetchone()[0])
            step.update(output=output,error=error,finished=stamp(),lease_until=None)
            if step.get('manual_lock') and (status=='FAILED' or current['state']=='CANCELLED'):
                held=step['manual_lock'];db.execute('DELETE FROM locks WHERE device_id=? AND job_id=?',(held['device_id'],held['owner']))
            db.execute('UPDATE steps SET state=?,payload=? WHERE run_id=? AND id=?',(status,json.dumps(step,allow_nan=False),run_id,node['id']))
            if current['state'] not in {'PAUSED','CANCELLED'}:current['state']='FAILED' if error else 'WAITING' if waiting else 'READY'
            if current['state']=='READY' and not db.execute("SELECT 1 FROM steps WHERE run_id=? AND state!='COMPLETED'",(run_id,)).fetchone():current['state']='COMPLETED'
            current['error']=error;db.execute('UPDATE runs SET payload=? WHERE id=?',(json.dumps(current),run_id));self.event(db,action='FINISH_STEP',run_id=run_id,node_id=node['id'],state=status,error=error)
        return current

    def run_ready(self,run_id,executor,max_steps=48,progress=None):
        for _ in range(min(48,int(max_steps))):
            run=self.get('runs',run_id)
            if not run or run['state'] not in {'READY','RUNNING','QUEUED'}:return run
            if progress:progress(self.summary(run_id))
            run=self.advance(run_id,executor)
            if run['state']=='QUEUED':return run
        return run

    def summary(self,run_id):
        run=self.get('runs',run_id)
        if not run:raise ValueError('Unknown workflow run.')
        steps=self.steps(run_id)
        return {'run_id':run_id,'workflow_id':run['workflow_id'],'name':run['spec']['name'],'origin':self.origin,'state':run['state'],
                'completed_steps':sum(s['state']=='COMPLETED' for s in steps),'total_steps':len(steps),'error':run['error'],
                'steps':[{k:s[k] for k in ['id','operation','state','attempts','error']} for s in steps]}
