"""Configurable device contracts and deterministic DEMO-only job simulators."""
from contextlib import closing
import copy
import hashlib
import io
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from .discovery.common import digest,identity,query,recipe_id,stamp
from .workflow_engine import IDENTIFIER
from .device_catalog import CATEGORIES,CAPABILITIES,PROFILES,SIMULATED_MODALITIES,category_of,modalities_of,profile_device


class SimulatedDeviceFailure(RuntimeError):pass


class DeviceBroker:
    def __init__(self,database,store):self.db=database;self.store=store

    def register(self,definition):
        allowed={'id','name','mode','capabilities','limits','mcp_endpoint','description','fail_once','category','equipment_type','modalities'}
        if not isinstance(definition,dict) or set(definition)-allowed:raise ValueError('Unsupported device fields.')
        device=copy.deepcopy(definition);ident=device.get('id','')
        if not IDENTIFIER.fullmatch(ident) or not isinstance(device.get('name'),str) or not device['name'].strip():raise ValueError('Name the device and use an identifier-like ID.')
        if device.get('mode') not in {'SIMULATOR','MCP_NOT_CONNECTED','INVENTORY_ONLY'}:raise ValueError('Choose inventory, a simulator or a disconnected MCP connection.')
        caps=device.get('capabilities',[])
        if not isinstance(caps,list) or not caps or any(c not in CAPABILITIES for c in caps) or len(set(caps))!=len(caps):raise ValueError('Declare distinct supported device capabilities.')
        device.setdefault('category',category_of(device))
        if device['category'] not in CATEGORIES:raise ValueError('Choose a supported equipment category.')
        device.setdefault('equipment_type','custom')
        if device['equipment_type']!='custom' and device['equipment_type'] not in PROFILES:raise ValueError('Choose a catalog equipment type or custom.')
        device.setdefault('modalities',modalities_of(device))
        modalities=device['modalities']
        if not isinstance(modalities,list) or any(not isinstance(m,str) or not m.strip() or len(m)>50 for m in modalities) or len(set(modalities))!=len(modalities):raise ValueError('Declare distinct measurement modalities.')
        if device['mode']=='SIMULATOR':
            if set(caps)-{'synthesize','characterize'}:raise ValueError('No simulator exists for these capabilities. Register planning inventory or a disconnected MCP device.')
            if 'synthesize' in caps and device['equipment_type'] not in {'custom','pulsed_heater'}:raise ValueError('Only the pulsed-heater synthesis fixture is implemented.')
            if 'characterize' in caps and (not modalities or set(modalities)-set(SIMULATED_MODALITIES)):raise ValueError('Only ICP, XRD, STEM and SEM acquisition fixtures are implemented.')
        limits=device.setdefault('limits',{'cutoff_C':[100,2600],'pulse_count':[1,16]} if 'synthesize' in caps else {})
        if not isinstance(limits,dict) or set(limits)-{'cutoff_C','pulse_count'}:raise ValueError('Device limits support cutoff_C and pulse_count.')
        for key,bounds in limits.items():
            if not isinstance(bounds,list) or len(bounds)!=2 or any(type(v) not in (int,float) or not np.isfinite(v) for v in bounds) or not 0<bounds[0]<=bounds[1]:raise ValueError('Invalid finite positive device limits.')
        endpoint=device.get('mcp_endpoint','')
        if endpoint:
            parsed=urlsplit(endpoint)
            if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:raise ValueError('Store a credential-free HTTPS endpoint only. MCP transport is not connected in this release.')
        if type(device.get('fail_once',False)) is not bool:raise ValueError('fail_once must be boolean.')
        device.update(origin=self.store.origin,updated=stamp(),connection_state='SIMULATED' if device['mode']=='SIMULATOR' else 'INVENTORY' if device['mode']=='INVENTORY_ONLY' else 'NOT_CONNECTED')
        with closing(self.db.connect()) as db,db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM locks WHERE device_id=?',(ident,)).fetchone():raise ValueError('Cannot edit a device with an active job.')
            db.execute('INSERT OR REPLACE INTO devices VALUES(?,?)',(ident,json.dumps(device,allow_nan=False)))
            self.db.event(db,action='REGISTER_DEVICE',device_id=ident,mode=device['mode'])
        return device

    def defaults(self):
        result=[]
        for ident,profile in [('demo_heater','pulsed_heater'),('demo_icp','icp_oes'),('demo_xrd','xrd'),('demo_stem','stem'),('demo_sem','sem')]:
            definition=profile_device(profile,ident,'Simulated '+PROFILES[profile]['name'],'SIMULATOR')
            result.append(self.db.get('devices',ident) or self.register(definition))
        return result

    def catalog(self):
        from .device_catalog import CATALOG
        return copy.deepcopy(CATALOG)

    def submit(self,device_id,capability,samples,key,synthesis=None,modalities=None):
        if self.store.origin!='DEMO':raise ValueError('Simulated acquisition is restricted to DEMO; it cannot populate REAL experiments.')
        device=self.db.get('devices',device_id)
        if device is None or capability not in device['capabilities']:raise ValueError('Device lacks the requested capability.')
        if device['mode']!='SIMULATOR':raise ValueError('The real-device MCP transport is not connected; no device command was sent.')
        if capability not in {'synthesize','characterize'}:raise ValueError('This device capability has no execution adapter.')
        if capability=='characterize':
            modalities=modalities_of(device) if modalities is None else modalities
            if not isinstance(modalities,list) or not modalities or any(m not in modalities_of(device) for m in modalities):raise ValueError('Requested measurement modality does not match this instrument.')
        if not samples or any(s.get('origin')!='DEMO' for s in samples):raise ValueError('Choose DEMO specimens only.')
        for sample in samples:
            recipe=query(sample)
            for name,field in ([('cutoff_C','pulse_limit_C'),('pulse_count','pulse_count')] if capability=='synthesize' else []):
                if name in device['limits'] and not device['limits'][name][0]<=recipe[field]<=device['limits'][name][1]:raise ValueError('Recipe exceeds device '+name+' limits.')
        payload_hash=digest({'device':device,'capability':capability,'recipes':[recipe_id(s) for s in samples],
                             'sample_keys':[identity(s) for s in samples],'synthesis':synthesis})
        if capability=='characterize':payload_hash=digest([payload_hash,modalities])
        job_id='job_'+hashlib.sha256((key+':'+device_id).encode()).hexdigest()[:24]
        with closing(self.db.connect()) as db,db:
            db.execute('BEGIN IMMEDIATE');previous=db.execute('SELECT payload FROM jobs WHERE id=?',(job_id,)).fetchone()
            job=json.loads(previous[0]) if previous else None
            if job:
                if job['payload_hash']!=payload_hash:raise ValueError('Job key reused with different inputs or a changed device.')
                if job['state']=='COMPLETED':return job['output']
                if job['state']=='RUNNING':raise ValueError('Device job already running; reconcile its job ID before retrying.')
            lock=db.execute('SELECT job_id FROM locks WHERE device_id=?',(device_id,)).fetchone()
            if lock:raise ValueError('Device is busy with '+lock[0])
            attempts=(job['attempts'] if job else 0)+1
            job={'id':job_id,'device_id':device_id,'capability':capability,'origin':'DEMO','state':'RUNNING',
                 'attempts':attempts,'payload_hash':payload_hash,'created':stamp(),'output':None,'simulated':True,
                 'operation_key':key,'lease_until':(datetime.now(timezone.utc)+timedelta(minutes=30)).isoformat()}
            if device.get('fail_once') and attempts==1:
                job['state']='FAILED_BEFORE_EXECUTION'
                db.execute('INSERT OR REPLACE INTO jobs VALUES(?,?)',(job_id,json.dumps(job)))
                self.db.event(db,action='SIMULATED_PRE_EXECUTION_FAILURE',job_id=job_id)
            else:
                db.execute('INSERT OR REPLACE INTO jobs VALUES(?,?)',(job_id,json.dumps(job)))
                db.execute('INSERT INTO locks VALUES(?,?)',(device_id,job_id))
                self.db.event(db,action='SUBMIT_JOB',job_id=job_id,device_id=device_id)
        if job['state']=='FAILED_BEFORE_EXECUTION':raise SimulatedDeviceFailure('Injected pre-execution failure. Explicit retry will reuse '+job_id)
        try:
            output=self._synthesize(samples,job_id) if capability=='synthesize' else self._characterize(samples,job_id,synthesis,modalities)
            job.update(state='COMPLETED',finished=stamp(),output=output)
        except Exception as exc:
            job.update(state='FAILED',error=type(exc).__name__)
            with closing(self.db.connect()) as db,db:
                db.execute('UPDATE jobs SET payload=? WHERE id=?',(json.dumps(job),job_id));db.execute('DELETE FROM locks WHERE job_id=?',(job_id,))
            raise
        with closing(self.db.connect()) as db,db:
            db.execute('UPDATE jobs SET payload=? WHERE id=?',(json.dumps(job,allow_nan=False),job_id));db.execute('DELETE FROM locks WHERE job_id=?',(job_id,))
            self.db.event(db,action='COMPLETE_JOB',job_id=job_id)
        return output

    def reconcile_expired(self,run_id):
        if self.store.origin!='DEMO':return []
        recovered=[]
        with closing(self.db.connect()) as db,db:
            db.execute('BEGIN IMMEDIATE')
            for ident,raw in db.execute('SELECT id,payload FROM jobs').fetchall():
                job=json.loads(raw)
                if job.get('operation_key','').startswith(run_id+':') and job['state']=='RUNNING' and job.get('lease_until','9999')<stamp():
                    job['state']='FAILED_INTERRUPTED_SIMULATOR'
                    db.execute('UPDATE jobs SET payload=? WHERE id=?',(json.dumps(job),ident));db.execute('DELETE FROM locks WHERE job_id=?',(ident,))
                    self.db.event(db,action='RECONCILE_EXPIRED_SIMULATOR',job_id=ident);recovered.append(ident)
        return recovered

    def _synthesize(self,samples,job_id):
        return {'job_id':job_id,'simulated':True,'sample_ids':[s['sample_id'] for s in samples],
                'sample_keys':[identity(s) for s in samples],'recipe_keys':[recipe_id(s) for s in samples],
                'completed':True,'note':'Simulator receipt, not a physical synthesis.'}

    def _characterize(self,samples,job_id,synthesis,modalities):
        from . import pulseicp_bridge as bridge
        from ._pulseicp.mass_decoder import prior
        if not synthesis or not synthesis.get('simulated') or not synthesis.get('completed'):raise ValueError('Characterization requires a completed simulated synthesis receipt.')
        source_job=self.db.get('jobs',synthesis.get('job_id',''))
        if not source_job or source_job['state']!='COMPLETED' or source_job['capability']!='synthesize' or digest(source_job['output'])!=digest(synthesis):
            raise ValueError('Synthesis receipt must match a completed device job.')
        if synthesis['sample_keys']!=[identity(s) for s in samples] or synthesis['recipe_keys']!=[recipe_id(s) for s in samples]:raise ValueError('Synthesis receipt differs from the current samples.')
        physics=bridge.configuration()['physics'];measurements=[]
        folder=self.store.root/'workflow_simulation'/'demo'/job_id;folder.mkdir(parents=True,exist_ok=True)
        for sample in samples:
            recipe=query(sample);amount=sum(recipe['concentrations_mM'].values())*recipe['solvent_volume_mL']/1000/len(recipe['elements'])
            passed=bool(amount<1.8 and 900<=recipe['pulse_limit_C']<=1800 and recipe['pulse_count']<=8)
            rng=np.random.default_rng(int(digest([synthesis['job_id'],identity(sample)])[:8],16))
            factor=float(np.clip(.84+.12*np.sin(recipe['pulse_limit_C']/500)+rng.normal(0,.025),.6,1.1))
            composition=bridge.composition(recipe,physics)
            components={e:prior(composition.component(e),physics) for e in recipe['elements']}
            retained={e:p['feed_metal_g']*float(np.clip(p['retention_prior']*factor,0.,1.)) for e,p in components.items()}
            matrix=next(iter(components.values()))['matrix_mass_g'];denominator=matrix+sum(retained.values())
            assays=[{'element':e,'value':str(100*mass/denominator),'unit':'wt%','flag':'MEASURED',
                     'source':'SIMULATED_WORKFLOW_FIXTURE'} for e,mass in retained.items()]
            x=np.linspace(10,80,701);y=12+15*np.exp(-.5*((x-25)/5)**2)
            if not passed:y+=100*np.exp(-.5*((x-44)/.18)**2)
            spectrum=('# SIMULATED_WORKFLOW_FIXTURE_NOT_EXPERIMENTAL\ntwo_theta_deg,intensity\n'+'\n'.join(f'{a:.5f},{b:.8f}' for a,b in zip(x,y))).encode()
            fields=[('XRD','.csv',spectrum)] if 'XRD' in modalities else []
            yy,xx=np.mgrid[:128,:128]
            for modality in ['STEM','SEM']:
                if modality not in modalities:continue
                image=np.full((128,128),100.)
                positions=[(28,28),(62,65),(95,92)] if passed else [(61,61),(64,61),(62,64),(90,90)]
                scale=1.3 if modality=='STEM' and passed else 3.5
                for px,py in positions:image+=1800*np.exp(-((xx-px)**2+(yy-py)**2)/(2*scale**2))
                buffer=io.BytesIO();Image.fromarray(image.astype('uint16')).save(buffer,format='TIFF',tiffinfo={270:'SIMULATED_WORKFLOW_FIXTURE_NOT_EXPERIMENTAL'});fields.append((modality,'.tif',buffer.getvalue()))
            files=[]
            for modality,extension,data in fields:
                path=folder/(digest(identity(sample))[:24]+'_'+modality+extension);path.write_bytes(data)
                files.append({'modality':modality,'relative_path':str(path.relative_to(self.store.root)),
                              'sha256':hashlib.sha256(data).hexdigest(),'file_name':path.name})
            measurements.append({'sample_id':sample['sample_id'],'record_key':identity(sample),'recipe_key':recipe_id(sample),
                'icp':assays if 'ICP' in modalities else [],'files':files,'fixture_structural_PASS':passed,'simulated':True,'source_job_id':job_id,
                'synthesis_job_id':synthesis['job_id'],
                'mass_balance':{'matrix_g':matrix,'retained_metal_g':retained,'feed_metal_g':{e:p['feed_metal_g'] for e,p in components.items()},'total_product_g':denominator}})
        return {'job_id':job_id,'simulated':True,'modalities':list(modalities),'measurements':measurements,
                'note':'Synthetic fixture truth is used only to validate workflow plumbing, never real recognition accuracy.'}
