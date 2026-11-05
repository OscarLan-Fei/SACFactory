"""Shared scientific operations for conversation and the workflow interface."""
from contextlib import closing
import copy
import hashlib
import json
import time
from pathlib import Path

from .compute_runtime import execution
from .discovery.common import FIELDS,digest,identity,query,recipe_id,stamp,prediction_metrics
from .discovery.campaigns import CampaignLedger
from .workflow_engine import WorkflowDB,OPERATIONS,validate_parameters
from .workflow_devices import DeviceBroker


def simulation_template(elements=None,batch_size=32,epochs=10):
    elements=elements or ['Fe'];steps=[]
    def add(ident,op,params,deps=()):steps.append({'id':ident,'operation':op,'parameters':params,'depends_on':list(deps)})
    def ref(path):return {'$ref':path}
    add('memory','knowledge.search',{'query':'precursor mass balance temperature pulse characterization'})
    add('design','inverse.design',{'elements':elements,'bounds_mmol':{e:[.2,3.] for e in elements},'batch_size':batch_size,
        'count':max(64,batch_size*2),'temperature':[800,1600],'pulses':[1,2,4,8,12],'model':'Fixed prior','seed':42,'allow_extrapolation':True},['memory'])
    add('pending','campaign.plan',{'design':ref('design.design')},['design'])
    add('synthesis','device.synthesize',{'device_id':'demo_heater','sample_ids':ref('pending.sample_ids')},['pending'])
    measurement_steps=[]
    for step,modality in [('measurement','ICP'),('xrd','XRD'),('stem','STEM'),('sem','SEM')]:
        add(step,'device.characterize',{'device_id':'demo_'+modality.lower(),'modalities':[modality],'synthesis':ref('synthesis.receipt')},['synthesis'])
        measurement_steps.append(step)
    add('recognition','characterization.analyze',{'measurements':{'acquisitions':[ref(step+'.acquisition') for step in measurement_steps]}},measurement_steps)
    add('review','characterization.fixture_review',{'evidence':ref('recognition.evidence')},['recognition'])
    add('feedback','campaign.evaluate',{},['review'])
    add('validation','validation.run',{'sample_ids':ref('pending.sample_ids'),'folds':3,'epochs':5,'include_sac':False},['pending','feedback'])
    add('update','model.train',{'sample_ids':ref('pending.sample_ids'),'family':'SAC-Gen-Set','epochs':epochs},['pending','validation'])
    add('comparison','model.compare',{'sample_ids':ref('pending.sample_ids'),'models':['Fixed prior','SAC-Gen-Set']},['pending','update'])
    add('hypothesis','hypothesis.register',{'sample_id':ref('pending.sample_ids.0'),'factor':'temperature','values':[1000,1400],
        'statement':'Changing controlled cutoff at fixed feed changes predicted retention.','model':'SAC-Gen-Set'},['pending','update'])
    add('next_design','inverse.design',{'elements':elements,'batch_size':4,'count':32,'model':'SAC-Gen-Set','allow_extrapolation':True},['update','comparison','hypothesis'])
    return {'name':'Complete simulated SAC-Gen loop','version':2,'description':'Independent ICP, XRD, STEM and SEM fixtures; not prospective scientific validation.','steps':steps}


class WorkflowService:
    def __init__(self,store,context=None,selected_ids=None):
        self.store=store;self.db=WorkflowDB(store.root,store.origin);self.ledger=CampaignLedger(store.root,store.origin)
        self.devices=DeviceBroker(self.db,store)
        self.selected_ids=list(selected_ids) if selected_ids is not None else [s['sample_id'] for s in store.all('sample')]
        default={'origin':store.origin,'elements':['Fe'],'concentrations_mM':{'Fe':50.},'pulse_limit_C':1000.,'pulse_count':1,
                 'pan_mass_g':2.,'solvent_volume_mL':20.,'precursor_family':'METAL_SALTS_V1','support_batch':'PAN_DERIVED_CARBON','protocol_id':'PAN_DMF_PULSE_V1'}
        candidate=context or default
        self.context={k:copy.deepcopy(v) for k,v in candidate.items() if k in FIELDS}
        self.context_warning=None
        try:self.context=query(self.context)
        except (ValueError,TypeError) as exc:self.context_warning=str(exc)
        if self.context['origin']!=store.origin:raise ValueError('Workflow context differs from the data space.')

    def capabilities(self):
        from .workflow_coordination import SPECIALISTS
        from .workflow_templates import catalog,OPTION_KEYS
        from .discovery.workflows import available_models
        return {'origin':self.store.origin,'operations':{name:{'purpose':spec[0],'required':spec[1],'optional':spec[2]} for name,spec in OPERATIONS.items()},
                'references':'{"$ref":"earlier_step.output.path"}; declare that step as a dependency.',
                'device_modes':['SIMULATOR','MCP_NOT_CONNECTED','INVENTORY_ONLY'],'simulation_scope':'DEMO only',
                'templates':catalog(self.store.origin),'template_options':'create_template uses spec_json={"template_id":"...","options":{...}}. Options: name, elements, batch_size, bounds_mmol, temperature, pulses, model, candidate_family, epochs, folds, comparison_models, sample_ids, artifact_ids, bindings, register_resources, allow_extrapolation. Saving does not start a run.',
                'template_parameter_keys':sorted(OPTION_KEYS),'conversation_aliases':{'bounds_mM':'Converted to bounds_mmol using solvent_volume_mL.'},
                'models':available_models(self.store.root,self.store.origin),'specialists':SPECIALISTS,
                'coordination_mode':'Tool-bounded roles; independent LLM agents are not launched by workflow execution.',
                'context_warning':self.context_warning,'manual_checkpoint':'Waits until acknowledged in the interface; assigned equipment stays reserved until acknowledgement or cancellation.',
                'devices':[{k:d.get(k) for k in ['id','name','category','equipment_type','mode','capabilities','modalities','limits','connection_state']} for d in self.db.all('devices')],
                'equipment_catalog':self.devices.catalog(),
                'device_contracts':{'synthesize':{'input':'Selected specimen IDs and controlled recipes','output':'Synthesis job receipt'},
                                    'characterize':{'input':'Verified completed synthesis receipt and matching modalities','output':'Only this instrument\'s measurements; analyze can join verified acquisitions from independent instruments'}},
                'scope':'Only selected source records and specimens produced by the current run. No arbitrary code, SQL, shell or raw file access.'}

    def template_plan(self,template_id,options=None):
        from .workflow_templates import INDEX,DEFAULT_DEVICES,PREPARATION_DEFAULTS,build,compatible
        from .device_catalog import profile_device
        from .discovery.workflows import available_models
        if options is not None and not isinstance(options,dict):raise ValueError('Template options must be an object.')
        o=copy.deepcopy(options or {});o.setdefault('elements',self.context.get('elements') or ['Fe'])
        for field,default in PREPARATION_DEFAULTS.items():o.setdefault(field,self.context.get(field,default))
        build(template_id,self.store.origin,o,self.selected_ids)
        meta=INDEX[template_id];lookup={d['id']:d for d in self.db.all('devices')};resources=[]
        bindings=copy.deepcopy(o.get('bindings',{}))
        for role in meta['roles']:
            role=copy.deepcopy(role)
            if template_id=='confirmed_synthesis_loop' and self.store.origin=='DEMO' and role['role'] in DEFAULT_DEVICES:role['automatic']=True
            default=DEFAULT_DEVICES[role['role']] if role['automatic'] else 'inventory_'+role['profile']
            ident=bindings.get(role['role'],default if role['automatic'] or o.get('register_resources',False) else None)
            if ident is None:continue
            device=lookup.get(ident)
            if device is None:
                if ident!=default or not o.get('register_resources',False):raise ValueError('Register the required instrument first or enable resource registration: '+str(ident))
                device=profile_device(role['profile'],ident,mode='SIMULATOR' if role['automatic'] else 'INVENTORY_ONLY')
                resources.append(device)
            if not compatible(role,device,self.store.origin):raise ValueError('Instrument does not match the template role: '+role['role'])
            bindings[role['role']]=ident
        o['bindings']=bindings
        spec=build(template_id,self.store.origin,o,self.selected_ids)
        available=available_models(self.store.root,self.store.origin)
        for node in spec['steps']:
            if node['operation']=='inverse.design' and node['id']!='next_design' and node['parameters']['model'] not in available:raise ValueError('Choose a registered prediction model.')
            if node['operation']=='model.compare' and template_id=='data_benchmark':
                names=node['parameters']['models']
                if not 1<=len(names)<=6 or set(names)-set(available):raise ValueError('Choose 1-6 registered models for comparison.')
        if meta['needs_artifacts']:
            artifacts={a['id']:a for a in self.ledger.records('characterization')}
            samples={identity(s):s for s in self.store.all('sample') if s['sample_id'] in o.get('sample_ids',self.selected_ids)}
            for ident in o.get('artifact_ids',[]):
                artifact=artifacts.get(ident);sample=samples.get(artifact.get('record_key')) if artifact else None
                if not sample or artifact['recipe_key']!=recipe_id(sample):raise ValueError('Choose current artifacts belonging to the selected specimens.')
        # Persist effective values so future chat edits do not inherit a changed UI context.
        o['name']=spec['name']
        if meta['needs_records']:o.setdefault('sample_ids',list(self.selected_ids))
        for node in spec['steps']:
            p=node['parameters']
            if node['id']=='design':
                for field in ['elements','batch_size','bounds_mmol','temperature','pulses','model','allow_extrapolation']:
                    if field in p:o.setdefault(field,copy.deepcopy(p[field]))
            if node['operation']=='model.train':o.setdefault('epochs',p.get('epochs',50));o.setdefault('candidate_family',p.get('family','SAC-Gen-Set'))
            if node['operation']=='validation.run':o.setdefault('folds',p.get('folds',5))
        return {'template':copy.deepcopy(meta),'spec':spec,'resources_to_register':resources,'options':o}

    def instantiate_template(self,template_id,options=None,revision_of=None):
        plan=self.template_plan(template_id,options)
        if revision_of:
            original=self.inspect_workflow(revision_of)
            plan['spec']['version']=int(original['spec'].get('version',1))+1
        for device in plan['resources_to_register']:self.devices.register(device)
        result=self.db.save(plan['spec'],{'id':template_id,'options':plan['options'],'revision_of':revision_of})
        with closing(self.db.connect()) as db,db:
            self.db.event(db,action='INSTANTIATE_TEMPLATE',template_id=template_id,workflow_id=result['id'],spec_hash=result['spec_hash'])
        return {'workflow_id':result['id'],'name':result['spec']['name'],'steps':len(result['spec']['steps']),'template_id':template_id,'revision_of':revision_of}

    def inspect_workflow(self,workflow_id):
        row=self.db.get('definitions',workflow_id)
        if row is None:raise ValueError('Unknown saved workflow in this data space.')
        for node in row['spec']['steps']:
            p=node['parameters']
            ids=p.get('sample_ids')
            if isinstance(ids,list) and set(ids)-set(self.selected_ids):raise ValueError('This workflow uses records outside the current conversation selection.')
            if isinstance(p.get('sample_id'),str) and p['sample_id'] not in self.selected_ids:raise ValueError('This workflow uses a record outside the current conversation selection.')
        if not row.get('template'):
            recovered=self._template_provenance(row)
            if recovered:row={**row,'template':recovered}
        return row

    def revise_board_workflow(self,workflow_id,bindings=None,scheduling=None):
        """Preserve template editing across board changes and conversation turns."""
        from .workflow_schedule import bind_devices
        from .workflow_templates import INDEX
        original=self.inspect_workflow(workflow_id)
        revised=bind_devices(original['spec'],bindings or {},self.db.all('devices'),scheduling)
        metadata=copy.deepcopy(original.get('template'))
        if metadata:
            options=metadata['options'];options['name']=revised['name']
            options['scheduling']={n['id']:n['schedule'] for n in revised['steps'] if n.get('schedule')}
            roles=INDEX[metadata['id']]['roles'];options.setdefault('bindings',{})
            for node in revised['steps']:
                p=node['parameters']
                if node['id'] not in (bindings or {}):continue
                if node['operation']=='device.synthesize':role='heater'
                elif node['operation']=='device.characterize':role=p['modalities'][0].lower()
                else:role=next((r['role'] for r in roles if r['profile']==p.get('equipment_type')),None)
                if role:options['bindings'][role]=p['device_id']
            plan=self.template_plan(metadata['id'],options)
            plan['spec']['version']=revised['version'];revised=plan['spec']
            metadata={'id':metadata['id'],'options':plan['options'],'revision_of':original['id']}
        result=self.db.save(revised,metadata)
        with closing(self.db.connect()) as db,db:
            self.db.event(db,action='REVISE_WORKFLOW',workflow_id=result['id'],revision_of=original['id'])
        return result

    def _template_provenance(self,row):
        """Recover older template copies from their audit event without rewriting them."""
        from .workflow_templates import INDEX,PREPARATION_DEFAULTS
        with closing(self.db.connect()) as db:
            event=db.execute("SELECT payload FROM events WHERE json_extract(payload,'$.action')='INSTANTIATE_TEMPLATE' AND json_extract(payload,'$.workflow_id')=? ORDER BY seq DESC LIMIT 1",(row['id'],)).fetchone()
        template_id=json.loads(event[0]).get('template_id') if event else None
        if template_id not in INDEX:return None
        options={'name':row['spec']['name'],'register_resources':False,'bindings':{}}
        previous=sorted([r for r in self.db.all('runs') if r['workflow_id']==row['id']],key=lambda r:r['created'],reverse=True)
        basis=previous[0]['context'] if previous else self.context
        for field,default in PREPARATION_DEFAULTS.items():options[field]=basis.get(field,default)
        for node in row['spec']['steps']:
            p=node['parameters']
            if node['id']=='design':
                for field in ['elements','batch_size','bounds_mmol','temperature','pulses','model','allow_extrapolation']:
                    if field in p:options[field]=copy.deepcopy(p[field])
            if node['operation']=='model.train':options['candidate_family']=p.get('family','SAC-Gen-Set');options['epochs']=p.get('epochs',50)
            if node['operation']=='validation.run':
                options['folds']=p.get('folds',5)
                options.setdefault('epochs',p.get('epochs',20))
                if isinstance(p.get('sample_ids'),list):options['sample_ids']=p['sample_ids']
            if node['operation']=='model.compare' and template_id=='data_benchmark':options['comparison_models']=p.get('models',[])
            if node['operation']=='characterization.analyze' and p.get('measurements',{}).get('artifact_ids'):options['artifact_ids']=p['measurements']['artifact_ids']
            if node['operation']=='manual.checkpoint' and isinstance(p.get('sample_ids'),list) and p['sample_ids']:options.setdefault('sample_ids',p['sample_ids'])
            if isinstance(p.get('device_id'),str):
                if node['operation']=='device.synthesize':role='heater'
                elif node['operation']=='device.characterize' and p.get('modalities'):role=p['modalities'][0].lower()
                else:role=next((r['role'] for r in INDEX[template_id]['roles'] if r['profile']==p.get('equipment_type')),None)
                if role:options['bindings'][role]=p['device_id']
            if node.get('schedule'):options.setdefault('scheduling',{})[node['id']]=node['schedule']
        return {'id':template_id,'options':options,'revision_of':None,'recovered_from':'template audit',
                'preparation_source':'latest recorded run' if previous else 'current planning context'}

    def start(self,workflow_id):
        samples=self.store.all('sample');selected=[s for s in samples if s['sample_id'] in self.selected_ids]
        if set(self.selected_ids)-{s['sample_id'] for s in selected}:raise ValueError('A selected source sample no longer exists.')
        from .workflow_templates import PREPARATION_DEFAULTS
        definition=self.db.get('definitions',workflow_id)
        if definition is None:raise ValueError('Unknown saved workflow.')
        options=definition.get('template',{}).get('options',{})
        context={**self.context,**{field:options[field] for field in PREPARATION_DEFAULTS if field in options}}
        return self.db.start(workflow_id,context,self.selected_ids,{s['sample_id']:identity(s) for s in selected},digest(selected))

    def _samples(self,ids,run):
        expected=dict(run.get('selected_keys',{}))
        for step in self.db.steps(run['id']):
            if step['operation']=='campaign.plan' and step['output']:expected.update(step['output']['record_keys'])
        current=self.store.all('sample');by_id={s['sample_id']:s for s in current}
        requested=list(expected) if ids is None else ids
        if not isinstance(requested,list) or any(not isinstance(i,str) for i in requested):raise ValueError('Use a list of sample IDs.')
        if set(requested)-set(expected):raise ValueError('Samples are outside the workflow selection and produced specimens.')
        result=[]
        for ident in requested:
            sample=by_id.get(ident)
            if sample is None or identity(sample)!=expected[ident]:raise ValueError('A sample identity changed; create a fresh workflow run.')
            result.append(sample)
        return result

    def acknowledge(self,run_id,node_id,note,returned_data=None):
        self.authorize_run(run_id)
        run=self.db.get('runs',run_id)
        step=next((s for s in self.db.steps(run_id) if s['id']==node_id),None)
        if not step:raise ValueError('Unknown workflow step.')
        if (step.get('output') or {}).get('approval_required'):
            current=self._samples(run['selected_ids'],run)
            if digest(current)!=run.get('source_fingerprint'):raise ValueError('Source records changed during plan review. Create a fresh plan before dispatch.')
            if any(s['operation']=='campaign.plan' and s['state']=='COMPLETED' for s in self.db.steps(run_id)):raise ValueError('This plan has already been dispatched.')
        self.db.acknowledge(run_id,node_id,note,returned_data)
        return self.db.summary(run_id)

    def step_result(self,run_id,node_id):
        self.authorize_run(run_id)
        step=next((s for s in self.db.steps(run_id) if s['id']==node_id),None)
        if not step:raise ValueError('Unknown workflow step.')
        return self.public_result(step)

    def execute(self,operation,params,run,key):
        if run['origin']!=self.store.origin or digest(run['spec'])!=run['spec_hash']:raise ValueError('Workflow data space or frozen definition changed.')
        validate_parameters(operation,params,allow_references=False)
        fingerprint=digest([operation,params]);receipt=self.db.get('receipts',key)
        if receipt:
            if receipt['fingerprint']!=fingerprint:raise ValueError('Operation key reused with changed inputs.')
            if receipt.get('state','COMPLETED')=='COMPLETED':return receipt['output']
        if operation=='model.train':
            source_hash=digest(self._samples(params.get('sample_ids'),run))
            if receipt and receipt['source_fingerprint']!=source_hash:raise ValueError('Training sources changed after this step started; create a fresh run.')
            if not receipt:
                with closing(self.db.connect()) as db,db:
                    db.execute('INSERT INTO receipts VALUES(?,?)',(key,json.dumps({'fingerprint':fingerprint,'source_fingerprint':source_hash,'state':'STARTED','output':None})))
        started=stamp();clock=time.perf_counter()
        with execution({'device':'cpu','threads':4,'cpu_jobs':4}):output=self._dispatch(operation,params,run,key)
        from .workflow_returns import pipeline_for
        output['_execution']={'operation':operation,'started':started,'finished':stamp(),'elapsed_seconds':round(time.perf_counter()-clock,4),
                              'compute':'Local CPU · 4 threads','model':params.get('model',params.get('family')),
                              'input_fields':sorted(params),'pipeline':pipeline_for(operation)}
        from .workflow_coordination import specialist_for
        specialist=specialist_for(operation)
        with closing(self.db.connect()) as db,db:
            db.execute('INSERT OR REPLACE INTO receipts VALUES(?,?)',(key,json.dumps({'fingerprint':fingerprint,'output':output,'state':'COMPLETED','specialist_id':specialist},allow_nan=False)))
            self.db.event(db,action='SPECIALIST_RESULT',run_id=run['id'],operation_key=key,
                          specialist_id=specialist,operation=operation,output_fields=sorted(output))
        return output

    def _dispatch(self,op,p,run,key):
        from .discovery.workflows import load_model,train_candidate,pending_from_proposal
        from .discovery.calibration import disagreement
        if op=='manual.checkpoint':
            if p.get('sample_ids'):self._samples(p['sample_ids'],run)
            return {'instruction':str(p['instruction']),'awaiting_manual_action':True,
                    **{k:copy.deepcopy(p[k]) for k in ['device_id','capability','sample_ids','equipment_type','approval_required','return_fields','plan'] if k in p}}
        if op=='records.training_pool':
            from .discovery.validation import usable
            historical=run['selected_ids'] if p.get('include_history',False) else []
            ids=list(dict.fromkeys(historical+p.get('sample_ids',[])))
            sources=self._samples(ids,run);eligible=usable(sources,self.store.origin)
            eligible_ids={s['sample_id'] for s in eligible}
            excluded=[]
            for sample in sources:
                if sample['sample_id'] in eligible_ids:continue
                try:query(sample);reason='No quantified ICP or resolved structural target.'
                except (ValueError,TypeError) as exc:reason=str(exc)
                excluded.append({'sample_id':sample['sample_id'],'reason':reason})
            return {'sample_ids':[s['sample_id'] for s in eligible],'source_count':len(sources),'eligible_count':len(eligible),
                    'historical_count':sum(s['sample_id'] in historical for s in eligible),
                    'returned_count':sum(s['sample_id'] not in historical for s in eligible),'excluded':excluded,'fingerprint':digest(eligible)}
        if op=='knowledge.search':
            from .discovery.knowledge import public_documents,history_documents,search
            docs=public_documents()+ (history_documents(self._samples(None,run)) if p.get('include_history') else [])
            return {'evidence':search(docs,p['query'],int(p.get('limit',5)))}
        if op=='inverse.design':
            from .discovery.inverse import candidates,rank
            elements=p.get('elements',run['context']['elements']);name=p.get('model','Fixed prior')
            count=int(p.get('count',64));size=int(p.get('batch_size',6))
            if not 1<=size<=48 or not 1<=count<=512:raise ValueError('Workflow design supports 1-48 experiments and at most 512 candidates.')
            bounds=p.get('bounds_mmol',{e:[.2,2.] for e in elements})
            pool=candidates(run['context'],elements,bounds,p.get('temperature',[800,1600]),p.get('pulses',[1,2,4]),count,int(p.get('seed',42)),composition_mode=p.get('composition_mode','independent'))
            model=load_model(self.store.root,self.store.origin,name,self.store.all('sample'))
            predictions=model.predict(pool)
            rows=rank(pool,predictions,p.get('target_total',[0,100]),float(p.get('min_probability',0)),size,bool(p.get('allow_extrapolation',False)))
            if not rows:raise ValueError('No candidate satisfies the declared support and probability constraints.')
            return {'design':{'rows':rows,'model_id':model.model_id,'model_name':name,
                    'training_keys':getattr(model,'training_keys',[]) or [identity(s) for s in getattr(model,'training_samples',[])]},
                    'proposed_experiments':len(rows),'model_id':model.model_id}
        if op=='campaign.plan':
            design=p['design'];rows=design['rows']
            if not any(s['operation']=='inverse.design' and s['state']=='COMPLETED' and digest(s['output']['design'])==digest(design) for s in self.db.steps(run['id'])):
                raise ValueError('Freeze only numerical forecasts produced by an inverse-design step in this run.')
            if not rows or any(r['recipe']['origin']!=self.store.origin or r['prediction']['model_id']!=design['model_id'] for r in rows):raise ValueError('Design model or data space differs.')
            frozen=next((r for r in self.ledger.records('round') if r['model_metadata'].get('workflow_operation_key')==key),None)
            frozen=frozen or self.ledger.freeze(self.store.all('sample'),{'model_id':design['model_id'],'model_name':design['model_name'],'workflow_operation_key':key})
            proposal=next((r for r in self.ledger.records('proposal') if r['round_id']==frozen['id']),None)
            proposal=proposal or self.ledger.propose(frozen['id'],[r['recipe'] for r in rows],[r['prediction'] for r in rows],design['training_keys'])
            ids=pending_from_proposal(self.store,self.ledger,proposal['id'])
            return {'proposal_id':proposal['id'],'round_id':frozen['id'],'sample_ids':ids,'record_keys':{i:identity(self.store.get('sample',i)) for i in ids}}
        if op=='campaign.evaluate':
            rounds={r['id'] for r in self.ledger.records('round') if r['model_metadata'].get('workflow_operation_key','').startswith(run['id']+':')}
            return {'evaluation':[r for r in self.ledger.evaluate(self.store.all('sample')) if r['round_id'] in rounds]}
        if op in {'device.synthesize','device.characterize'}:
            if op=='device.synthesize':samples=self._samples(p['sample_ids'],run);capability='synthesize';synthesis=None
            else:synthesis=p['synthesis'];samples=self._samples(synthesis['sample_ids'],run);capability='characterize'
            receipt=self.devices.submit(p['device_id'],capability,samples,key,synthesis,p.get('modalities'))
            return {'receipt':receipt} if capability=='synthesize' else {'acquisition':receipt}
        if op=='characterization.analyze':return {'evidence':self._analyze(p['measurements'],run,key)}
        if op=='characterization.fixture_review':return self._fixture_review(p['evidence'],run,key)
        if op=='characterization.commit':
            from .discovery.characterization import commit_assessment,evidence_summary
            samples=self._samples(p['sample_ids'],run);events=[]
            for sample in samples:
                state=p['state']
                if state=='FROM_REVIEWED_EVIDENCE':state=evidence_summary(sample,self.ledger)['recommended_screening_state']
                events.append(commit_assessment(self.store,self.ledger,sample['sample_id'],state,p['reviewer'],p['criteria'],bool(p.get('override',False))))
            return {'assessment_ids':[r['id'] for r in events],'sample_ids':[s['sample_id'] for s in samples]}
        if op=='model.train':
            samples=self._samples(p.get('sample_ids'),run);epochs=int(p.get('epochs',50));family=p.get('family','SAC-Gen-Set')
            if not 1<=epochs<=500:raise ValueError('Workflow training supports 1-500 epochs.')
            result=train_candidate(self.store.root,samples,self.store.origin,family,epochs,int(p.get('seed',42)),operation_key=key)
            if not any(r.get('workflow_operation_key')==key for r in self.ledger.records('training')):
                self.ledger.add('training',{'origin':self.store.origin,'candidate':result,'workflow_run_id':run['id'],'workflow_operation_key':key})
            from .discovery.validation import usable
            return {'candidate':result,'family':family,'source_specimens':len(samples),'training_specimens':len(usable(samples,self.store.origin))}
        if op=='validation.run':
            from .discovery.validation import validate
            existing=next((r for r in self.ledger.records('validation') if r.get('workflow_operation_key')==key),None)
            if existing:return existing['result']
            result=validate(self._samples(p.get('sample_ids'),run),self.store.origin,int(p.get('folds',5)),int(p.get('epochs',20)),include_sac=bool(p.get('include_sac',False)))
            self.ledger.add('validation',{'origin':self.store.origin,'mode':'Grouped model validation','result':result,'workflow_run_id':run['id'],'workflow_operation_key':key})
            return result
        if op=='model.compare':
            samples=self._samples(p.get('sample_ids'),run)
            recipes=p.get('recipes') or [query(s) for s in samples]
            names=p.get('models',['Fixed prior','SAC-Gen-Set'])
            if not isinstance(names,list) or not 1<=len(names)<=6:raise ValueError('Compare 1-6 registered models.')
            forecasts={n:load_model(self.store.root,self.store.origin,n,self.store.all('sample')).predict(recipes) for n in names}
            return {'disagreement':disagreement(recipes,forecasts),'metrics':{n:prediction_metrics(samples,ps) for n,ps in forecasts.items()} if not p.get('recipes') else {},
                    'interpretation':'Observed-recipe diagnostic; may be in-sample. Protected validation is a separate operation.'}
        if op=='hypothesis.register':
            from .discovery.hypotheses import contrasts,register
            sample=self._samples([p['sample_id']],run)[0];model=load_model(self.store.root,self.store.origin,p.get('model','Fixed prior'),self.store.all('sample'))
            recipes=contrasts(sample,p['factor'],p['values']);card=register(recipes,model.predict(recipes),p['factor'],p['statement'])
            existing=next((r for r in self.ledger.records('hypothesis') if r.get('workflow_operation_key')==key),None)
            item=existing or self.ledger.add('hypothesis',{'origin':self.store.origin,'card':card,'workflow_operation_key':key})
            return {'hypothesis_id':item['card']['hypothesis_id'],'card':item['card']}
        raise ValueError('Unsupported workflow operation.')

    def _analyze(self,acquisition,run,key):
        from .discovery.characterization import xrd_analysis,microscopy_analysis,register_artifact
        if 'artifact_ids' in acquisition:
            requested=acquisition['artifact_ids'];artifacts={a['id']:a for a in self.ledger.records('characterization')}
            rows=[artifacts[i] for i in requested]
            self._samples([a['sample_id'] for a in rows],run)
            return {'artifact_ids':requested,'sample_ids':list(dict.fromkeys(a['sample_id'] for a in rows)),'simulated':False}
        measurements,job_ids=self._join_acquisitions(acquisition)
        acquisition_id=job_ids[0] if len(job_ids)==1 else 'bundle_'+digest(sorted(job_ids))[:24]
        evidence=[]
        selected=self._samples([m['sample_id'] for m in measurements],run)
        for measurement,sample in zip(measurements,selected):
            if identity(sample)!=measurement['record_key'] or recipe_id(sample)!=measurement['recipe_key']:raise ValueError('Measurement identity or recipe changed.')
            for field in measurement['files']:
                path=(self.store.root/field['relative_path']).resolve()
                if not path.is_relative_to((self.store.root/'workflow_simulation'/'demo').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=field['sha256']:raise ValueError('Simulator source path or checksum differs.')
        existing_by_token={r['metadata'].get('workflow_operation_key'):r for r in self.ledger.records('characterization')}
        updates=[]
        for measurement,sample in zip(measurements,selected):
            if identity(sample)!=measurement['record_key'] or recipe_id(sample)!=measurement['recipe_key']:raise ValueError('Measurement identity or recipe changed.')
            prior_job=sample.get('simulation',{}).get('acquisition_job_id')
            if prior_job and prior_job!=acquisition_id:raise ValueError('A different acquisition already populated this sample.')
            if not prior_job:
                sample.update(batch_id='SIM_'+acquisition_id,sample_condition='RECOVERED',result_reported_at=stamp(),simulation={'acquisition_job_id':acquisition_id,'source_job_ids':job_ids,'workflow_run_id':run['id'],'synthetic':True})
                if measurement['icp']:sample['icp']=measurement['icp']
                updates.append(sample)
        if updates:self.store.import_bundle(updates,[],overwrite=True)
        for measurement,sample in zip(measurements,selected):
            for field in measurement['files']:
                token=key+':'+measurement['record_key']+':'+field['modality']
                existing=existing_by_token.get(token)
                if existing:artifact=existing
                else:
                    path=(self.store.root/field['relative_path']).resolve()
                    content=path.read_bytes();modality=field['modality']
                    analysis=xrd_analysis(content)[0] if modality=='XRD' else microscopy_analysis(content,modality,nm_per_pixel=.02 if modality=='STEM' else 1.)
                    artifact=register_artifact(self.store,self.ledger,sample['sample_id'],modality,field['file_name'],content,analysis,
                        {'workflow_operation_key':token,'simulated':True,'source_job_id':field['source_job_id'],'fixture_structural_PASS':measurement['fixture_structural_PASS']})
                evidence.append(artifact['id'])
        return {'artifact_ids':evidence,'sample_ids':[m['sample_id'] for m in measurements],'simulated':True,'source_job_id':job_ids[0],'source_job_ids':job_ids}

    def _join_acquisitions(self,acquisition):
        """Verify every instrument receipt before combining measurements or writing results."""
        if not isinstance(acquisition,dict):raise ValueError('Use a verified acquisition or an acquisitions list.')
        receipts=acquisition.get('acquisitions',[acquisition])
        if self.store.origin!='DEMO' or not isinstance(receipts,list) or not 1<=len(receipts)<=16:raise ValueError('Use 1-16 verified DEMO instrument receipts.')
        merged={};job_ids=[];keys=None;synthesis_ids=set()
        for receipt in receipts:
            if not isinstance(receipt,dict):raise ValueError('Invalid instrument receipt.')
            job=self.db.get('jobs',receipt.get('job_id',''))
            if not job or job['state']!='COMPLETED' or job['capability']!='characterize' or digest(job['output'])!=digest(receipt):raise ValueError('Use registered artifacts or a verified DEMO acquisition receipt.')
            if job['id'] in job_ids:raise ValueError('An instrument receipt is repeated.')
            job_ids.append(job['id']);rows=receipt['measurements'];current={m['record_key'] for m in rows}
            if len(current)!=len(rows) or keys is not None and keys!=current:raise ValueError('Instrument acquisitions must cover the same distinct specimens.')
            keys=current
            for row in rows:
                if row.get('synthesis_job_id'):synthesis_ids.add(row['synthesis_job_id'])
                result=merged.setdefault(row['record_key'],{**copy.deepcopy(row),'icp':[],'files':[]})
                if any(result[k]!=row[k] for k in ['sample_id','recipe_key','fixture_structural_PASS']):raise ValueError('Instrument acquisitions disagree on specimen or fixture identity.')
                if result['icp'] and row['icp']:raise ValueError('More than one ICP result exists for a specimen; choose the intended instrument receipt.')
                if row['icp']:result['icp']=copy.deepcopy(row['icp'])
                for field in row['files']:
                    if any(f['modality']==field['modality'] for f in result['files']):raise ValueError('Duplicate modality results require explicit selection.')
                    result['files'].append({**copy.deepcopy(field),'source_job_id':job['id']})
        if len(synthesis_ids)>1:raise ValueError('Acquisitions must follow the same synthesis receipt.')
        return list(merged.values()),job_ids

    def _fixture_review(self,evidence,run,key):
        from .discovery.characterization import review_artifact,commit_assessment
        if self.store.origin!='DEMO' or not evidence.get('simulated'):raise ValueError('Fixture review is for simulated DEMO evidence only; real measurements require reviewed observations.')
        allowed={}
        for job_id in evidence.get('source_job_ids',[evidence.get('source_job_id','')]):
            job=self.db.get('jobs',job_id)
            if not job or job['state']!='COMPLETED' or job['capability']!='characterize' or not job.get('output',{}).get('simulated'):raise ValueError('Missing simulated acquisition receipt.')
            allowed.update({(job_id,m['record_key']):m for m in job['output']['measurements']})
        ids=evidence['artifact_ids']
        artifacts={r['id']:r for r in self.ledger.records('characterization')};self._samples(evidence['sample_ids'],run)
        reviewed={(r['artifact_id'],r['reviewer']) for r in self.ledger.records('characterization_review')}
        for ident in ids:
            artifact=artifacts[ident];metadata=artifact['metadata'];truth=allowed.get((metadata.get('source_job_id'),artifact['record_key']))
            if not metadata.get('simulated') or truth is None:raise ValueError('Evidence does not belong to the simulated job.')
            passed=truth['fixture_structural_PASS'];modality=artifact['modality']
            observation={'XRD':'NO_RESOLVED_TARGET_CRYSTALLINE_PEAKS' if passed else 'TARGET_CRYSTALLINE_PHASE_DETECTED',
                         'STEM':'ISOLATED_SITE_CANDIDATES_CONFIRMED' if passed else 'CLUSTERS_OBSERVED',
                         'SEM':'MORPHOLOGY_RECORDED' if passed else 'AGGREGATES_OBSERVED'}[modality]
            if (ident,'SIMULATED_FIXTURE_REVIEWER') not in reviewed:review_artifact(self.store,self.ledger,ident,observation,'SIMULATED_FIXTURE_REVIEWER','Synthetic generator truth validates workflow plumbing, not recognition accuracy.')
        truth_by_key={record_key:measurement for (_,record_key),measurement in allowed.items()}
        events=[commit_assessment(self.store,self.ledger,ident,'PASS' if truth_by_key[identity(self.store.get('sample',ident))]['fixture_structural_PASS'] else 'FAIL',
            'SIMULATED_FIXTURE_REVIEWER','DEMO fixture outcome; not an experimental observation.') for ident in evidence['sample_ids']]
        return {'assessment_ids':[e['id'] for e in events],'sample_ids':evidence['sample_ids'],'simulated':True}

    def conversation(self,action,workflow_id=None,run_id=None,spec_json=None):
        if action=='capabilities':return self.capabilities()
        if action=='templates':
            from .workflow_templates import catalog
            return {'templates':catalog(self.store.origin)}
        if action=='create_template':
            request=json.loads(spec_json or '{}')
            if not isinstance(request,dict) or set(request)-{'template_id','options'}:raise ValueError('Use template_id and options only.')
            return self.instantiate_template(request.get('template_id'),request.get('options'))
        if action=='list':return {'workflows':[{'id':w['id'],'name':w['spec']['name']} for w in sorted(self.db.all('definitions'),key=lambda r:r['created'],reverse=True)[:20]],
                                  'runs':[self.db.summary(r['id']) for r in sorted(self.db.all('runs'),key=lambda r:r['created'],reverse=True)[:20]],'limit':20}
        if action in {'create_demo','save'}:
            if action=='create_demo':
                if self.store.origin!='DEMO':raise ValueError('Switch to DEMO for the simulated closed loop.')
                self.devices.defaults();spec=simulation_template(['Fe'])
            else:spec=json.loads(spec_json)
            result=self.db.save(spec);return {'workflow_id':result['id'],'name':result['spec']['name'],'steps':len(result['spec']['steps'])}
        if action=='start':return {'run_id':self.start(workflow_id)['id']}
        self.authorize_run(run_id)
        if action=='run':self.db.run_ready(run_id,self.execute,max_steps=48)
        elif action=='step':self.db.advance(run_id,self.execute)
        elif action in {'pause','resume','retry','cancel','recover'}:
            self.db.control(run_id,action)
            if action=='recover':self.devices.reconcile_expired(run_id)
        elif action!='status':raise ValueError('Unsupported workflow action.')
        result=self.db.summary(run_id)
        if action=='status':
            from .workflow_graph import snapshot
            result['coordination']=snapshot(self,run_id=run_id)['coordination']
        return result

    def public_result(self,value):
        if isinstance(value,dict):return {k:self.public_result(v) for k,v in value.items()}
        if isinstance(value,list):return [self.public_result(v) for v in value]
        if isinstance(value,str):return value.replace(str(self.store.root),'[local data]').replace(self.store.root.as_posix(),'[local data]')
        return value

    def authorize_run(self,run_id):
        run=self.db.get('runs',run_id)
        if not run or run['origin']!=self.store.origin:raise ValueError('Unknown workflow run in this data space.')
        if set(run['selected_ids'])-set(self.selected_ids):raise ValueError('This run uses records outside the current conversation selection.')


def local_workflow_message(text,service):
    """Explicit local commands share the same workflow service as LLM tools."""
    if not text.strip().lower().startswith('/workflow'):return None
    parts=text.strip().split();action=parts[1].lower() if len(parts)>1 else 'capabilities'
    if action=='demo':action='create_demo'
    if action=='template':
        if len(parts)!=3:raise ValueError('Use /workflow template <template_id>. Inspect /workflow templates first.')
        return service.instantiate_template(parts[2],{'register_resources':True})
    if action=='start':return service.conversation(action,workflow_id=parts[2] if len(parts)>2 else None)
    if action in {'run','step','status','pause','resume','cancel','retry','recover'}:return service.conversation(action,run_id=parts[2] if len(parts)>2 else None)
    if action not in {'capabilities','list','create_demo','templates'}:raise ValueError('Use /workflow capabilities, templates, template <template_id>, demo, list, start <workflow_id>, or run/status/step/pause/resume/retry/cancel <run_id>.')
    return service.conversation(action)
