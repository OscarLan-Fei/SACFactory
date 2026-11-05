"""Conversational template state, immutable revisions and bounded run controls."""
import copy
import json
import math
from contextlib import closing

from .discovery.common import digest
from .workflow_templates import catalog,PREPARATION_DEFAULTS


def merge_options(base,patch):
    result=copy.deepcopy(base)
    for key,value in patch.items():
        if value is None:result.pop(key,None)
        elif isinstance(value,dict) and isinstance(result.get(key),dict):result[key]=merge_options(result[key],value)
        else:result[key]=copy.deepcopy(value)
    if 'elements' in patch and 'bounds_mmol' not in patch and 'bounds_mM' not in patch:
        result['bounds_mmol']={e:result.get('bounds_mmol',{}).get(e,[.2,2.4]) for e in result['elements']}
    if 'bounds_mM' in result:
        if 'bounds_mmol' in patch:raise ValueError('Specify bounds_mM or bounds_mmol, not both in one edit.')
        volume=result.get('solvent_volume_mL',20.)
        try:
            bounds=result.pop('bounds_mM')
            if type(volume) not in (int,float) or not math.isfinite(volume) or volume<=0:raise ValueError()
            if any(not isinstance(limits,list) or len(limits)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in limits) for limits in bounds.values()):raise ValueError()
            result['bounds_mmol']={**result.get('bounds_mmol',{}),**{e:[v*volume/1000 for v in limits] for e,limits in bounds.items()}}
        except (ValueError,TypeError,AttributeError):raise ValueError('Use element-specific finite concentration bounds and a positive solvent volume.') from None
    return result


def scoped_state(current,origin,selected_ids,discussion='initial'):
    scope=digest([origin,sorted(selected_ids),discussion])
    if isinstance(current,dict) and current.get('scope')==scope:return current
    if (isinstance(current,dict) and current.get('origin')==origin and current.get('discussion')==discussion
            and set(current.get('selected_ids',[]))<=set(selected_ids) and any(current.get(k) for k in ['preview','workflow_id','run_id'])):
        current['scope']=scope
        return current
    return {'scope':scope,'origin':origin,'discussion':discussion}


class WorkflowChat:
    def __init__(self,service,state=None):
        self.service=service;self.state=state if state is not None else {'origin':service.store.origin};self.cache={}
        if self.state.get('origin')!=service.store.origin:raise ValueError('Conversation workflows cannot cross data spaces.')
        self.state['selected_ids']=list(service.selected_ids);self.state['context']=copy.deepcopy(service.context)

    @staticmethod
    def payload(raw):
        if raw is None:return {}
        if not isinstance(raw,str) or len(raw)>120000:raise ValueError('Use a JSON object smaller than 120 KB.')
        value=json.loads(raw)
        if not isinstance(value,dict):raise ValueError('Use a JSON object.')
        return value

    def context(self):
        draft=self.state.get('preview')
        result={k:self.state.get(k) for k in ['workflow_id','run_id'] if self.state.get(k)}
        if draft:
            result.update(template_id=draft['template_id'],preview_name=draft['options'].get('name'),
                          parameters={k:v for k,v in draft['options'].items() if k not in {'sample_ids','artifact_ids'}},
                          selected_records=len(draft['selected_ids']),saved=False)
        return result

    def _source(self,template_id=None,workflow_id=None):
        if workflow_id:
            row=self.service.inspect_workflow(workflow_id)
            if not row.get('template'):raise ValueError('This older/imported definition has no template parameters. Inspect it with workflow_control get, then save an edited definition as a new workflow.')
            if template_id and template_id!=row['template']['id']:raise ValueError('The selected template does not match this saved workflow.')
            return row['template']['id'],copy.deepcopy(row['template']['options']),row['id']
        draft=self.state.get('preview')
        if draft and (template_id is None or template_id==draft['template_id']):
            return draft['template_id'],copy.deepcopy(draft['options']),draft.get('revision_of')
        if template_id:return template_id,{'register_resources':True,**{field:self.service.context.get(field,default) for field,default in PREPARATION_DEFAULTS.items()}},None
        if self.state.get('workflow_id'):return self._source(workflow_id=self.state['workflow_id'])
        raise ValueError('Select a workflow template first.')

    def template(self,action,template_id=None,workflow_id=None,options_json=None):
        patch=self.payload(options_json)
        if action=='list':return {'origin':self.service.store.origin,'templates':catalog(self.service.store.origin),'current':self.context()}
        if action=='inspect':
            row=self.service.inspect_workflow(workflow_id or self.state.get('workflow_id'))
            if self.state.get('workflow_id')!=row['id']:self.state.update(workflow_id=row['id'],run_id=None);self.state.pop('preview',None)
            result={'workflow_id':row['id'],'name':row['spec']['name'],'template':row.get('template'),'spec':row['spec'],'board':{'kind':'definition','workflow_id':row['id']}}
            self.state['last_result']=result
            return result
        if action not in {'preview','create','revise'}:raise ValueError('Choose list, preview, create, revise or inspect.')
        cache_key=digest(['template',action,template_id,workflow_id,patch])
        if action in {'create','revise'} and cache_key in self.cache:return self.cache[cache_key]
        source_id,options,revision_of=self._source(template_id,workflow_id)
        options=merge_options(options,patch)
        if action=='revise' and 'name' not in patch:options['name']=(options.get('name',source_id)[:108]+' · revised')
        plan=self.service.template_plan(source_id,options)
        if action=='preview':
            self.state['preview']={'template_id':source_id,'options':plan['options'],'revision_of':revision_of,
                'selected_ids':list(self.service.selected_ids),'context':copy.deepcopy(self.service.context)}
            result={'template_id':source_id,'name':plan['spec']['name'],'saved':False,'steps':len(plan['spec']['steps']),
                    'parameters':plan['options'],'resources_to_register':[{k:d.get(k) for k in ['id','name','mode','modalities']} for d in plan['resources_to_register']],
                    'board':{'kind':'preview'},'note':'Preview only. No experiment, resource registration or training was started.'}
        else:
            result=self.service.instantiate_template(source_id,options,revision_of=revision_of)
            self.state.update(workflow_id=result['workflow_id'],run_id=None);self.state.pop('preview',None)
            result.update(saved=True,started=False,board={'kind':'definition','workflow_id':result['workflow_id']},parameters=plan['options'])
            self.cache[cache_key]=result
        self.state['last_result']=result
        return result

    def devices(self,action,definition_json=None):
        request=self.payload(definition_json)
        if action=='catalog':
            rows=self.service.devices.catalog();query=str(request.get('search','')).casefold();category=request.get('category')
            return {'profiles':[d for d in rows if (not category or d['category']==category) and (not query or query in (d['name']+' '+d['id']).casefold())]}
        if action=='list':
            with closing(self.service.db.connect()) as db:locks=dict(db.execute('SELECT device_id,job_id FROM locks'))
            return {'devices':[{**{k:d.get(k) for k in ['id','name','category','equipment_type','mode','capabilities','modalities','limits','connection_state']},
                               'busy':d['id'] in locks} for d in self.service.db.all('devices')]}
        if action!='register':raise ValueError('Choose list, catalog or register.')
        cache_key=digest(['device',request])
        if cache_key in self.cache:return self.cache[cache_key]
        profile=request.pop('profile_id',None)
        if profile:
            from .device_catalog import PROFILES,profile_device
            if profile not in PROFILES:raise ValueError('Choose a catalog profile ID.')
            definition={**profile_device(profile),**request}
        else:definition=request
        row=self.service.devices.register(definition)
        result={'device':{k:row.get(k) for k in ['id','name','category','mode','capabilities','modalities','limits','connection_state']},
                'note':'Device contract registered. Registration sends no instrument command.'}
        self.cache[cache_key]=result
        return result

    def control(self,action,workflow_id=None,run_id=None,spec_json=None):
        if action in {'run','step'}:action=action+'_async'
        workflow_id=workflow_id or self.state.get('workflow_id');run_id=run_id or self.state.get('run_id')
        if action=='get':
            return self.template('inspect',workflow_id=workflow_id)
        if action=='step_result':
            request=self.payload(spec_json)
            if set(request)!={'step_id'}:raise ValueError('Specify step_id.')
            return self.service.step_result(run_id,request['step_id'])
        if action=='create_template':
            request=self.payload(spec_json)
            if set(request)-{'template_id','options'}:raise ValueError('Use template_id and options only.')
            return self.template('create',template_id=request.get('template_id'),options_json=json.dumps(request.get('options',{})))
        if action=='start':
            if self.state.get('preview'):raise ValueError('Save the current preview before starting it.')
            self.service.inspect_workflow(workflow_id)
            key=digest(['start',workflow_id])
            if key in self.cache:return self.cache[key]
            result=self.service.conversation('start',workflow_id=workflow_id)
            self.state.update(workflow_id=workflow_id,run_id=result['run_id']);self.state.pop('preview',None)
            result.update(workflow_id=workflow_id,state='READY',board={'kind':'run','run_id':result['run_id']})
            self.cache[key]=result
        elif action in {'run_async','step_async','acknowledge'}:
            self.service.authorize_run(run_id)
            if action=='acknowledge':
                request=self.payload(spec_json)
                if not {'step_id','note'}<=set(request) or set(request)-{'step_id','note','returned_data'}:raise ValueError('Document step_id, the operator completion note and optional returned_data.')
                result=self.service.acknowledge(run_id,request['step_id'],request['note'],request.get('returned_data'))
            else:
                from .workflow_worker import launch,worker_status
                worker=worker_status(self.service)
                if worker['active'] and worker['run_id']==run_id:result={'run_id':run_id,'already_running':True}
                else:result=launch(self.service,run_id,1 if action=='step_async' else 48)
                result['status']=self.service.db.summary(run_id)
            self.state.update(run_id=run_id,workflow_id=self.service.db.get('runs',run_id)['workflow_id'])
            result['board']={'kind':'run','run_id':run_id}
        else:
            result=self.service.conversation(action,workflow_id=workflow_id,run_id=run_id,spec_json=spec_json)
            if result.get('workflow_id') and action in {'save','create_demo'}:
                self.state.update(workflow_id=result['workflow_id'],run_id=None);self.state.pop('preview',None)
                result['board']={'kind':'definition','workflow_id':result['workflow_id']}
            if run_id and action in {'run','step','status','pause','resume','retry','cancel','recover'}:
                self.state.update(run_id=run_id,workflow_id=self.service.db.get('runs',run_id)['workflow_id'])
                result['board']={'kind':'run','run_id':run_id}
        self.state['last_result']=result
        return result


def local_chat_workflow(text,controller):
    """Offline workflow commands use the same persistent conversation controller."""
    parts=text.strip().split(maxsplit=2)
    if not parts or parts[0].lower()!='/workflow':return None
    action=parts[1].lower() if len(parts)>1 else 'capabilities';rest=parts[2] if len(parts)>2 else ''
    if action in {'preview','create','template'}:
        if rest.startswith('{'):template_id=None;raw=rest
        else:
            bits=rest.split(maxsplit=1);template_id=bits[0] if bits else None;raw=bits[1] if len(bits)>1 else None
        return controller.template('create' if action in {'create','template'} else 'preview',template_id=template_id,options_json=raw)
    if action in {'edit','set','revise'}:return controller.template('revise' if action=='revise' else 'preview',options_json=rest or None)
    if action=='select':return controller.template('inspect',workflow_id=rest)
    if action=='devices':return controller.devices('list')
    if action=='catalog':return controller.devices('catalog',rest if rest.startswith('{') else json.dumps({'search':rest}))
    if action=='register':return controller.devices('register',rest)
    if action=='ack':
        if not rest.strip():raise ValueError('Use /workflow ack <step_id> <completion note>.')
        step,*note=rest.split(maxsplit=1)
        return controller.control('acknowledge',spec_json=json.dumps({'step_id':step,'note':note[0] if note else ''}))
    if action in {'start'}:return controller.control(action,workflow_id=rest or None)
    if action in {'run','step','status','pause','resume','cancel','retry','recover'}:return controller.control(action,run_id=rest or None)
    if action=='demo':return controller.control('create_demo')
    if action in {'capabilities','templates','list'}:return controller.control(action)
    raise ValueError('Use /workflow templates, preview <template_id>, edit {options}, create, start, run, status, devices or ack <step_id> <note>.')
