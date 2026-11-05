"""Dependency, priority and device-aware scheduling with explicit time estimates."""
import copy
from datetime import datetime,timezone
import math
from .device_catalog import modalities_of

DURATIONS={'knowledge.search':10,'inverse.design':30,'campaign.plan':5,'campaign.evaluate':5,
           'model.compare':60,'model.train':120,'validation.run':180,'hypothesis.register':15,
           'characterization.analyze':60,'characterization.fixture_review':30,'characterization.commit':10,
           'device.synthesize':120,'device.characterize':180,'manual.checkpoint':60}


def validate_schedule(value):
    if not isinstance(value,dict) or set(value)-{'priority','estimated_seconds'}:raise ValueError('Scheduling supports priority and estimated_seconds.')
    priority=value.get('priority',50);duration=value.get('estimated_seconds',60)
    if type(priority) is not int or not 0<=priority<=100:raise ValueError('Priority must be an integer from 0 to 100.')
    if type(duration) not in (int,float) or not math.isfinite(duration) or not 1<=duration<=86400:raise ValueError('Estimated duration must be 1-86400 seconds.')
    return value


def critical_paths(spec):
    children={n['id']:[] for n in spec['steps']};result={}
    for node in spec['steps']:
        for dep in node.get('depends_on',[]):children[dep].append(node['id'])
    for node in reversed(spec['steps']):
        duration=node.get('schedule',{}).get('estimated_seconds',DURATIONS.get(node['operation'],60))
        result[node['id']]=float(duration)+max((result[c] for c in children[node['id']]),default=0.)
    return result


def device_requirement(node,outputs=None):
    manual=node['operation']=='manual.checkpoint' and node.get('parameters',{}).get('device_id')
    if not node['operation'].startswith('device.') and not manual:return None,None
    device=node.get('parameters',{}).get('device_id')
    capability=node.get('parameters',{}).get('capability') if manual else 'synthesize' if node['operation']=='device.synthesize' else 'characterize'
    if isinstance(device,dict):
        try:
            from .workflow_engine import resolve
            device=resolve(device,outputs or {})
        except (KeyError,IndexError,TypeError,ValueError):return None,'Unresolved device reference'
    if isinstance(capability,dict):
        try:
            from .workflow_engine import resolve
            capability=resolve(capability,outputs or {})
        except (KeyError,IndexError,TypeError,ValueError):return device,'Unresolved capability reference'
    return device,capability


def resource_check(node,devices,locks,origin,outputs=None):
    ident,capability=device_requirement(node,outputs)
    if capability is None:return True,None,None
    lookup={d['id']:d for d in devices};device=lookup.get(ident)
    if not device:return False,ident,'Device is not registered'
    if capability not in device.get('capabilities',[]):return False,ident,'Required capability is unavailable'
    required=node.get('parameters',{}).get('modalities',[])
    if isinstance(required,dict):
        try:
            from .workflow_engine import resolve
            required=resolve(required,outputs or {})
        except (KeyError,IndexError,TypeError,ValueError):return False,ident,'Unresolved measurement modality reference'
    if isinstance(required,list) and any(m not in modalities_of(device) for m in required):return False,ident,'Required measurement modality is unavailable'
    if node['operation']=='manual.checkpoint':
        if device['mode']=='SIMULATOR' and origin!='DEMO':return False,ident,'Simulator resources cannot represent a REAL manual operation'
        if ident in locks:return False,ident,'Device is occupied by '+str(locks[ident])
        return True,ident,None
    if device['mode']=='INVENTORY_ONLY':return False,ident,'Planning inventory only; no execution adapter is connected'
    if device['mode']!='SIMULATOR':return False,ident,'MCP connection is not active'
    if origin!='DEMO':return False,ident,'Simulator acquisition is restricted to DEMO'
    if ident in locks:return False,ident,'Device is occupied by '+str(locks[ident])
    return True,ident,None


def choose_ready(spec,steps,devices,locks,origin):
    states={s['id']:s for s in steps};paths=critical_paths(spec)
    outputs={s['id']:s['output'] for s in steps if s['state']=='COMPLETED'}
    choices=[]
    for index,node in enumerate(spec['steps']):
        if states[node['id']]['state']!='PENDING' or not all(states[d]['state']=='COMPLETED' for d in node.get('depends_on',[])):continue
        available,_,_=resource_check(node,devices,locks,origin,outputs)
        if available:choices.append((node.get('schedule',{}).get('priority',50),paths[node['id']],-index,node))
    return max(choices,key=lambda item:item[:3])[3] if choices else None


def schedule_rows(spec,steps,devices,locks,origin,run_state='READY'):
    """A conservative single-worker plan; durations are declarations, not telemetry."""
    from .workflow_engine import OPERATIONS
    states={s['id']:s for s in steps};paths=critical_paths(spec)
    outputs={s['id']:s['output'] for s in steps if s['state']=='COMPLETED'}
    todo=list(spec['steps']);placed={};rows=[];worker_free=0.
    while todo:
        candidates=[n for n in todo if all(d in placed for d in n.get('depends_on',[]))]
        node=max(candidates,key=lambda n:(n.get('schedule',{}).get('priority',50),paths[n['id']],-spec['steps'].index(n)))
        todo.remove(node);step=states[node['id']];state=step['state'];deps=node.get('depends_on',[])
        available,device,reason=resource_check(node,devices,locks,origin,outputs)
        duration=float(node.get('schedule',{}).get('estimated_seconds',DURATIONS.get(node['operation'],60)))
        actual=None
        if step.get('started'):
            try:
                begin=datetime.fromisoformat(step['started']);end=datetime.fromisoformat(step['finished']) if step.get('finished') and step['state']!='WAITING' else datetime.now(timezone.utc)
                actual=max(0.,(end-begin).total_seconds())
            except (ValueError,TypeError):pass
        dependencies_done=all(states[d]['state']=='COMPLETED' for d in deps)
        if state=='PENDING':
            state='READY' if dependencies_done and available else 'DEVICE_BUSY' if dependencies_done and reason and 'occupied' in reason else 'BLOCKED' if dependencies_done and not available else 'WAITING_DEPENDENCIES'
        if state in {'PENDING','READY','WAITING_DEPENDENCIES','DEVICE_BUSY','BLOCKED'} and run_state in {'PAUSED','CANCELLED','FAILED','WAITING'}:
            state='BLOCKED' if run_state in {'FAILED','WAITING'} else run_state
            reason='Waiting for manual acknowledgement' if run_state=='WAITING' else 'Run is '+run_state.lower()
        dependency_ends=[placed[d] for d in deps]
        if step['state']=='COMPLETED':start=end=0.
        elif state in {'FAILED','CANCELLED','WAITING','BLOCKED','DEVICE_BUSY'} or any(v is None for v in dependency_ends):start=end=None
        else:
            start=max([worker_free]+dependency_ends)
            remaining=max(1.,duration-(actual or 0.)) if state=='RUNNING' else duration
            end=start+remaining;worker_free=end
        placed[node['id']]=end
        modalities=node.get('parameters',{}).get('modalities',[])
        action='Acquire '+', '.join(modalities)+' measurements' if node['operation']=='device.characterize' and isinstance(modalities,list) and modalities else OPERATIONS[node['operation']][0]
        rows.append({'step_id':node['id'],'action':action,'operation':node['operation'],
            'state':state,'device_id':device,'priority':node.get('schedule',{}).get('priority',50),
            'estimated_seconds':duration,'actual_seconds':actual,'critical_path_seconds':paths[node['id']],
            'estimated_start_seconds':start,'estimated_end_seconds':end,'reason':reason or ('Waiting for dependencies' if not dependencies_done else ''),
            'dependencies':deps})
    return rows


def bind_devices(spec,bindings,devices,settings=None):
    """Create a revised definition; frozen run specifications are never modified."""
    from .workflow_engine import validate_spec
    result=copy.deepcopy(spec);lookup={d['id']:d for d in devices};steps={n['id']:n for n in result['steps']}
    for step_id,device_id in bindings.items():
        if step_id not in steps or device_id not in lookup:raise ValueError('Choose a known step and registered device.')
        _,capability=device_requirement(steps[step_id])
        if steps[step_id]['operation']=='manual.checkpoint':capability=steps[step_id]['parameters'].get('capability')
        if capability is None or capability not in lookup[device_id]['capabilities']:raise ValueError('Device capability does not match the step.')
        required=steps[step_id].get('parameters',{}).get('modalities',[])
        if isinstance(required,list) and any(m not in modalities_of(lookup[device_id]) for m in required):raise ValueError('Device measurement modality does not match the step.')
        steps[step_id]['parameters']['device_id']=device_id
    for step_id,value in (settings or {}).items():
        if step_id not in steps:raise ValueError('Unknown scheduling step.')
        steps[step_id]['schedule']=validate_schedule(value)
    result['version']=int(result.get('version',1))+1;result['name']=result['name'][:108]+' · revised'
    return validate_spec(result)
