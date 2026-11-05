"""View models for the live device/task graph and scheduling table."""
from contextlib import closing
import math

from .workflow_engine import OPERATIONS,validate_spec
from .workflow_schedule import schedule_rows,device_requirement
from .device_catalog import CATEGORIES,CATALOG,PROFILES,category_of,modalities_of
from .workflow_coordination import coordination,specialist_for,SPECIALISTS


def snapshot(service,run_id=None,definition_id=None,bindings=None,include_catalog=False,draft_spec=None,planned_devices=None):
    devices=service.db.all('devices');jobs=service.db.all('jobs')
    registered={d['id'] for d in devices}
    devices+= [{**d,'connection_state':'PLANNED'} for d in (planned_devices or []) if d['id'] not in registered]
    with closing(service.db.connect()) as db:locks=dict(db.execute('SELECT device_id,job_id FROM locks'))
    if run_id:
        run=service.db.get('runs',run_id)
        if not run:raise ValueError('Unknown graph run.')
        spec=run['spec'];steps=service.db.steps(run_id);state=run['state'];scope=run_id
    elif definition_id:
        definition=service.db.get('definitions',definition_id)
        if not definition:raise ValueError('Unknown graph definition.')
        spec=definition['spec'];steps=[{'id':n['id'],'state':'PENDING','output':None} for n in spec['steps']];state='PREVIEW';scope=definition_id
    elif draft_spec is not None:
        spec=validate_spec(draft_spec);steps=[{'id':n['id'],'state':'PENDING','output':None} for n in spec['steps']];state='PREVIEW';scope='template:'+spec['name']
    else:spec={'name':'Laboratory equipment catalog' if include_catalog else 'Registered device network','steps':[]};steps=[];state='PREVIEW';scope='devices'
    schedule=schedule_rows(spec,steps,devices,locks,service.store.origin,state) if steps else []
    rows={r['step_id']:r for r in schedule};states={s['id']:s for s in steps};nodes=[];edges=[];depths={};counts={}
    for node in spec['steps']:
        key=node['id'];depth=1+max((depths[d] for d in node.get('depends_on',[])),default=-1);depths[key]=depth
        position=counts.get(depth,0);counts[depth]=position+1
        row=rows[key];manual=node['operation']=='manual.checkpoint'
        kind='device_step' if node['operation'].startswith('device.') or manual and node['parameters'].get('capability') else 'step'
        category=('Synthesis' if node['operation']=='device.synthesize' else 'Measurements' if node['operation'].startswith(('device.','characterization.')) else
                  'Learning' if node['operation'].startswith(('model.','validation.','records.')) else 'Knowledge' if node['operation'].startswith('knowledge.') else
                  'Feedback' if node['operation'].startswith('campaign.') else 'Planning')
        _,capability=device_requirement(node)
        if manual:
            capability=node['parameters'].get('capability')
            profile=PROFILES.get(node['parameters'].get('equipment_type'),{})
            category={'Synthesis equipment':'Synthesis','Characterization equipment':'Measurements','Performance testing':'Performance','General equipment':'Preparation'}.get(profile.get('category'),'Planning')
        modalities=node.get('parameters',{}).get('modalities',[])
        label='Acquire '+', '.join(modalities)+' measurements' if node['operation']=='device.characterize' and isinstance(modalities,list) and modalities else OPERATIONS[node['operation']][0]
        if manual:label='Operator: '+profile.get('name',node['id'].replace('_',' '))
        from .workflow_returns import pipeline_for,result_summary,model_names
        output=states[key].get('output') or {};execution=output.get('_execution',{})
        nodes.append({'id':'step:'+key,'step_id':key,'label':label,'kind':kind,'category':category,'required_modalities':modalities if isinstance(modalities,list) else [],
            'specialist':SPECIALISTS[specialist_for(node['operation'])]['name'],
            'model':model_names(output,node['parameters']),
            'compute':execution.get('compute'),'pipeline':execution.get('pipeline',pipeline_for(node['operation'])),
            'inputs':list(node['parameters']),'result':result_summary(output),
            'state':row['state'],'device_id':row['device_id'],'requires':capability,'priority':row['priority'],
            'reason':row['reason'],'dependencies':row['dependencies'],'estimated_seconds':row['estimated_seconds'],
            'actual_seconds':row['actual_seconds'],'x':300+(depth%4)*230,'y':110+(depth//4)*180+position*85})
        for dep in node.get('depends_on',[]):edges.append({'source':'step:'+dep,'target':'step:'+key,'kind':'dependency','label':'depends on'})
        if row['device_id']:edges.append({'source':'device:'+row['device_id'],'target':'step:'+key,'kind':'uses','label':'assigned device'})
    # Branches share a depth but need their own vertical space before the next
    # wrapped lane begins. Fixed-height lanes overlapped four-instrument panels.
    lane_heights={}
    for depth,count in counts.items():lane_heights[depth//4]=max(lane_heights.get(depth//4,0),count*90+70)
    for node in nodes:
        if 'step_id' not in node:continue
        depth=depths[node['step_id']];lane=depth//4
        original_position=(node['y']-110-(depth//4)*180)/85
        node['y']=110+sum(lane_heights[i] for i in range(lane))+original_position*90
    known={d['id'] for d in devices}
    assigned={r['device_id'] for r in schedule if r['device_id']}|set((bindings or {}).values())
    for i,device in enumerate(devices):
        if device['id']=='demo_characterizer' and device['id'] not in assigned:continue
        category=category_of(device);group_id='group:'+str(CATEGORIES.index(category))
        nodes.append({'id':'device:'+device['id'],'device_id':device['id'],'label':device['name'],'kind':'device','category':category,
            'group_id':group_id,'assigned':device['id'] in assigned,'modalities':modalities_of(device),'equipment_type':device.get('equipment_type','custom'),
            'state':'BUSY' if device['id'] in locks else device['connection_state'],'capabilities':device['capabilities'],
            'limits':device['limits'],'mode':device['mode'],'job_id':locks.get(device['id']),'x':45,'y':110+i*125})
    missing={r['device_id'] for r in schedule if r['device_id'] and r['device_id'] not in known}
    for i,ident in enumerate(sorted(missing)):
        nodes.append({'id':'device:'+ident,'device_id':ident,'label':ident,'kind':'device','category':'Unregistered equipment','state':'NOT_REGISTERED',
                      'capabilities':[],'limits':{},'mode':'MISSING','x':45,'y':110+(len(devices)+i)*125})
    for key,device_id in (bindings or {}).items():
        if key in rows and device_id in known:edges.append({'source':'device:'+device_id,'target':'step:'+key,'kind':'proposed','label':'draft binding for a new workflow'})
    if include_catalog:
        registered_types={d.get('equipment_type') for d in devices}
        for profile in CATALOG:
            if profile['id'] in registered_types:continue
            nodes.append({'id':'catalog:'+profile['id'],'label':profile['name'],'kind':'catalog','category':profile['category'],
                'group_id':'group:'+str(CATEGORIES.index(profile['category'])),'state':'CATALOG','equipment_type':profile['id'],
                'capabilities':profile['capabilities'],'modalities':profile['modalities'],'mode':'CATALOG_ONLY','limits':{}})
    # A conceptual grouping layout complements the operational dependency layout.
    categories=['Knowledge','Planning','Synthesis','Measurements','Performance','Preparation','Learning','Feedback','Devices']
    occupied=[]
    for index,category in enumerate(categories):
        members=[n for n in nodes if n['category']==category]
        angle=-math.pi/2+2*math.pi*index/len(categories)
        for j,node in enumerate(members):
            x=620+500*math.cos(angle);y=320+210*math.sin(angle)+(j-(len(members)-1)/2)*84
            while any(abs(x-px)<197 and abs(y-py)<83 for px,py in occupied):y+=84
            node['mind_x']=x;node['mind_y']=y;occupied.append((x,y))
    base_y=max((n.get('mind_y',0) for n in nodes),default=0)+125
    flow_y=max((n.get('y',0) for n in nodes if n['kind'] in {'step','device_step'}),default=-50)+170
    for index,category in enumerate(CATEGORIES):
        group_id='group:'+str(index);members=[n for n in nodes if n.get('group_id')==group_id]
        group={'id':group_id,'label':category,'kind':'device_group','category':category,'state':str(len(members))+' ITEMS',
               'members':[n['label'] for n in members],'registered_count':sum(n['kind']=='device' for n in members),
               'catalog_count':sum(p['category']==category for p in CATALOG),
               'default_expanded':bool(steps) and category in CATEGORIES[:2],
               'x':45+index*455,'y':flow_y,'mind_x':45+index*455,'mind_y':base_y}
        nodes.append(group)
        for j,node in enumerate(members):
            node.update(x=group['x']+(j%2)*210,y=flow_y+100+(j//2)*90,
                        mind_x=group['mind_x']+(j%2)*210,mind_y=base_y+100+(j//2)*90)
            edges.append({'source':group_id,'target':node['id'],'kind':'membership','label':'equipment collection'})
    instrument_count=sum(n['kind']=='device' for n in nodes)
    summary=(f'{len(CATALOG)} equipment types · {instrument_count} registered instruments' if include_catalog else f'{instrument_count} registered instruments') if not steps else None
    return {'scope':service.store.origin+':'+scope+(':catalog' if include_catalog else ''),'name':spec['name'],'state':state,'summary':summary,'nodes':nodes,'edges':edges,
            'schedule':schedule,'devices':devices,'completed':sum(s['state']=='COMPLETED' for s in steps),'total':len(steps),
            'jobs':[{'id':j['id'],'device_id':j['device_id'],'state':j['state']} for j in jobs],
            'source_spec':spec,'editable':bool(steps),'coordination':coordination(spec,steps,schedule)}
