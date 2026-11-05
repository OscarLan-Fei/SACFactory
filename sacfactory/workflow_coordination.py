"""Tool-bounded specialist roles and auditable handoffs for scientific workflows.

Roles route registered local operations; they are not independent LLM sessions.
Observing this projection never dispatches a task or changes a scientific record.
"""
from .workflow_engine import references


SPECIALISTS={
    'knowledge':{'name':'Knowledge specialist','module':'Knowledge','responsibility':'Retrieve traceable chemical and synthesis evidence.'},
    'design':{'name':'Design specialist','module':'Inverse design','responsibility':'Propose recipes and freeze their numerical forecasts.'},
    'execution':{'name':'Execution specialist','module':'Devices','responsibility':'Dispatch compatible instruments or wait for documented operator actions.'},
    'characterization':{'name':'Characterization specialist','module':'Characterization','responsibility':'Join registered measurements and preserve evidence-review gates.'},
    'learning':{'name':'Learning specialist','module':'Validation','responsibility':'Validate, train versioned candidates and compare model outputs.'},
    'feedback':{'name':'Feedback specialist','module':'Evolution','responsibility':'Match frozen forecasts to reviewed outcomes and register contrasts.'},
}


def specialist_for(operation):
    if operation=='knowledge.search':return 'knowledge'
    if operation in {'inverse.design','campaign.plan'}:return 'design'
    if operation.startswith('device.') or operation=='manual.checkpoint':return 'execution'
    if operation.startswith('characterization.'):return 'characterization'
    if operation.startswith(('model.','validation.','records.')):return 'learning'
    if operation in {'campaign.evaluate','hypothesis.register'}:return 'feedback'
    raise ValueError('No registered specialist for '+str(operation))


def output_summary(operation,output):
    if not isinstance(output,dict):return 'No recorded result'
    if output.get('awaiting_manual_action'):return 'Operator action requested; measurements remain separate'
    if operation=='knowledge.search':return f"{len(output.get('evidence',[]))} retrieved evidence entries"
    if operation=='inverse.design':return f"{output.get('proposed_experiments',0)} proposed recipes · {output.get('design',{}).get('model_name','registered model')}"
    if operation=='campaign.plan':return f"{len(output.get('sample_ids',[]))} Pending specimens · forecasts frozen"
    if operation.startswith('device.'):
        receipt=output.get('receipt',output.get('acquisition',{}))
        return f"{len(receipt.get('sample_ids',receipt.get('measurements',[])))} specimens · recorded instrument receipt"
    if operation=='characterization.analyze':return f"{len(output.get('evidence',{}).get('artifact_ids',[]))} registered raw exports linked"
    if operation in {'characterization.fixture_review','characterization.commit'}:return f"{len(output.get('assessment_ids',[]))} reviewed assessments"
    if operation=='campaign.evaluate':return f"{len(output.get('evaluation',[]))} frozen-forecast evaluations"
    if operation=='validation.run':return f"{len(output.get('folds',[]))} protected validation folds"
    if operation=='model.train':return f"{output.get('training_specimens',0)} training specimens · {output.get('family','candidate')}"
    if operation=='model.compare':return 'Registered model comparison recorded'
    if operation=='hypothesis.register':return 'One-factor contrast registered'
    return 'Recorded structured result'


def coordination(spec,steps,schedule=None):
    states={s['id']:s for s in steps};scheduled={r['step_id']:r for r in schedule or []}
    owners={n['id']:specialist_for(n['operation']) for n in spec['steps']}
    tasks=[];handoffs=[]
    for node in spec['steps']:
        ident=node['id'];step=states.get(ident,{})
        state=scheduled.get(ident,{}).get('state',step.get('state','PENDING'))
        output=step.get('output');owner=owners[ident]
        summary=('Operator completion recorded; measurements remain separate' if node['operation']=='manual.checkpoint' and state=='COMPLETED'
                 else output_summary(node['operation'],output) if output is not None else 'Not executed')
        tasks.append({'step_id':ident,'specialist_id':owner,'specialist':SPECIALISTS[owner]['name'],
                      'operation':node['operation'],'module':SPECIALISTS[owner]['module'],'state':state,
                      'summary':summary,
                      'result_recorded':output is not None,'output_fields':sorted(output) if isinstance(output,dict) else []})
        paths=list(references(node['parameters']))
        for dep in node.get('depends_on',[]):
            upstream=states.get(dep,{})
            available=upstream.get('state')=='COMPLETED'
            fields=[p.split('.',1)[1] for p in paths if p.split('.',1)[0]==dep]
            handoffs.append({'source_step':dep,'target_step':ident,'from_specialist':SPECIALISTS[owners[dep]]['name'],
                'to_specialist':SPECIALISTS[owner]['name'],'cross_specialist':owners[dep]!=owner,
                'contract':', '.join(fields) if fields else 'Completion dependency',
                'state':'AVAILABLE' if available else 'WAITING','consumer_state':state})
    team=[]
    for ident,role in SPECIALISTS.items():
        assigned=[t for t in tasks if t['specialist_id']==ident]
        if not assigned:continue
        labels={t['state'] for t in assigned}
        status=next((s for s in ['RUNNING','FAILED','WAITING','DEVICE_BUSY','READY','BLOCKED','PAUSED'] if s in labels),
                    'COMPLETED' if labels=={'COMPLETED'} else 'CANCELLED' if labels=={'CANCELLED'} else 'PENDING')
        team.append({'id':ident,**role,'state':status,'tasks':len(assigned),
                     'completed':sum(t['state']=='COMPLETED' for t in assigned)})
    return {'mode':'Registered tool specialists; one bounded local worker per data space',
            'specialists':team,'tasks':tasks,'handoffs':handoffs,
            'recorded_results':sum(t['result_recorded'] for t in tasks)}
