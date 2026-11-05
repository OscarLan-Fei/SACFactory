"""Current conversational workflow, live status and scoped board navigation."""
import json
import pandas as pd
import streamlit as st

from .workflow_chat import WorkflowChat
from .workflow_service import WorkflowService
from .workflow_graph import snapshot
from .workflow_templates import PREPARATION_DEFAULTS


def workflow_message(result):
    if result.get('error'):return 'Workflow request could not be applied: '+result['error']
    if 'templates' in result:return f"{len(result['templates'])} workflow templates are available. Use /workflow preview <template_id>, then /workflow edit {{options}} and /workflow create."
    if 'profiles' in result:return f"Found {len(result['profiles'])} equipment profiles. Registration creates a contract; it does not connect an instrument."
    if 'devices' in result:return f"{len(result['devices'])} instruments are registered in this data space."
    if result.get('saved') is False:return 'Workflow preview updated: '+result['name']+'. No run was started. Open Live board from the current-workflow card to inspect it.'
    if result.get('saved'):return 'Saved '+result['name']+' as '+result['workflow_id']+'. Existing runs retain their original parameters.'
    if result.get('device'):return 'Registered '+result['device']['name']+' ('+result['device']['id']+').'
    if result.get('run_id'):
        summary=result.get('status',result)
        return 'Run '+result['run_id']+' · '+summary.get('state','background execution requested')+'. The current-workflow card shows live recorded progress.'
    if result.get('workflow_id'):return 'Selected workflow '+result['workflow_id']+'. Use /workflow start to create its run, then /workflow run to execute ready steps.'
    return 'Workflow service result is available. Inspect the current-workflow card or use /workflow templates.'


def render_workflow_result(result):
    for key in ['templates','devices','profiles']:
        if key not in result:continue
        rows=result[key]
        if rows:
            fields=['id','name','category','mode','busy'] if key=='devices' else ['id','name','category']
            st.dataframe(pd.DataFrame([{k:r.get(k) for k in fields if k in r} for r in rows]),hide_index=True,width='stretch')


def queue_board(session,state,kind=None):
    kind=kind or ('preview' if state.get('preview') else 'run' if state.get('run_id') else 'definition')
    origin=state['origin']
    request={'origin':origin,'kind':kind,'selected_ids':list(state.get('selected_ids',[])),
             'context':state.get('context',{}),'workflow_id':state.get('workflow_id'),'run_id':state.get('run_id')}
    if kind=='preview':
        if not state.get('preview'):raise ValueError('No conversation preview is available.')
        request['preview']=state['preview']
    session['workflow_chat_board_request']=request
    session['page_jump']='Discovery & Workflows'


def apply_board_request(session,samples,origin):
    request=session.get('workflow_chat_board_request')
    if not request or request['origin']!=origin:return
    from .domain import ELEMENTS
    selected=set(request['selected_ids']);known={s['sample_id'] for s in samples}
    if selected-known:raise ValueError('Conversation source records changed. Refresh the preview before opening its board.')
    prefix='workflow_'+origin
    elements=[e for e in ELEMENTS if any(s['sample_id'] in selected and e in s['elements'] for s in samples)] or request['context'].get('elements',[])
    eligible={s['sample_id'] for s in samples if set(s['elements'])&set(elements)}
    session[prefix+'_elements']=elements;session[prefix+'_all_sources']=selected==eligible
    session[prefix+'_selection']=request['selected_ids'];session[prefix+'_tabs']='Live board'
    if request['kind']=='preview':
        session['workflow_chat_preview_'+origin]=request['preview'];session[prefix+'_graph_scope']='Chat preview'
    else:
        session.pop('workflow_chat_preview_'+origin,None)
        session[prefix+'_graph_scope']='Run' if request['kind']=='run' else 'Saved workflow'
        session[prefix+('_board_run_selection' if request['kind']=='run' else '_board_definition')]=request['run_id'] if request['kind']=='run' else request['workflow_id']
    session.pop('workflow_chat_board_request',None)


def apply_conversation_request(session,service,state):
    """Select a board item without widening the conversation's data permission."""
    request=session.get('workflow_conversation_request')
    if not request or request['origin']!=service.store.origin:return False
    session.pop('workflow_conversation_request',None)
    controller=WorkflowChat(service,state)
    if request.get('run_id'):controller.control('status',run_id=request['run_id'])
    else:controller.template('inspect',workflow_id=request['workflow_id'])
    state.pop('preview',None)
    return True


@st.fragment(run_every=2)
def render_current_workflow(store,state):
    if not state.get('preview') and not state.get('workflow_id') and not state.get('run_id'):return
    try:
        service=WorkflowService(store,state.get('context'),state.get('selected_ids',[]))
        prefix='chat_flow_'+store.origin;draft=state.get('preview');run=None
        st.write('**Current workflow**')
        if draft:
            plan=service.template_plan(draft['template_id'],draft['options']);name=plan['spec']['name'];options=plan['options'];label='Unsaved preview'
            if state.get('run_id'):
                previous=service.db.get('runs',state['run_id'])
                if previous:st.caption('Existing run '+previous['id']+' remains '+previous['state']+'. Preview edits do not change it.')
        elif state.get('run_id'):
            service.authorize_run(state['run_id']);run=service.db.get('runs',state['run_id'])
            name=run['spec']['name'];options=service.db.get('definitions',run['workflow_id']).get('template',{}).get('options',{});label=run['state']
        else:
            row=service.inspect_workflow(state['workflow_id']);name=row['spec']['name'];options=row.get('template',{}).get('options',{});label='Saved · not started'
        st.markdown('**'+name+'**');st.caption(label)
        if run:
            summary=service.db.summary(run['id'])
            st.progress(summary['completed_steps']/max(1,summary['total_steps']),text=f"{summary['completed_steps']}/{summary['total_steps']} steps · {summary['state']}")
            if summary.get('error'):st.error(service.public_result(summary['error']))
            waiting=[s for s in service.db.steps(run['id']) if s['state']=='WAITING']
            for step in waiting:st.info('Waiting for operator: '+step['id']+' · '+step['output']['instruction'])
            from .ui_workflow_cards import render_task_cards
            render_task_cards(service,run['id'],prefix,controls=True)
        cols=st.columns(2)
        if cols[0].button('Open Live board',key=prefix+'_open'):
            queue_board(st.session_state,state);st.rerun()
        if draft and cols[1].button('Save preview as workflow',key=prefix+'_save'):
            WorkflowChat(service,state).template('create');st.rerun(scope='fragment')
        with st.expander('Current workflow parameters'):
            rows=[{'Parameter':k,'Value':json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else str(v)} for k,v in options.items() if k not in {'sample_ids','artifact_ids'}]
            if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            st.caption(f"{len(state.get('selected_ids',[]))} conversation source records. Editing creates a new definition; running specimens retain their frozen controls.")
    except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:st.warning(str(exc))


def render_board_chat_preview(store,prefix):
    from .ui_workflow_board import _canvas
    draft=st.session_state.get('workflow_chat_preview_'+store.origin)
    if not draft:return
    service=WorkflowService(store,draft['context'],draft['selected_ids'])
    plan=service.template_plan(draft['template_id'],draft['options'])
    st.caption(f"Conversation preview · {len(draft['selected_ids'])} selected source records · No run started")
    o=plan['options']
    st.caption(f"Preparation: {o['pan_mass_g']:g} g PAN · {o['solvent_volume_mL']:g} mL DMF · {o['voltage_V']:g} V · {o['current_A']:g} A")
    view=snapshot(service,draft_spec=plan['spec'],planned_devices=plan['resources_to_register'])
    graph={k:v for k,v in view.items() if k in {'scope','name','state','summary','nodes','edges','completed','total'}};graph['editable']=False
    _canvas(graph=graph,key=prefix+'_chat_preview_canvas',default=None)
    from .ui_workflow_coordination import render_coordination
    render_coordination(service,view,prefix+'_chat_preview')
    st.dataframe(pd.DataFrame([{'Step':n['id'],'Operation':n['operation'],'Instrument':n['parameters'].get('device_id',''),
                              'Depends on':', '.join(n.get('depends_on',[]))} for n in plan['spec']['steps']]),hide_index=True,width='stretch')
    st.caption('Continue editing through Experiment chat, or save this preview to create a reusable workflow.')
    if st.button('Save conversation preview',key=prefix+'_chat_preview_save'):
        state=st.session_state.get('chat_workflow_state_'+store.origin,{})
        if state.get('preview')!=draft:raise ValueError('The conversation preview changed. Open the latest preview from chat.')
        created=WorkflowChat(service,state).template('create')
        st.session_state['workflow_chat_board_request']={'origin':store.origin,'kind':'definition','selected_ids':draft['selected_ids'],
             'context':draft['context'],'workflow_id':created['workflow_id'],'run_id':None}
        st.rerun()
