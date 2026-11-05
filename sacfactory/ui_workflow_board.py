"""Live relationship canvas, resource-aware queue and non-blocking run controls."""
from pathlib import Path
import json
import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from .workflow_graph import snapshot
from .workflow_schedule import bind_devices,device_requirement
from .workflow_worker import launch,worker_status
from .device_catalog import modalities_of

_canvas=components.declare_component('sac_workflow_canvas',path=str(Path(__file__).resolve().parents[1]/'resources/workflow_canvas'))


def _queue(service,view,include_all=False):
    rows=[]
    if include_all:
        for run in service.db.all('runs'):
            if run['state'] not in {'READY','RUNNING','QUEUED','WAITING','PAUSED','FAILED'}:continue
            for row in snapshot(service,run_id=run['id'])['schedule']:
                if row['state']=='COMPLETED':continue
                rows.append({**row,'run_id':run['id'],'workflow':run['spec']['name']})
    else:rows=[{**r,'run_id':None,'workflow':view['name']} for r in view['schedule']]
    order={'RUNNING':0,'READY':1,'DEVICE_BUSY':2,'WAITING_DEPENDENCIES':3,'WAITING':4,'BLOCKED':5,'PAUSED':6,'FAILED':7,'CANCELLED':8,'COMPLETED':9}
    return sorted(rows,key=lambda r:(order.get(r['state'],10),-r['priority'],-r['critical_path_seconds'],r['step_id']))


def _run_buttons(service,run_id,prefix):
    run=service.db.get('runs',run_id);worker=worker_status(service);state=run['state']
    cols=st.columns(4)
    actions=[('Execute next step','step',state not in {'READY','RUNNING','QUEUED'} or worker['active']),
             ('Run remaining steps','run',state not in {'READY','RUNNING','QUEUED'} or worker['active']),
             ('Resume','resume',worker['active']) if state=='PAUSED' else ('Pause','pause',state not in {'READY','RUNNING','QUEUED','WAITING'})]
    for i,(label,action,disabled) in enumerate(actions):
        if cols[i%4].button(label,key=prefix+'_control_'+action,disabled=disabled):
            if action in {'step','run'}:launch(service,run_id,1 if action=='step' else 48)
            else:
                service.db.control(run_id,action)
                if action=='recover':service.devices.reconcile_expired(run_id)
    with cols[3].popover('More run controls'):
        for label,action,disabled in [('Retry failed step','retry',state!='FAILED' or worker['active']),
            ('Cancel remaining','cancel',state in {'COMPLETED','CANCELLED'}),
            ('Reconcile expired worker','recover',state!='RUNNING' or worker['active'])]:
            if st.button(label,key=prefix+'_control_'+action,disabled=disabled):
                service.db.control(run_id,action)
                if action=='recover':service.devices.reconcile_expired(run_id)
    if worker['active']:st.caption('Background worker: '+worker['run_id']+'. Status refreshes every two seconds; pause stops subsequent steps.')
    if worker.get('error'):st.error(worker['error'])


@st.fragment(run_every=2)
def render_dashboard(service,prefix,run_id=None,definition_id=None,include_catalog=False):
    """Refresh only this panel; observing it never starts a job implicitly."""
    try:
        if run_id:_run_buttons(service,run_id,prefix)
        draft_key=prefix+'_draft_bindings';bindings=st.session_state.get(draft_key,{})
        view=snapshot(service,run_id,definition_id,bindings,include_catalog=include_catalog)
        if run_id:
            run=service.db.get('runs',run_id)
            if run.get('error'):st.error(run['error'])
        definition=service.db.get('definitions',run['workflow_id'] if run_id else definition_id) if run_id or definition_id else None
        if definition and st.button('Continue in conversation',key=prefix+'_conversation'):
            st.session_state['workflow_conversation_request']={'origin':service.store.origin,'workflow_id':definition['id'],'run_id':run_id}
            st.session_state['page_jump']='Experiment chat';st.rerun()
        basis=run['context'] if run_id else definition.get('template',{}).get('options',{}) if definition else {}
        if all(type(basis.get(k)) in (int,float) for k in ['pan_mass_g','solvent_volume_mL','voltage_V','current_A']):
            st.caption(f"Preparation: {basis['pan_mass_g']:g} g PAN · {basis['solvent_volume_mL']:g} mL DMF · {basis['voltage_V']:g} V · {basis['current_A']:g} A")
        data={k:v for k,v in view.items() if k in {'scope','name','state','summary','nodes','edges','completed','total','editable'}}
        event=_canvas(graph=data,key=prefix+'_canvas',default=None)
        if event and event.get('event_id')!=st.session_state.get(prefix+'_processed_event'):
            st.session_state[prefix+'_processed_event']=event['event_id']
            if event.get('action') in {'open_devices','register_profile'}:
                if event.get('action')=='register_profile':
                    from .device_catalog import PROFILES
                    if event.get('profile_id') not in PROFILES:raise ValueError('Unknown equipment profile.')
                    st.session_state['workflow_device_profile']={'origin':service.store.origin,'profile_id':event['profile_id']}
                st.session_state['workflow_board_tab_jump']={'origin':service.store.origin,'tab':'Devices'};st.rerun()
            elif event.get('action')=='bind_device':
                proposed={**bindings,event.get('step_id'):event.get('device_id')}
                bind_devices(view['source_spec'],proposed,view['devices'])
                st.session_state[draft_key]=proposed
                st.success('Device binding staged. Save a revised workflow below to use it.')
        if not view['nodes']:st.info('Register a device or save a workflow to populate the graph.')
        if run_id:
            from .ui_workflow_cards import render_task_cards
            render_task_cards(service,run_id,prefix,controls=True)
        from .ui_workflow_coordination import render_coordination
        render_coordination(service,view,prefix,run_id)
        st.write('**Scheduling queue**')
        all_runs=st.checkbox('Show tasks from all active runs',key=prefix+'_all_runs') if run_id else False
        queue=_queue(service,view,all_runs)
        if all_runs:
            ready=next((r for r in queue if r['state']=='READY'),None)
            if st.button('Run next scheduled task',disabled=ready is None or worker_status(service)['active'],key=prefix+'_dispatch_next'):
                launch(service,ready['run_id'],1)
        if queue:
            table=pd.DataFrame([{'Order':i+1,'Task':r['action'],'Step':r['step_id'],'Status':r['state'],
                'Device':r['device_id'] or 'Local worker','Priority':r['priority'],
                'Estimated min':round(r['estimated_seconds']/60,2),
                'Recorded / elapsed s':None if r['actual_seconds'] is None else round(r['actual_seconds'],1),
                'Reason':r['reason'],**({'Workflow':r['workflow']} if all_runs else {})} for i,r in enumerate(queue)])
            st.dataframe(table,hide_index=True,width='stretch')
        else:st.info('No tasks in this view.')
        timeline=[{'Task':r['step_id'],'Start':r['estimated_start_seconds']/60,'Finish':r['estimated_end_seconds']/60,'State':r['state']}
            for r in view['schedule'] if r['estimated_start_seconds'] is not None and r['estimated_end_seconds']>r['estimated_start_seconds']]
        if timeline:
            with st.expander('Estimated execution timeline',expanded=False):
                chart=alt.Chart(pd.DataFrame(timeline)).mark_bar(cornerRadius=3).encode(
                    x=alt.X('Start:Q',title='Estimated minutes from plan start'),x2='Finish:Q',
                    y=alt.Y('Task:N',sort=[r['Task'] for r in timeline],title=None),color='State:N',tooltip=['Task','Start','Finish','State']).properties(height=max(160,len(timeline)*25))
                st.altair_chart(chart,width='stretch')
        st.caption('Queue order uses ready dependencies, priority and remaining critical path. One worker advances this data space at a time. Durations are editable planning estimates; elapsed values come from recorded execution.')
        if view['source_spec']['steps']:
            with st.expander('Bind devices and configure scheduling for a new workflow',expanded=bool(bindings)):
                tasks=[n for n in view['source_spec']['steps'] if n['operation'].startswith('device.') or n['operation']=='manual.checkpoint' and n['parameters'].get('capability')]
                if tasks:
                    step_id=st.selectbox('Device task',[n['id'] for n in tasks],key=prefix+'_bind_step')
                    node=next(n for n in tasks if n['id']==step_id);_,capability=device_requirement(node)
                    if node['operation']=='manual.checkpoint':capability=node['parameters'].get('capability')
                    required=node.get('parameters',{}).get('modalities',[])
                    if not isinstance(required,list):required=[]
                    choices=[d['id'] for d in view['devices'] if capability in d['capabilities'] and all(m in modalities_of(d) for m in required)]
                    if choices:
                        device_id=st.selectbox('Compatible registered device',choices,key=prefix+'_bind_device')
                        if st.button('Stage device binding',key=prefix+'_stage_binding'):
                            bindings={**st.session_state.get(draft_key,{}),step_id:device_id}
                            st.session_state[draft_key]=bindings;st.success('Binding staged for a revised workflow.')
                if st.session_state.get(draft_key):st.write('Draft bindings: '+', '.join(k+' → '+v for k,v in st.session_state[draft_key].items()))
                settings=pd.DataFrame([{'Step':r['step_id'],'Priority':r['priority'],'Estimated seconds':r['estimated_seconds']} for r in view['schedule']])
                edited=st.data_editor(settings,disabled=['Step'],hide_index=True,key=prefix+'_schedule_editor',
                    column_config={'Priority':st.column_config.NumberColumn(min_value=0,max_value=100,step=1),
                                   'Estimated seconds':st.column_config.NumberColumn(min_value=1,max_value=86400,step=1)})
                if st.button('Save as a revised workflow',key=prefix+'_save_revision'):
                    values={r['Step']:{'priority':int(r['Priority']),'estimated_seconds':float(r['Estimated seconds'])} for r in edited.to_dict('records')}
                    result=service.revise_board_workflow(run['workflow_id'] if run_id else definition_id,st.session_state.get(draft_key,{}),values)
                    st.session_state[draft_key]={}
                    st.session_state['workflow_chat_board_request']={'origin':service.store.origin,'kind':'definition',
                        'selected_ids':service.selected_ids,'context':service.context,'workflow_id':result['id'],'run_id':None}
                    st.rerun()
                st.caption('Revisions create a new definition. Active runs keep their original device bindings and scientific inputs.')
        if run_id:
            for step in service.db.steps(run_id):
                if step['state']=='WAITING':
                    if step['output'].get('return_fields') or step['output'].get('approval_required'):continue
                    st.info(step['output']['instruction'])
                    if step['output'].get('device_id'):st.caption('Reserved instrument: '+step['output']['device_id'])
                    if step['output'].get('sample_ids'):st.caption('Specimens: '+', '.join(step['output']['sample_ids']))
                    note=st.text_input('Manual completion note',key=prefix+'_manual_'+step['id'])
                    if st.button('Acknowledge manual action',key=prefix+'_ack_'+step['id']):
                        service.acknowledge(run_id,step['id'],note);st.rerun(scope='fragment')
            with st.expander('Export run records'):
                st.download_button('Download complete run record',json.dumps({'run':service.db.get('runs',run_id),'steps':service.db.steps(run_id)},indent=2).encode(),
                    'workflow_run.json','application/json',key=prefix+'_export_run')
    except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:st.error(str(exc))
