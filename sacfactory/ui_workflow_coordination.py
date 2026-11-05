"""Specialist responsibilities, data handoffs and scientific result navigation."""
import pandas as pd
import streamlit as st


def science_target(service,tab,run_id=None):
    origin=service.store.origin;target={'origin':origin,'tab':tab,'selected_ids':list(service.selected_ids),'run_id':run_id}
    if run_id:
        run=service.db.get('runs',run_id)
        if run is None:raise ValueError('Unknown source workflow run.')
        produced=[ident for step in service.db.steps(run_id) if step['operation']=='campaign.plan' and step['state']=='COMPLETED'
                  for ident in step['output'].get('sample_ids',[])]
        known={s['sample_id'] for s in service.store.all('sample')}
        target['selected_ids']=list(dict.fromkeys(i for i in run['selected_ids']+produced if i in known))
        target.update(workflow_id=run['workflow_id'],context=run['context'])
        reports=[r for r in service.ledger.records('validation') if r.get('workflow_run_id')==run_id]
        if tab=='Validation' and reports:target['report_id']=reports[-1]['id']
    return target


def open_science(service,tab,run_id=None):
    target=science_target(service,tab,run_id);origin=target['origin']
    if run_id:
        st.session_state['workflow_chat_board_request']={**target,'kind':'run'}
    if target.get('report_id'):st.session_state['discovery_saved_evaluation_'+origin]=target['report_id']
    st.session_state['discovery_tabs_'+origin]=tab
    st.session_state['workflow_board_tab_jump']={'origin':origin,'tab':'Scientific tools'}
    st.rerun()


def render_coordination(service,view,prefix,run_id=None):
    team=view['coordination']
    if not team['tasks']:return
    with st.expander('Specialist coordination & results',expanded=view['state']=='COMPLETED'):
        st.caption('Knowledge → Design → Execute → Characterize → Learn → Redesign. Registered tools perform each role; all status comes from saved task records.')
        cols=st.columns(3)
        cols[0].metric('Specialist roles',len(team['specialists']))
        cols[1].metric('Cross-role handoffs',sum(h['cross_specialist'] for h in team['handoffs']))
        cols[2].metric('Recorded results',team['recorded_results'])
        roles,handoffs,results=st.tabs(['Specialist team','Data handoffs','Recorded results'])
        with roles:
            st.dataframe(pd.DataFrame([{'Specialist':r['name'],'Status':r['state'],'Completed':f"{r['completed']}/{r['tasks']}",
                'Responsibility':r['responsibility']} for r in team['specialists']]),hide_index=True,width='stretch')
        with handoffs:
            if team['handoffs']:
                st.dataframe(pd.DataFrame([{'From':h['source_step'],'To':h['target_step'],'Sender':h['from_specialist'],
                    'Receiver':h['to_specialist'],'Input contract':h['contract'],'Availability':h['state']} for h in team['handoffs']]),hide_index=True,width='stretch')
            st.caption('References resolve only after declared dependencies complete. An available input does not mean the downstream task has executed.')
        with results:
            st.dataframe(pd.DataFrame([{'Task':t['step_id'],'Specialist':t['specialist'],'Status':t['state'],
                                       'Result':t['summary']} for t in team['tasks']]),hide_index=True,width='stretch')
            if run_id:
                steps=service.db.steps(run_id)
                designs=[s for s in steps if s['operation']=='inverse.design' and s['state']=='COMPLETED']
                if designs:
                    chosen=st.selectbox('Recorded recipe proposal',[s['id'] for s in designs],index=len(designs)-1,key=prefix+'_result_design')
                    output=next(s['output'] for s in designs if s['id']==chosen)
                    from .ui_discovery import preview_table
                    table=preview_table(output['design']['rows'])
                    st.dataframe(table,hide_index=True,width='stretch')
                    st.download_button('Export recorded proposals',table.to_csv(index=False).encode(),'workflow_proposals.csv','text/csv',key=prefix+'_result_csv')
                for step in steps:
                    if step['operation']=='model.train' and step['state']=='COMPLETED':
                        st.success('Candidate registered: '+step['output']['candidate']['model_id'])
        cols=st.columns(4)
        for col,(label,tab) in zip(cols,[('Inspect evidence','Characterization'),('Inspect validation','Validation'),
                                       ('Inspect models','Set model & updates'),('Inspect feedback','Evolution')]):
            if col.button(label,key=prefix+'_science_'+tab):open_science(service,tab,run_id)
        st.caption('Scientific tools share the selected source records and ledger. These links inspect results; they do not rerun scientific calculations.')
