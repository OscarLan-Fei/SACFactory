"""Persistent workflow library, run control and configurable device contracts."""
import json
import pandas as pd
import streamlit as st

from .domain import ELEMENTS
from .ui_elements import element_multiselect
from .workflow_service import WorkflowService
from .workflow_engine import OPERATIONS


def render_workflows(store,samples,context,methods=None,unified=False):
    from .ui_navigation import preserve_analysis_controls
    preserve_analysis_controls(('discovery_','workflow_'))
    from .ui_chat_workflows import apply_board_request
    try:apply_board_request(st.session_state,samples,store.origin)
    except ValueError as exc:
        st.session_state.pop('workflow_chat_board_request',None);st.warning(str(exc))
    st.title('SAC-Gen Discovery & Workflows' if unified else 'Devices & Workflows')
    st.caption('Design → Execute → Characterize → Learn. Templates, scientific tools, device scheduling and reviewed feedback share one workspace.')
    prefix='workflow_'+store.origin
    with st.expander('Source data selection',expanded=False):
        observed=[e for e in ELEMENTS if any(e in s['elements'] for s in samples)]
        if st.button('Use all source records and observed elements',key=prefix+'_all_data'):
            st.session_state[prefix+'_elements']=observed;st.session_state[prefix+'_all_sources']=True
        elements=element_multiselect('Workflow elements',ELEMENTS,default=observed or context.get('elements',[]),key=prefix+'_elements')
        eligible=[s['sample_id'] for s in samples if set(s['elements'])&set(elements)]
        use_all=st.checkbox('Use all matching source records',value=True,key=prefix+'_all_sources')
        if not use_all:
            if prefix+'_selection' in st.session_state:st.session_state[prefix+'_selection']=[i for i in st.session_state[prefix+'_selection'] if i in eligible]
            selected=st.multiselect('Source experiment records',eligible,default=eligible,key=prefix+'_selection')
        else:selected=eligible
    st.caption(f'{store.origin} · {len(selected)} selected source records · {len(elements)} selected elements')
    service=WorkflowService(store,context,selected)
    jump=st.session_state.get('workflow_board_tab_jump')
    if jump and jump['origin']==store.origin:
        st.session_state[prefix+'_tabs']=jump['tab'];del st.session_state['workflow_board_tab_jump']
    run_jump=st.session_state.get('workflow_board_run_jump')
    if run_jump and run_jump['origin']==store.origin:
        st.session_state[prefix+'_tabs']='Live board';st.session_state[prefix+'_graph_scope']='Run'
        st.session_state[prefix+'_board_run_selection']=run_jump['run_id'];del st.session_state['workflow_board_run_jump']
    board,library,science,runs,devices=st.tabs(['Live board','Workflow library','Scientific tools','Runs','Devices'],default='Workflow library' if unified else 'Live board',key=prefix+'_tabs',on_change='rerun')
    try:
        if board.open:
            with board:
                from .ui_workflow_board import render_dashboard
                available=sorted(service.db.all('runs'),key=lambda r:r['created'],reverse=True)
                definitions=sorted(service.db.all('definitions'),key=lambda d:d['created'],reverse=True)
                modes=['Run','Saved workflow','Device network','Equipment catalog'] if available else ['Saved workflow','Device network','Equipment catalog'] if definitions else ['Device network','Equipment catalog']
                if st.session_state.get('workflow_chat_preview_'+store.origin):modes.append('Chat preview')
                if st.session_state.get(prefix+'_graph_scope') not in modes:st.session_state[prefix+'_graph_scope']=modes[0]
                mode=st.radio('Graph scope',modes,horizontal=True,key=prefix+'_graph_scope')
                if mode=='Run':
                    lookup={r['id']:r for r in available}
                    ident=st.selectbox('Board run',list(lookup),format_func=lambda key:lookup[key]['spec']['name']+' · '+key[-6:],key=prefix+'_board_run_selection')
                    render_dashboard(service,prefix+'_board_'+ident,run_id=ident)
                elif mode=='Saved workflow':
                    lookup={d['id']:d for d in definitions}
                    ident=st.selectbox('Board workflow',list(lookup),format_func=lambda key:lookup[key]['spec']['name']+' · v'+str(lookup[key]['spec'].get('version',1))+' · '+key[-6:],key=prefix+'_board_definition')
                    if st.button('Start this workflow',key=prefix+'_board_start'):
                        result=service.start(ident)
                        st.session_state['workflow_board_run_jump']={'origin':store.origin,'run_id':result['id']};st.rerun()
                    render_dashboard(service,prefix+'_preview_'+ident,definition_id=ident)
                elif mode=='Chat preview':
                    from .ui_chat_workflows import render_board_chat_preview
                    render_board_chat_preview(store,prefix)
                else:render_dashboard(service,prefix+'_devices_board_'+mode.replace(' ','_'),include_catalog=mode=='Equipment catalog')
        if library.open:
            with library:
                from .ui_workflow_templates import render_templates
                render_templates(service,prefix)
                st.divider()
                st.write('**Reusable workflows**')
                if store.origin=='REAL':st.info('Use DEMO for simulator acquisition. REAL workflows can analyze selected records, train candidates and use reviewed characterization.')
                definitions=sorted(service.db.all('definitions'),key=lambda d:d['created'],reverse=True)
                if definitions:
                    lookup={d['id']:d for d in definitions}
                    ident=st.selectbox('Saved workflow',list(lookup),format_func=lambda key:lookup[key]['spec']['name']+' · '+key,key=prefix+'_definition')
                    definition=lookup[ident]
                    st.dataframe(pd.DataFrame([{'Step':n['id'],'Action':OPERATIONS[n['operation']][0],
                        'Depends on':', '.join(n.get('depends_on',[]))} for n in definition['spec']['steps']]),hide_index=True,width='stretch')
                    if st.button('Start saved workflow',key=prefix+'_start'):
                        result=service.start(ident)
                        st.session_state['workflow_board_run_jump']={'origin':store.origin,'run_id':result['id']};st.rerun()
                    st.download_button('Export workflow definition',json.dumps(definition['spec'],indent=2).encode(),'sac_workflow.json','application/json',key=prefix+'_export_definition')
                with st.expander('Create or revise a workflow definition'):
                    st.caption('Edits save a new immutable definition. References point to declared earlier dependencies. Executable code is not part of a definition.')
                    initial={'name':'Research memory and manual review','version':1,'steps':[
                        {'id':'memory','operation':'knowledge.search','parameters':{'query':'single atom synthesis'}},
                        {'id':'review','operation':'manual.checkpoint','parameters':{'instruction':'Review the retrieved sources and document the next action.'},'depends_on':['memory']}]}
                    raw=st.text_area('Workflow JSON',value=json.dumps(initial,indent=2),height=290,key=prefix+'_json')
                    if st.button('Save workflow definition',key=prefix+'_save'):
                        item=service.db.save(json.loads(raw));st.success('Saved '+item['id'])
                    st.dataframe(pd.DataFrame([{'Operation':name,'Required':', '.join(value['required']),'Optional':', '.join(value['optional'])}
                        for name,value in service.capabilities()['operations'].items()]),hide_index=True,width='stretch')
                st.write('**Conversation commands**')
                st.write('Research conversation can build and run these workflows through registered tools. Local commands work without an API:')
                st.text('/workflow templates\n/workflow template <template_id>\n/workflow capabilities\n/workflow start <workflow_id>\n/workflow run <run_id>')
        if science.open:
            with science:
                st.caption('Single-step scientific tools use the shared source selection. Their results and workflow runs use the same discovery ledger and model registry.')
                current=st.session_state.get('discovery_tabs_'+store.origin,'Evolution')
                mapping={'Evolution':'real_feedback_loop','Knowledge':'inverse_screening','Model disagreement':'data_benchmark','Set model & updates':'candidate_update',
                         'Inverse design':'inverse_screening','Characterization':'reviewed_characterization','Validation':'data_benchmark'}
                if current in mapping and st.button('Use this task in a workflow',key=prefix+'_science_template'):
                    st.session_state['workflow_template_jump']={'origin':store.origin,'template_id':mapping[current]}
                    st.session_state['workflow_board_tab_jump']={'origin':store.origin,'tab':'Workflow library'};st.rerun()
                from .ui_discovery import render_discovery
                selected_samples=[s for s in samples if s['sample_id'] in selected]
                render_discovery(store,selected_samples,methods or {},context,embedded=True)
        if runs.open:
            with runs:
                saved=sorted(service.db.all('runs'),key=lambda r:r['created'],reverse=True)
                if not saved:st.info('Start a saved workflow from Workflow library.');return
                ids=[r['id'] for r in saved];ident=st.selectbox('Workflow run',ids,key=prefix+'_selected_run')
                from .ui_workflow_board import render_dashboard
                render_dashboard(service,prefix+'_runs_'+ident,run_id=ident)
        if devices.open:
            with devices:
                from .ui_devices import render_devices
                render_devices(service,prefix)
    except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:st.error(str(exc))
