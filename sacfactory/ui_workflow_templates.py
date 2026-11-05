"""Starter workflows, parameter editing and a side-effect-free relationship preview."""
import json
import pandas as pd
import streamlit as st

from .workflow_templates import TEMPLATES,INDEX,catalog,compatible
from .workflow_engine import OPERATIONS,validate_spec
from .workflow_graph import snapshot
from .ui_elements import element_multiselect
from .domain import ELEMENTS

TOOL_LINKS={'simulation_loop':'Evolution','simulated_screening':'Characterization','inverse_screening':'Inverse design',
            'multielement_bridge':'Inverse design','data_benchmark':'Validation','candidate_update':'Set model & updates',
            'reviewed_characterization':'Characterization','characterization_panel':'Characterization','real_feedback_loop':'Evolution'}


def render_templates(service,prefix):
    from .discovery.workflows import available_models
    from .ui_workflow_board import _canvas
    st.write('**Workflow templates**')
    available=catalog(service.store.origin)
    jump=st.session_state.get('workflow_template_jump')
    if jump and jump['origin']==service.store.origin:
        st.session_state[prefix+'_template_category']='All templates'
        st.session_state[prefix+'_template_search']=''
        st.session_state[prefix+'_starter_selected']=jump['template_id']
        del st.session_state['workflow_template_jump']
    cols=st.columns([1,2])
    category=cols[0].selectbox('Template category',['All templates']+list(dict.fromkeys(t['category'] for t in available)),key=prefix+'_template_category')
    search=cols[1].text_input('Find a workflow template',key=prefix+'_template_search').strip().casefold()
    shown=[t for t in available if (category=='All templates' or t['category']==category) and (not search or search in (t['name']+' '+t['description']).casefold())]
    st.caption(f'{len(TEMPLATES)} built-in patterns · {len(available)} available in {service.store.origin}. Copies remain editable; saving a template never starts an experiment or trains a model.')
    if shown:
        choices=[t['id'] for t in shown];selection_key=prefix+'_starter_selected'
        if st.session_state.get(selection_key) not in choices:st.session_state[selection_key]=choices[0]
        ident=st.selectbox('Starter workflow',choices,format_func=lambda key:INDEX[key]['name'],key=selection_key)
        template=INDEX[ident];key=prefix+'_template_'+ident
        st.write(template['description'])
        options={'name':st.text_input('Workflow name',value=template['name'],key=key+'_name')}
        design=ident in {'simulation_loop','simulated_screening','inverse_screening','multielement_bridge','manual_synthesis','real_feedback_loop','confirmed_synthesis_loop'}
        with st.expander('Template parameters'):
            if design:
                elements=element_multiselect('Target elements',ELEMENTS,default=service.context.get('elements') or ['Fe'],key=key+'_elements')
                options['elements']=elements
                options['batch_size']=st.number_input('Experiments to propose',1,48,32 if ident=='simulation_loop' else 6,key=key+'_size')
                bounds=st.data_editor(pd.DataFrame([dict(element=e,lower_mmol=.2,upper_mmol=2.4) for e in elements],columns=['element','lower_mmol','upper_mmol']),
                    disabled=['element'],hide_index=True,key=key+'_bounds',column_config={'lower_mmol':st.column_config.NumberColumn(min_value=.000001),'upper_mmol':st.column_config.NumberColumn(min_value=.000001)})
                options['bounds_mmol']={r['element']:[float(r['lower_mmol']),float(r['upper_mmol'])] for r in bounds.to_dict('records')}
                cols=st.columns(2)
                options['temperature']=[cols[0].number_input('Lower controlled temperature / C',10,3990,800,step=10,key=key+'_tmin'),
                                        cols[1].number_input('Upper controlled temperature / C',10,3990,1600,step=10,key=key+'_tmax')]
                raw=st.text_input('Pulse counts',value='1,2,4,8,12',key=key+'_pulses')
                try:options['pulses']=[int(v.strip()) for v in raw.split(',')]
                except ValueError:options['pulses']=[]
                if ident not in {'simulation_loop','simulated_screening'}:
                    options['allow_extrapolation']=st.checkbox('Allow explicitly flagged proposals outside recorded training support',key=key+'_outside')
                st.caption(f'Feed quantities are mmol of metal precursor. Current preparation context: {service.context.get("pan_mass_g",2)} g PAN and {service.context.get("solvent_volume_mL",20)} mL DMF.')
            names=available_models(service.store.root,service.store.origin)
            preferred=next((n for n in ['SAC-Gen-GB','Pretrained SAC-Gen-GB','Fixed prior'] if n in names),names[0])
            if design or ident in {'data_benchmark','candidate_update'}:
                options['model']=st.selectbox('Workflow prediction model',names,index=names.index(preferred),key=key+'_model')
            if ident=='data_benchmark':options['comparison_models']=st.multiselect('Models to compare',names,default=names[:2],key=key+'_models')
            if ident=='confirmed_synthesis_loop':
                options['composition_mode']=st.selectbox('Precursor feed constraint',['independent','equimolar'],key=key+'_composition')
                protocol=st.text_area('Performance protocol for operator review',key=key+'_performance_protocol')
                if protocol.strip():options['performance_protocol']=protocol.strip()
            if ident in {'simulation_loop','candidate_update','data_benchmark','confirmed_synthesis_loop'}:
                options['epochs']=st.number_input('Training epochs',1,500,10 if ident=='simulation_loop' else 50,key=key+'_epochs')
            if ident in {'data_benchmark','candidate_update','confirmed_synthesis_loop'}:options['folds']=st.number_input('Protected recipe-group folds',2,10,5,key=key+'_folds')
            if ident in {'candidate_update','confirmed_synthesis_loop'}:options['candidate_family']=st.selectbox('Candidate family',['SAC-Gen-Set','SAC-Gen-GB','SAC-Gen-Diffusion','Calibrated discovery ensemble'],key=key+'_family')
            if template['needs_records']:
                options['sample_ids']=service.selected_ids
                st.caption(f'{len(service.selected_ids)} records from the shared source selection. Data eligibility is checked by the scientific operation when it runs.')
            if template['needs_artifacts']:
                samples={s['sample_id']:s for s in service.store.all('sample') if s['sample_id'] in service.selected_ids}
                from .discovery.common import identity
                keys={identity(s) for s in samples.values()}
                artifacts={a['id']:a for a in service.ledger.records('characterization') if a['record_key'] in keys}
                options['artifact_ids']=st.multiselect('Registered characterization artifacts',list(artifacts),format_func=lambda value:artifacts[value]['sample_id']+' · '+artifacts[value]['modality']+' · '+artifacts[value]['file_name'],key=key+'_artifacts')
            registered=service.db.all('devices');bindings={}
            if template['roles']:
                options['register_resources']=st.checkbox('Create missing planning resources or DEMO fixtures when saving',value=True,key=key+'_register')
                st.caption('Operator steps reserve their assigned equipment while waiting for acknowledgement. Inventory resources send no instrument commands.')
                for role in template['roles']:
                    devices=[d for d in registered if compatible(role,d,service.store.origin) and d['id']!='demo_characterizer']
                    lookup={d['id']:d for d in devices}
                    choice=st.selectbox(role['name']+' binding',['Automatic / planning entry']+list(lookup),
                        format_func=lambda value:lookup[value]['name']+' · '+value if value in lookup else value,key=key+'_binding_'+role['role'])
                    if choice in lookup:bindings[role['role']]=choice
                options['bindings']=bindings
        try:
            plan=service.template_plan(ident,options);spec=plan['spec']
            cols=st.columns(3)
            cols[0].metric('Workflow steps',len(spec['steps']))
            cols[1].metric('Operator checkpoints',sum(n['operation']=='manual.checkpoint' for n in spec['steps']))
            cols[2].metric('Equipment roles',len(template['roles']))
            st.dataframe(pd.DataFrame([{'Step':n['id'],'Action':OPERATIONS[n['operation']][0],'Instrument':n['parameters'].get('device_id',''),
                  'Depends on':', '.join(n.get('depends_on',[])),'Operator instruction':n['parameters'].get('instruction','')} for n in spec['steps']]),hide_index=True,width='stretch',height=min(420,75+35*len(spec['steps'])))
            with st.expander('Workflow relationship preview'):
                view=snapshot(service,draft_spec=spec,planned_devices=plan['resources_to_register'])
                graph={k:v for k,v in view.items() if k in {'scope','name','state','summary','nodes','edges','completed','total'}}
                graph['editable']=False
                event=_canvas(graph=graph,key=key+'_preview_canvas',default=None)
                if event and event.get('action')=='open_devices' and event.get('event_id')!=st.session_state.get(key+'_event'):
                    st.session_state[key+'_event']=event['event_id'];st.session_state['workflow_board_tab_jump']={'origin':service.store.origin,'tab':'Devices'};st.rerun()
                st.caption('Preview only. Configure roles above, save a copy, then use Live board to edit bindings or start a run.')
            cols=st.columns(3)
            if cols[0].button('Create workflow from template',key=key+'_create',type='primary'):
                created=service.instantiate_template(ident,options)
                st.session_state[prefix+'_definition']=created['workflow_id']
                st.session_state[prefix+'_board_definition']=created['workflow_id']
                st.session_state[prefix+'_graph_scope']='Saved workflow'
                st.session_state['workflow_board_tab_jump']={'origin':service.store.origin,'tab':'Live board'}
                st.rerun()
            cols[1].download_button('Export configured workflow',json.dumps(spec,indent=2).encode(),'sac_workflow_'+ident+'.json','application/json',key=key+'_export')
            if ident in TOOL_LINKS and cols[2].button('Open single-step tool',key=key+'_tool'):
                st.session_state['discovery_tabs_'+service.store.origin]=TOOL_LINKS[ident]
                st.session_state['workflow_board_tab_jump']={'origin':service.store.origin,'tab':'Scientific tools'};st.rerun()
        except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:st.info(str(exc))
    else:st.info('No matching starter workflow. Change the category or search.')
    with st.expander('Import an existing workflow'):
        uploaded=st.file_uploader('Workflow JSON',type=['json'],key=prefix+'_import_file')
        if uploaded is not None:
            try:
                if uploaded.size>120000:raise ValueError('Use a workflow definition smaller than 120 KB.')
                content=json.loads(uploaded.getvalue().decode('utf-8-sig'))
                spec=validate_spec(content.get('spec',content) if isinstance(content,dict) else content)
                st.write(spec['name']+' · '+str(len(spec['steps']))+' steps')
                if st.button('Save imported workflow',key=prefix+'_import_save'):
                    saved=service.db.save(spec);st.session_state[prefix+'_definition']=saved['id'];st.success('Imported '+saved['spec']['name']+'. No run was started.')
            except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
