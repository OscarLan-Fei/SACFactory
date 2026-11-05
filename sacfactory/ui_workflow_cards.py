"""Inspectable task cards shared by conversation and the live workflow board."""
import json
import pandas as pd
import streamlit as st
from .workflow_returns import result_summary,pipeline_for,model_names


def readable(value,path=''):
    if isinstance(value,dict):return [row for k,v in value.items() if not str(k).startswith('_') for row in readable(v,(path+' / '+str(k)).strip(' /'))]
    if isinstance(value,list):return [row for i,v in enumerate(value[:48]) for row in readable(v,path+' / '+str(i+1))]
    return [{'Field':path,'Value':'—' if value is None else str(value)}]


def show_values(value):
    rows=readable(value)
    if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')


def recipe_table(design):
    rows=[]
    for index,row in enumerate(design.get('rows',[])):
        r=row['recipe'];volume=r.get('solvent_volume_mL',20.)
        rows.append({'Recipe':index+1,'Elements':'–'.join(r['elements']),
            'Precursor / mmol':'; '.join(f'{e}: {v*volume/1000:.4g}' for e,v in r['concentrations_mM'].items()),
            'Precursor / mM':'; '.join(f'{e}: {v:.4g}' for e,v in r['concentrations_mM'].items()),
            'Cutoff / °C':r['pulse_limit_C'],'Pulses':r['pulse_count'],
            'Predicted total / wt%':row.get('predicted_total_wt_percent'),'P(structural PASS)':row.get('p_xrd_pass')})
    return rows


def render_task_cards(service,run_id,prefix,controls=False):
    steps=service.db.steps(run_id);run=service.db.get('runs',run_id)
    available=[s for s in steps if s.get('output')]
    if not available:return
    waiting=next((s for s in available if s['state']=='WAITING'),None)
    default=(waiting or available[-1])['id']
    ids=[s['id'] for s in steps]
    follow=st.checkbox('Follow active task',value=True,key=prefix+'_follow_task')
    if follow:st.session_state[prefix+'_work_card']=default
    selected=st.selectbox('Workflow work card',ids,index=ids.index(default),key=prefix+'_work_card')
    step=next(s for s in steps if s['id']==selected);node=next(n for n in run['spec']['steps'] if n['id']==selected)
    output=step.get('output') or {};execution=output.get('_execution',{})
    with st.container(border=True):
        st.markdown('**'+selected.replace('_',' ').title()+'** · '+step['state'])
        st.write(result_summary(output))
        cols=st.columns(3)
        cols[0].caption('Device / compute');cols[0].write(output.get('device_id') or node['parameters'].get('device_id') or execution.get('compute','Local worker / operator'))
        cols[1].caption('Model');cols[1].write(model_names(output,node['parameters']) or '—')
        cols[2].caption('Dependencies');cols[2].write(', '.join(node.get('depends_on',[])) or 'None')
        pipeline=execution.get('pipeline',pipeline_for(node['operation']))
        if pipeline:st.caption(' → '.join(pipeline))
        if output.get('plan',{}).get('design'):
            plan=output['plan'];st.dataframe(pd.DataFrame(recipe_table(plan['design'])),hide_index=True,width='stretch')
            st.write('**Preparation**');show_values(plan['preparation'])
            st.write('**Equipment assignments**');show_values(plan['equipment'])
            st.write('**Performance protocol**');st.write(plan.get('performance_protocol') or 'Required before dispatch')
        elif 'design' in output:st.dataframe(pd.DataFrame(recipe_table(output['design'])),hide_index=True,width='stretch')
        elif 'returned_data' in output:
            data=output['returned_data']
            if isinstance(data.get('results'),list) and data['results'] and all(isinstance(r,dict) for r in data['results']):
                st.dataframe(pd.DataFrame(data['results']),hide_index=True,width='stretch')
                show_values({k:v for k,v in data.items() if k!='results'})
            else:show_values(data)
        elif 'acquisition' in output:
            acquisition=output['acquisition'];quantified=[{'Specimen':m['sample_id'],'Element':v.get('element'),'W / wt%':v.get('value')}
                for m in acquisition.get('measurements',[]) for v in m.get('icp',[])]
            files=[{'Specimen':m['sample_id'],'Modality':f.get('modality'),'Raw file':f.get('file_name')}
                for m in acquisition.get('measurements',[]) for f in m.get('files',[])]
            if quantified:st.dataframe(pd.DataFrame(quantified),hide_index=True,width='stretch')
            if files:st.dataframe(pd.DataFrame(files),hide_index=True,width='stretch')
            with st.expander('Acquisition receipt'):show_values(service.public_result(acquisition))
        elif output:
            with st.expander('Returned data',expanded=selected in {'training_pool','update','performance'}):show_values(service.public_result(output))
        with st.expander('Input contract and execution details'):
            show_values(service.public_result({'operation':node['operation'],'inputs':node['parameters'],'execution':execution}))
        if controls and step['state']=='WAITING':
            st.info(output['instruction'])
            if output.get('plan',{}).get('approved_performance_protocol'):st.write(output['plan']['approved_performance_protocol'])
            if output.get('approval_required'):
                protocol=st.text_area('Approved performance protocol',value=output.get('plan',{}).get('performance_protocol') or '',key=prefix+'_protocol')
                checked=st.checkbox('I have reviewed the recipes, preparation, device assignments and schedule.',key=prefix+'_approved')
                if st.button('Confirm plan and release workflow',disabled=not checked or not protocol.strip(),key=prefix+'_confirm'):
                    service.acknowledge(run_id,selected,'Operator reviewed and approved the plan.',{'approved':True,'performance_protocol':protocol});st.rerun()
            else:
                with st.expander('Operator return form'):
                    note=st.text_input('Operator completion note',key=prefix+'_return_note_'+selected)
                    fields=output.get('return_fields',[])
                    raw=st.text_area('Structured returned data (JSON)',value=json.dumps({k:None for k in fields},indent=2),key=prefix+'_return_data_'+selected) if fields else '{}'
                    if st.button('Record step return',key=prefix+'_return_'+selected):
                        service.acknowledge(run_id,selected,note,json.loads(raw));st.rerun()
