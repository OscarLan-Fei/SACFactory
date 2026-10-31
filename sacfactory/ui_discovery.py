"""Local discovery workbench for campaigns, evidence, decisions and validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from .compute_runtime import execution
from .discovery.common import digest,query,identity
from .discovery.campaigns import CampaignLedger
from .discovery.workflows import available_models,load_model,train_candidate,pending_from_proposal
from .ui_elements import element_multiselect
from .domain import ELEMENTS


@st.cache_resource(show_spinner=False,max_entries=8)
def cached_model(root,origin,name,serialized,version):
    return load_model(root,origin,name,json.loads(serialized))


def selected_model(store,samples,key):
    names=available_models(store.root,store.origin)
    preferred=next((n for n in ['SAC-Gen-GB','Pretrained SAC-Gen-GB','Pretrained SAC-Gen-Diffusion'] if n in names),names[0])
    name=st.selectbox('Prediction model',names,index=names.index(preferred),key=key)
    return name


def get_model(store,samples,name):
    files=[store.root/'discovery_models'/store.origin.lower()/'index.json',store.root/'sac_gen'/store.origin.lower()/'variants.json']
    from .pretrained import bundle_root
    files.append(bundle_root()/'registry.json')
    version=digest([p.read_text() if p.exists() else '' for p in files])
    return cached_model(str(store.root),store.origin,name,json.dumps(samples,sort_keys=True),version)


def preview_table(rows):
    return pd.DataFrame([{'elements':'+'.join(r['recipe']['elements']),
        'precursor_mmol':'; '.join(f"{e}: {v*r['recipe']['solvent_volume_mL']/1000:.6g}" for e,v in r['recipe']['concentrations_mM'].items()),
        'precursor_mM':'; '.join(f'{e}: {v:.6g}' for e,v in r['recipe']['concentrations_mM'].items()),
        'PAN_g':r['recipe']['pan_mass_g'],'DMF_mL':r['recipe']['solvent_volume_mL'],
        'cutoff_C':r['recipe']['pulse_limit_C'],'pulses':r['recipe']['pulse_count'],
        'predicted_total_W_wt_percent':r['prediction'].get('predicted_total_wt_percent'),
        'predicted_structural_PASS':r['prediction'].get('p_xrd_pass'),
        'probability_status':r['prediction'].get('xrd_probability_status',r['prediction'].get('calibration',{}).get('probability_status')),
        'outside_support':r['prediction'].get('outside_training_support'),
        'model_disagreement':r.get('model_disagreement')} for r in rows])


def save_preview(store,samples,ledger,rows,key,model):
    st.dataframe(preview_table(rows),hide_index=True,width='stretch')
    st.download_button('Export proposed recipes',preview_table(rows).to_csv(index=False).encode(),
                       'discovery_recipes.csv','text/csv',key=key+'_csv')
    preview_signature=digest(rows)
    committed=st.session_state.get(key+'_committed',{})
    already_saved=committed.get('signature')==preview_signature
    if already_saved:st.success('Added pending samples: '+', '.join(committed['sample_ids']))
    if st.button('Freeze forecasts and add to Pending',disabled=not rows or already_saved,key=key+'_pending'):
        if any(row['prediction'].get('model_id')!=model.model_id for row in rows):
            raise ValueError('The prediction model changed; generate fresh forecasts before freezing.')
        frozen=ledger.freeze(samples,{'model_id':model.model_id,'model_name':getattr(model,'variant_name',type(model).__name__),
            'training_membership_status':'NOT_RELEASED' if getattr(model,'is_public_release',False) else 'LOCAL_TRAINING_IDENTITY'})
        training_keys=[identity(s) for s in getattr(model,'training_samples',[])] or getattr(model,'training_keys',[])
        proposal=ledger.propose(frozen['id'],[r['recipe'] for r in rows],[r['prediction'] for r in rows],training_keys,
                                [{'strategy':r.get('strategy'),'target_distance':r.get('target_distance')} for r in rows])
        ids=pending_from_proposal(store,ledger,proposal['id'])
        st.success('Added pending samples: '+', '.join(ids))
        st.session_state[key+'_committed']={'signature':preview_signature,'proposal_id':proposal['id'],'sample_ids':ids}


def render_discovery(store,samples,methods,context):
    from .ui_navigation import preserve_analysis_controls
    preserve_analysis_controls(('discovery_',))
    st.title('SAC-Gen Discovery lab')
    st.caption('Scientific knowledge, protected model updates and executable experimental decisions. ICP W is wt%; precursor amounts are mmol.')
    ledger=CampaignLedger(store.root,store.origin)
    prediction,cycle,knowledge,selection,training,inverse,hypothesis,characterization,validation=st.tabs(
        ['Predict','Evolution','Knowledge','Model disagreement','Set model & updates','Inverse design','Hypotheses','Characterization','Validation'],
        default='Evolution' if samples else 'Predict',key='discovery_tabs_'+store.origin,on_change='rerun')
    try:
        if prediction.open:
            with prediction:
                render_pretrained_prediction(store,context)
        if cycle.open:
            with cycle:
                st.write('**Frozen experiments and feedback**')
                rounds=ledger.records('round');proposals=ledger.records('proposal')
                cols=st.columns(3)
                for col,label,value in zip(cols,['Frozen rounds','Forecast batches','Linked feedback batches'],[len(rounds),len(proposals),len(ledger.records('link'))]):col.metric(label,value)
                evaluation=ledger.evaluate(samples)
                if evaluation:
                    frame=pd.DataFrame(evaluation);st.dataframe(frame,hide_index=True,width='stretch')
                    plotted=frame.dropna(subset=['ICP_MAE_wt_percent'])
                    if len(plotted):st.line_chart(plotted.set_index('created')[['ICP_MAE_wt_percent']])
                else:st.info('Freeze an inverse-design or hypothesis batch, run it, and enter results under Samples & data. The original forecasts remain immutable.')
                if rounds:st.dataframe(pd.DataFrame([{k:r[k] for k in ['id','created','dataset_fingerprint']} for r in rounds]),hide_index=True,width='stretch')
                st.caption('Successive batches can differ in difficulty. Fixed masked-validation and replay curves complement prospective round summaries; a downward round error alone is not causal evidence of improvement.')
                assessments=ledger.records('characterization_assessment')
                if assessments:
                    st.write('**Reviewed characterization feedback**')
                    st.dataframe(pd.DataFrame([{k:r[k] for k in ['created','sample_id','state','reviewer']} for r in assessments]),hide_index=True,width='stretch')
                    st.caption('Committed XRD/STEM screening assessments enter the existing outcomes used by validation and future model fits. SEM remains morphology evidence.')
                st.download_button('Export discovery ledger',json.dumps({'rounds':rounds,'proposals':proposals,'links':ledger.records('link'),'evaluation':evaluation,
                                   'characterization':ledger.records('characterization'),'characterization_reviews':ledger.records('characterization_review'),
                                   'characterization_assessments':assessments,'identity_bindings':ledger.records('identity_binding')},indent=2).encode(),
                                   'discovery_ledger.json','application/json',key='discovery_export_'+store.origin)
        if knowledge.open:
            with knowledge:
                from .discovery.knowledge import public_documents,document,read_upload,history_documents,search
                st.write('**Source-linked research memory**')
                st.caption('Local lexical retrieval over cited method summaries, imported documents and optionally selected experimental records. Retrieved passages are context, not model parameters or instructions.')
                uploads=st.file_uploader('Add a PDF or UTF-8 text document',type=['pdf','txt','md'],key='discovery_upload_'+store.origin)
                source=st.text_input('Document DOI or source URL',key='discovery_document_source_'+store.origin)
                if st.button('Index uploaded document',disabled=uploads is None,key='discovery_index_'+store.origin):
                    entry=read_upload(uploads.name,uploads.getvalue(),source)
                    if any(d['document']['document_id']==entry['document_id'] for d in ledger.records('document')):st.info('This document is already indexed.')
                    else:ledger.add('document',{'origin':store.origin,'document':entry});st.success('Document indexed locally.')
                selected=element_multiselect('Research-memory elements',ELEMENTS,default=context.get('elements',[]),key='discovery_memory_elements_'+store.origin)
                include=st.checkbox('Include matching historical experiment records in local search',key='discovery_history_'+store.origin)
                question=st.text_input('Search literature and experimental memory',value='single atom synthesis temperature pulse precursor mass balance',key='discovery_question_'+store.origin)
                if st.button('Retrieve evidence',key='discovery_search_'+store.origin):
                    documents=public_documents()+[r['document'] for r in ledger.records('document')]
                    if include:documents+=history_documents([s for s in samples if set(s['elements'])&set(selected)])
                    result=search(documents,question)
                    if not result:st.info('No matching indexed evidence.')
                    for row in result:
                        st.write('**'+row['title']+'**');st.write(row['passage']);st.caption(row['source'])
                st.caption('Experiment chat can retrieve the bundled public sources and explicitly shared samples. Uploaded documents stay local and are not automatically shared with the API.')
        if selection.open:
            with selection:
                from .discovery.calibration import disagreement
                st.write('**Compare model opinions on the same recipes**')
                selected=element_multiselect('Comparison elements',ELEMENTS,default=context.get('elements',[]),key='discovery_comparison_elements_'+store.origin)
                names=st.multiselect('Decision models',available_models(store.root,store.origin),default=available_models(store.root,store.origin),key='discovery_models_'+store.origin)
                if st.button('Compare selected observed recipes',disabled=len(names)<2 or not selected,key='discovery_compare_'+store.origin):
                    candidates=[query(s) for s in samples if set(s['elements'])&set(selected)]
                    if not candidates:raise ValueError('No recorded recipes match the selected elements.')
                    predictions={name:get_model(store,samples,name).predict(candidates) for name in names}
                    rows=disagreement(candidates,predictions)
                    frame=pd.DataFrame(rows);st.dataframe(frame.sort_values('ICP_model_SD_wt_percent',ascending=False),hide_index=True,width='stretch')
                    st.download_button('Export model disagreement',frame.to_csv(index=False).encode(),'model_disagreement.csv',key='discovery_disagree_export_'+store.origin)
                st.caption('Disagreement between models is a selection signal, not calibrated uncertainty or independent validation. Historical recipe predictions can be in-sample; use Validation for protected evaluation.')
        if training.open:
            with training:
                st.write('**Train a versioned candidate without replacing the current deployment**')
                family=st.selectbox('Candidate family',['SAC-Gen-Set','Calibrated discovery ensemble','SAC-Gen-GB','SAC-Gen-Diffusion'],key='discovery_family_'+store.origin)
                epochs=st.number_input('Candidate training epochs',5,2000,100,key='discovery_train_epochs_'+store.origin)
                st.caption('Set uses a small permutation-invariant element encoder and a shared mass denominator. The calibrated ensemble reserves separate weight-fit and interval-calibration recipes. Candidate models become selectable here; the existing Next batch deployment remains unchanged.')
                if st.button('Train and verify candidate',disabled=not samples,key='discovery_train_'+store.origin):
                    with st.spinner('Training and checking saved-model forecasts...'),execution({'device':'auto','threads':4,'cpu_jobs':4}):
                        item=train_candidate(store.root,samples,store.origin,family,int(epochs))
                    ledger.add('training',{'origin':store.origin,'candidate':item});st.success('Verified candidate: '+item['model_id'])
                    st.json(item)
                previous=ledger.records('training')
                if previous:st.dataframe(pd.DataFrame([r['candidate'] for r in previous]),hide_index=True,width='stretch')
        if inverse.open:
            with inverse:
                from .discovery.inverse import candidates,rank
                from .discovery.calibration import disagreement
                st.write('**Generate executable recipes for a target response range**')
                elements=element_multiselect('Inverse-design elements',ELEMENTS,default=context.get('elements',[]),key='discovery_inverse_elements_'+store.origin)
                rows=[{'element':e,'lower_mmol':.1,'upper_mmol':1.} for e in elements]
                bounds=st.data_editor(pd.DataFrame(rows,columns=['element','lower_mmol','upper_mmol']),disabled=['element'],hide_index=True,key='discovery_bounds_'+store.origin)
                cols=st.columns(4)
                pan=cols[0].number_input('PAN / g',.001,100.,2.,key='discovery_pan_'+store.origin)
                volume=cols[1].number_input('DMF / mL',.01,10000.,20.,key='discovery_dmf_'+store.origin)
                tmin=cols[2].number_input('Lower cutoff / C',10,3990,800,step=10,key='discovery_tmin_'+store.origin)
                tmax=cols[3].number_input('Upper cutoff / C',10,3990,1600,step=10,key='discovery_tmax_'+store.origin)
                pulses=st.multiselect('Allowed pulse counts',list(range(1,17)),default=[1,2,4],key='discovery_pulses_'+store.origin)
                cols=st.columns(4)
                wmin=cols[0].number_input('Target lower total ICP / wt%',0.,100.,1.,key='discovery_wmin_'+store.origin)
                wmax=cols[1].number_input('Target upper total ICP / wt%',0.,100.,3.,key='discovery_wmax_'+store.origin)
                pmin=cols[2].number_input('Minimum predicted structural PASS',0.,1.,.5,key='discovery_pmin_'+store.origin)
                size=cols[3].number_input('Proposed experiments',1,12,6,key='discovery_size_'+store.origin)
                name=selected_model(store,samples,'discovery_inverse_model_'+store.origin)
                strategy=st.selectbox('Selection objective',['Target attainment','Model disagreement','Balanced learning'],key='discovery_strategy_'+store.origin)
                outside=st.checkbox('Allow explicitly flagged extrapolation',key='discovery_extrapolation_'+store.origin)
                count=st.number_input('Candidate search budget',16,2048,128,step=16,key='discovery_pool_'+store.origin)
                cap=st.number_input('Maximum total precursor amount / mmol',.001,10000.,max(.1,float(len(elements))),key='discovery_total_cap_'+store.origin)
                signature=digest([samples,elements,bounds.to_dict('records'),pan,volume,tmin,tmax,pulses,wmin,wmax,pmin,size,name,strategy,outside,count,cap])
                if st.button('Generate target-conditioned batch',disabled=not elements,key='discovery_inverse_run_'+store.origin):
                    base={**context,'pan_mass_g':pan,'solvent_volume_mL':volume}
                    pool=candidates(base,elements,{r['element']:(r['lower_mmol'],r['upper_mmol']) for r in bounds.to_dict('records')},(tmin,tmax),pulses,int(count),max_total_mmol=cap)
                    model=get_model(store,samples,name)
                    with st.spinner('Predicting candidate recipes...'),execution({'device':'auto','threads':4,'cpu_jobs':4}):
                        predictions=model.predict(pool) if pool else []
                        other=get_model(store,samples,'Fixed prior').predict(pool) if pool else []
                    signals=disagreement(pool,{name:predictions,'Fixed reference':other})
                    selected=rank(pool,predictions,(wmin,wmax),pmin,int(size),outside,signals,strategy)
                    st.session_state['discovery_inverse_result_'+store.origin]=(signature,selected)
                saved=st.session_state.get('discovery_inverse_result_'+store.origin)
                if saved and saved[0]==signature:
                    selected=saved[1]
                    if selected:save_preview(store,samples,ledger,selected,'discovery_inverse_preview_'+store.origin,get_model(store,samples,name))
                    else:st.info('No supported candidates satisfy these constraints. Review the declared ranges, model support and probability threshold.')
                st.caption('This is target-conditioned bounded search using the selected forward model. It does not yet train a separate generative distribution over recipes. Outcomes remain predictions, not observations.')
        if hypothesis.open:
            with hypothesis:
                from .discovery.hypotheses import contrasts,register,assess
                st.write('**Pre-register a one-factor physical hypothesis**')
                selected=element_multiselect('Hypothesis elements',ELEMENTS,default=context.get('elements',[]),key='discovery_hypothesis_elements_'+store.origin)
                choices=[s for s in samples if set(s['elements'])&set(selected)]
                if choices:
                    baseline=st.selectbox('Baseline recipe',[s['sample_id'] for s in choices],key='discovery_baseline_'+store.origin)
                    factor=st.selectbox('Changed factor',['temperature','pulse','amount'],key='discovery_factor_'+store.origin)
                    specified=st.text_input('Intervention values (C, pulse count, or mmol)',value='900,1200',key='discovery_interventions_'+store.origin)
                    statement=st.text_area('Testable hypothesis',value='Changing controlled cutoff while keeping the feed fixed changes metal retention.',key='discovery_statement_'+store.origin)
                    name=selected_model(store,samples,'discovery_hypothesis_model_'+store.origin)
                    if st.button('Predict and register contrasts',key='discovery_hypothesis_run_'+store.origin):
                        base=next(s for s in choices if s['sample_id']==baseline)
                        values=[float(v.strip()) for v in specified.split(',')]
                        if factor=='pulse':
                            if any(not v.is_integer() for v in values):raise ValueError('Pulse interventions must be integers.')
                            values=[int(v) for v in values]
                        recipes=contrasts(base,factor,values);model=get_model(store,samples,name)
                        predictions=model.predict(recipes)
                        card=register(recipes,predictions,factor,statement)
                        ledger.add('hypothesis',{'origin':store.origin,'card':card})
                        st.session_state['discovery_hypothesis_result_'+store.origin]=(card,name)
                    saved=st.session_state.get('discovery_hypothesis_result_'+store.origin)
                    if saved:
                        card,used_name=saved
                        rows=[{'recipe':r,'prediction':p,'strategy':'Controlled '+card['factor']+' contrast'} for r,p in zip(card['recipes'],card['predictions'])]
                        save_preview(store,samples,ledger,rows,'discovery_hypothesis_preview_'+store.origin,get_model(store,samples,used_name))
                for item in ledger.records('hypothesis')[-10:]:
                    card=item['card']
                    with st.expander(card['statement']):st.dataframe(pd.DataFrame(assess(card,samples)),hide_index=True,width='stretch')
                st.caption('Evidence assessment uses only measurements reported after registration. Replicated controlled experiments remain necessary for a mechanistic conclusion.')
        if characterization.open:
            with characterization:
                from .ui_characterization import render_characterization
                render_characterization(store,samples,ledger,context)
        if validation.open:
            with validation:
                from .discovery.validation import validate,replay
                st.write('**Protected validation and equal-budget replay**')
                mode=st.radio('Evaluation',['Grouped model validation','Acquisition replay'],key='discovery_validation_mode_'+store.origin)
                epochs=st.number_input('Validation set-model epochs',5,500,80,key='discovery_val_epochs_'+store.origin)
                folds=st.number_input('Recipe-group folds',2,10,5,key='discovery_folds_'+store.origin)
                include=st.checkbox('Include fold-trained SAC-Gen-GB',value=True,key='discovery_include_sac_'+store.origin)
                cols=st.columns(3)
                initial=cols[0].number_input('Initial replay observations',12,1000,16,key='discovery_initial_'+store.origin)
                budget=cols[1].number_input('Replay observation budget',12,2000,36,key='discovery_budget_'+store.origin)
                step=cols[2].number_input('Replay queries per round',1,100,10,key='discovery_step_'+store.origin)
                if st.button('Run protected evaluation',disabled=not samples,key='discovery_validate_'+store.origin):
                    progress=st.empty()
                    def report(event):progress.info(event.get('stage','Evaluating'))
                    with execution({'device':'cpu','threads':4,'cpu_jobs':4}):
                        result=validate(samples,store.origin,int(folds),int(epochs),include_sac=include,progress=report) if mode=='Grouped model validation' else replay(samples,store.origin,int(budget),int(initial),int(step),int(epochs),progress=report)
                    ledger.add('validation',{'origin':store.origin,'mode':mode,'result':result});progress.success('Evaluation complete')
                reports=ledger.records('validation')
                if reports:
                    chosen=st.selectbox('Saved evaluation',[r['id'] for r in reports],key='discovery_saved_evaluation_'+store.origin)
                    report=next(r for r in reports if r['id']==chosen);result=report['result']
                    frame=pd.DataFrame(result.get('metrics',result.get('curves',[])))
                    st.dataframe(frame,hide_index=True,width='stretch')
                    if 'curves' in result and len(frame):st.line_chart(frame.pivot(index='observations',columns='policy',values='ICP_MAE_wt_percent'))
                    st.caption(result['note'])
                    st.download_button('Export full evaluation and masks',json.dumps(result,indent=2).encode(),'discovery_validation.json','application/json',key='discovery_validation_export_'+store.origin)
                st.caption('Validation reserves separate model-fit, ensemble-weight, calibration and outer-test recipe groups. All component targets from one specimen stay together.')
    except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:
        st.error(str(exc))


def render_pretrained_prediction(store,context):
    """Predict a new recipe directly, without recording an experimental outcome."""
    from .pretrained import available,load
    from .discovery.common import query
    st.write('**Predict with the included trained SAC-Gen models**')
    if store.origin!='REAL':
        st.info('Switch to Real experiments to use the public pretrained release. DEMO models remain separate.')
        return
    names=list(available())
    if not names:
        st.info('This installation has no public pretrained bundle.')
        return
    name=st.selectbox('Pretrained model',names,index=names.index('SAC-Gen-GB') if 'SAC-Gen-GB' in names else 0,key='pretrained_variant')
    elements=element_multiselect('Prediction elements',ELEMENTS,default=['Fe'],key='pretrained_elements')
    amounts=st.data_editor(pd.DataFrame([{'element':e,'precursor_mmol':1.} for e in elements],columns=['element','precursor_mmol']),
        disabled=['element'],hide_index=True,width='stretch',key='pretrained_amounts_'+digest(elements))
    cols=st.columns(4)
    pan=cols[0].number_input('Prediction PAN / g',.001,100.,2.,key='pretrained_pan')
    volume=cols[1].number_input('Prediction DMF / mL',.01,10000.,20.,key='pretrained_dmf')
    cutoff=cols[2].number_input('Prediction cutoff / C',10,3990,1000,step=10,key='pretrained_cutoff')
    pulses=cols[3].number_input('Prediction pulses',1,100,1,key='pretrained_pulses')
    st.caption('These are proposed controls. No ICP/XRD result, API key or local training is required. '
        'Exact training-recipe membership is withheld, so predictions carry a conservative support flag. '
        'Forecast intervals and probabilities need evaluation on your campaign.')
    signature=digest([name,amounts.to_dict('records'),pan,volume,cutoff,pulses])
    if st.button('Predict recipe',disabled=not elements,key='pretrained_predict'):
        model=load(name)
        from .precursor_catalog import resolve_planning_precursors
        recipe=query({**model.process_defaults,'origin':'REAL','elements':elements,
            'concentrations_mM':{r['element']:float(r['precursor_mmol'])*1000/volume for r in amounts.to_dict('records')},
            'pan_mass_g':pan,'solvent_volume_mL':volume,'pulse_limit_C':cutoff,'pulse_count':int(pulses),
            'precursors':resolve_planning_precursors({},context,elements)})
        with st.spinner('Predicting with the bundled model'):
            result=model.predict([recipe])[0]
        st.session_state['pretrained_prediction']=(signature,result)
    saved=st.session_state.get('pretrained_prediction')
    if saved and saved[0]==signature:
        result=saved[1]
        st.metric('Predicted total ICP W / wt%',f"{result['predicted_total_wt_percent']:.4f}")
        st.metric('Predicted structural PASS probability',f"{result['p_xrd_pass']:.3f}")
        st.dataframe(pd.DataFrame(result['icp_predictions']),hide_index=True,width='stretch')
        st.download_button('Download pretrained forecast',json.dumps(result,indent=2,default=str).encode(),
            'sac_gen_forecast.json','application/json',key='pretrained_download')
