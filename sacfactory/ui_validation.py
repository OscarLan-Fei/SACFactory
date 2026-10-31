"""SAC-Gen prediction validation interface."""
import json
from pathlib import Path
import time
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from .domain import ELEMENTS
from .planning import fingerprint
from .weighted_validation import weighted_validation,PROTOCOLS
from .ui_elements import element_multiselect
from .compute_runtime import execution,hardware
from . import pulseicp_bridge as bridge
from .model_benchmarks import PUBLIC_MODELS as MODELS
from .validation_groups import RANDOM_RECIPE_CV


def calculate_validation(payload,progress=None):
    """Run live callbacks outside Streamlit's cached-element replay."""
    parameters=json.loads(payload);runtime=parameters.pop('compute_runtime',None)
    models=parameters.pop('comparison_models',None)
    with execution(runtime,progress):
        if models is None:return weighted_validation(**parameters)
        from .model_benchmarks import compare
        points,summary,info=compare(parameters['samples'],parameters['methods'],parameters['origin'],parameters['targets'],
            models=models,profile=parameters['profile'],batches_per_fold=parameters['batches_per_fold'],
            sac_settings=parameters['icp_settings'],sac_family=parameters['icp_family'],
            protocol=parameters['protocol'],max_folds=parameters.get('max_folds',3),
            scale=parameters['delta'],agreement_weight=parameters['weight'],seed=parameters.get('seed',42))
        return {'kind':'multi_model_validation','points':points,'summary':summary,'info':info}


def render_weighted_validation(samples,methods,origin,root=None):
    st.write('**Weighted validation**')
    if root is not None:
        registry_path=Path(root)/'comparison_controls'/origin.lower()/'latest.json'
        if registry_path.exists():
            try:
                controls=json.loads(registry_path.read_text())
                selected=controls.get('data_only',{})
                if selected:st.caption(f"Trained experimental-only reference: {selected['backend']} · {selected['model_id']}")
                if controls.get('science'):st.caption('A fixed science-prior reference is available; it is not fitted to current experimental labels.')
                if controls.get('scientific_descriptors'):st.caption('A supervised scientific-descriptor reference is available using the same experimental-only backend.')
                name=controls.get('run_id','')
                if Path(name).name==name and name not in {'.','..',''}:
                    comparison=registry_path.parent/name/'comparison_metrics.csv'
                    if comparison.exists():
                        key=f'show_information_comparison_{origin}_{name}'
                        if st.button('Show saved information-control comparison',key=key):
                            st.session_state[key+'_visible']=True
                        if st.session_state.get(key+'_visible'):
                            frame=pd.read_csv(comparison)
                            columns=[c for c in ['model','ICP_MAE','ICP_R2','XRD_Brier','XRD_ROC_AUC','balanced_accuracy','FAIL_recall','PASS_recall'] if c in frame]
                            st.dataframe(frame[columns],hide_index=True,width='stretch')
                            st.caption('Fitted-model entries are means of three recipe-grouped five-fold repeats over the same frozen experimental snapshot. The fixed prior is evaluated once. No training is started by this view.')
            except (ValueError,KeyError,OSError):pass
    observed=list(dict.fromkeys(e for s in samples if s.get('origin')==origin for e in s['elements']))
    st.session_state.setdefault(f'weighted_elements_{origin}',observed)
    elements=element_multiselect('Validation elements',ELEMENTS,key=f'weighted_elements_{origin}')
    if st.button('Select all validation models',key=f'weighted_models_all_{origin}'):
        st.session_state[f'weighted_models_{origin}']=[model for model in MODELS if model!='SAC-Gen']
    if f'weighted_models_{origin}' in st.session_state:
        st.session_state[f'weighted_models_{origin}']=list(dict.fromkeys('SAC-Gen-Diffusion' if m=='SAC-Gen' else m for m in st.session_state[f'weighted_models_{origin}'] if m=='SAC-Gen' or m in MODELS))
    models=st.multiselect('Validation models',[model for model in MODELS if model!='SAC-Gen'],
        default=None if f'weighted_models_{origin}' in st.session_state else ['SAC-Gen-Diffusion'],key=f'weighted_models_{origin}',
        format_func=lambda m: 'Descriptor GB (database chemistry)' if m=='Scientific descriptors Gradient boosting' else m)
    st.caption('Selected models share the same outer masks and score formula. SAC-Gen-Diffusion and SAC-Gen-GB share the physical framework and structural-assessment head; their ICP learners differ. Descriptor GB excludes elicited Joule-heating priors.')
    deployment_settings(root,origin)
    cols=st.columns(2)
    variants=[m for m in models if m in {'SAC-Gen-Diffusion','SAC-Gen-GB'}]
    options=variants or ['SAC-Gen-Diffusion']
    variant_key=f'weighted_deployment_variant_{origin}'
    if st.session_state.get(variant_key) not in options:st.session_state[variant_key]=options[0]
    variant=cols[0].selectbox('Deployment variant',options,key=variant_key,disabled=not variants)
    family='Gradient boosting residual' if variant=='SAC-Gen-GB' else 'Diffusion Transformer'
    protocol=cols[1].selectbox('Validation protocol',PROTOCOLS,key=f'weighted_protocol_{origin}')
    maximum=max(1,len({s.get('batch_id') for s in samples})-2)
    if f'weighted_cv_batches_{origin}' in st.session_state:
        st.session_state[f'weighted_cv_batches_{origin}']=min(maximum,max(1,int(st.session_state[f'weighted_cv_batches_{origin}'])))
    if protocol == RANDOM_RECIPE_CV:
        folds=st.number_input('Recipe folds',2,8,(None if f'weighted_recipe_folds_{origin}' in st.session_state else 5),key=f'weighted_recipe_folds_{origin}')
        groups=1
        st.caption('Whole recipes are hidden together. Batch identifiers may occur in both pools; no specimen or identical recipe does.')
    else:
        folds=3
        groups=st.number_input('Test batches per fold',1,maximum,(None if f'weighted_cv_batches_{origin}' in st.session_state else max(1,min(maximum,(maximum+6)//5))),key=f'weighted_cv_batches_{origin}')
    with st.expander('Training and scoring settings'):
        machine=hardware();device=st.selectbox('Compute device',['auto','cuda','cpu'],key=f'weighted_device_{origin}')
        cols=st.columns(3)
        members=cols[0].number_input('Ensemble members',1,5,(None if f'weighted_aux_members_{origin}' in st.session_state else 1),key=f'weighted_aux_members_{origin}')
        epochs=cols[1].number_input('Training epochs',1,2000,(None if f'weighted_aux_epochs_{origin}' in st.session_state else 100),key=f'weighted_aux_epochs_{origin}')
        draws=cols[2].number_input('Draws per member',2,512,(None if f'weighted_aux_draws_{origin}' in st.session_state else 32),key=f'weighted_aux_draws_{origin}')
        steps=st.number_input('Diffusion steps',10,200,(None if f'weighted_aux_steps_{origin}' in st.session_state else 30),key=f'weighted_aux_steps_{origin}',disabled='diffusion' not in family.lower())
        seed=st.number_input('Random seed',0,2147483647,(None if f'weighted_seed_{origin}' in st.session_state else 42),key=f'weighted_seed_{origin}')
        refined=st.checkbox('Validation-driven training and XRD/STEM refinement',value=(None if f'weighted_refine_{origin}' in st.session_state else False),
            disabled='SAC-Gen-Diffusion' not in models,key=f'weighted_refine_{origin}')
        if refined and 'SAC-Gen-Diffusion' in models:
            st.caption('Each fit selects its learning rate and 100–400 training epochs using an internal recipe holdout, then refits its complete training pool. XRD/STEM calibration and the decision threshold use training-only head cross-fitting.')
        cols=st.columns(2)
        threads=cols[0].number_input('CPU training threads',1,machine['cpu_count'],(None if f'weighted_threads_{origin}' in st.session_state else min(4,machine['cpu_count'])),key=f'weighted_threads_{origin}')
        workers=cols[1].number_input('CPU workers',1,machine['cpu_count'],(None if f'weighted_jobs_{origin}' in st.session_state else min(8,machine['cpu_count'])),key=f'weighted_jobs_{origin}')
        cols=st.columns(2)
        weight=cols[0].number_input('ICP agreement weight',.05,.95,(None if f'weighted_weight_{origin}' in st.session_state else .6),step=.05,key=f'weighted_weight_{origin}')
        scale=cols[1].number_input('ICP error scale / wt% points',.0001,100.,(None if f'weighted_delta_{origin}' in st.session_state else .03 if origin=='REAL' else .25),format='%.4f',key=f'weighted_delta_{origin}')
        st.caption(f"XRD weight: {1-weight:.2f}. Agreement is an error score, not percent accuracy. {machine['gpu_name'] or 'CPU'}.")
    settings={**bridge.configuration()['model'],'members':int(members),'epochs_baseline':int(epochs),'epochs_diffusion':int(epochs),
        'draws_per_member':int(draws),'diffusion_steps':int(steps),'token_width':32,'attention_heads':4,'attention_layers':2,
        'crossfit_grouping':'recipe' if protocol == RANDOM_RECIPE_CV else 'batch','seed':int(seed),'refine_xrd':True}
    if refined and 'SAC-Gen-Diffusion' in models:
        settings.update(training_variant='validated-token-v1',refine_xrd=True,tuning_max_epochs=400,
            tuning_min_epochs=100,tuning_interval=20,tuning_patience=3,tuning_draws=16,
            candidate_learning_rates=[.0003,.001])
    runtime={'device':device,'threads':int(threads),'cpu_jobs':int(workers),'runtime_version':'device-batched-v1'}
    payload=json.dumps({'samples':samples,'methods':methods,'origin':origin,'targets':elements,'protocol':protocol,
        'profile':'chemistry_pulseicp','batches_per_fold':int(groups),'max_folds':int(folds),'seed':int(seed),'icp_family':family,'icp_settings':settings,
        'endpoint_weights':{'icp':float(weight),'xrd_pass':1-float(weight)},'delta':float(scale),'weight':float(weight),
        'sac_gen':True,'include_system_score':False,'agreement_method':'component_agreement_v1','compute_runtime':runtime},sort_keys=True)
    if models!=['SAC-Gen']:payload=json.dumps({**json.loads(payload),'comparison_models':models},sort_keys=True)
    signature=fingerprint(payload)
    if st.button('Run masked validation',disabled=not elements or not models,key=f'weighted_run_{origin}'):
        started=time.perf_counter();bar=st.progress(0.,text='Preparing validation');state={'fold':0,'folds':1}
        def progress(event):
            if 'fold' in event:state.update(fold=event['fold'],folds=event['folds'])
            value=1. if event.get('completed') else max(0.,(state['fold']-1)/max(1,state['folds']))
            bar.progress(min(1.,value),text=f"Fold {state['fold']}/{state['folds']} · {event.get('stage','Training')} · {time.perf_counter()-started:.0f} s")
        try:
            previous=st.session_state.get(f'weighted_result_{origin}')
            if not previous or previous[0]!=signature:
                st.session_state[f'weighted_result_{origin}']=(signature,calculate_validation(payload,progress))
            bar.progress(1.,text='Validation complete')
        except (ValueError,RuntimeError) as exc:bar.empty();st.error(str(exc))
    if root is not None and st.button('Train deployment model',key=f'sac_gen_deploy_{origin}',disabled=not variants):
        from .sac_gen import train_deployment
        try:
            with st.spinner('Training deployment model'),execution(runtime):saved=train_deployment(samples,origin,root,family,settings,methods)
            st.success(f"{variant} model {saved['model_id']} is ready.")
        except (ValueError,RuntimeError) as exc:st.error(str(exc))
    saved=st.session_state.get(f'weighted_result_{origin}')
    if saved and saved[0]==signature:
        if isinstance(saved[1],dict):render_multiple_models(saved[1],origin)
        else:render_results(saved[1],origin)
    elif saved:st.info('Settings or data changed. Run validation to refresh the results.')
    render_saved_deployment_validation(root,origin,samples)


def render_saved_deployment_validation(root,origin,samples):
    """Load frozen validation results on demand, without launching training."""
    if root is None:return
    base=Path(root)/'sac_gen'/origin.lower();pointer=base/'latest.json'
    if not pointer.exists():return
    try:
        item=json.loads(pointer.read_text());name=item['run_id']
        variant_registry=base/'variants.json'
        if variant_registry.exists():
            registry=json.loads(variant_registry.read_text());available=list(registry.get('variants',{}))
            current=item.get('name','SAC-Gen-Diffusion')
            selected=st.selectbox('Saved SAC-Gen variant',available,index=available.index(current) if current in available else 0,
                key=f'saved_variant_{origin}')
            if st.button('Use saved variant for forecasting',disabled=selected==current,key=f'activate_variant_{origin}'):
                from .sac_gen import activate_variant
                activate_variant(root,origin,selected,samples)
                st.rerun()
        if Path(name).name!=name or name in {'.','..'}:raise ValueError('Invalid model reference.')
        directory=base/name/'validation'
        if not (directory/'validation_metrics.csv').exists():return
        st.write('**Saved deployment validation**')
        st.caption(f"Model {item['model_id']} · {item.get('validation_protocol','Recorded validation masks')}. These predictions were made before the full-data deployment fit.")
        key=f"saved_deployment_validation_{origin}_{item['model_id']}"
        if st.button('Show saved validation results',key=f'show_{key}'):
            points=pd.read_csv(directory/'out_of_fold_predictions.csv')
            summary=pd.read_csv(directory/'validation_metrics.csv')
            info=json.loads((directory/'validation_methods.json').read_text())
            st.session_state[key]={'points':points,'summary':summary,'info':info}
        result=st.session_state.get(key)
        if result:
            if result['info']['dataset_fingerprint']!=fingerprint(samples):
                st.info('Current records differ from the saved validation snapshot. The results below describe the saved training run.')
            render_multiple_models(result,f"{origin}_saved_{item['model_id']}")
            repeat_file=directory/'repeat_metrics.csv'
            if repeat_file.exists():
                repeats=pd.read_csv(repeat_file)
                with st.expander('Repeated validation stability'):
                    columns=[c for c in ['seed','ICP_MAE','ICP_R2','XRD_Brier','XRD_ROC_AUC','balanced_accuracy','FAIL_recall','PASS_recall'] if c in repeats]
                    st.dataframe(repeats[columns],hide_index=True,width='stretch')
                    st.caption('The plot above uses the matched seed-42 repeat. Rows here are separate five-fold repeats over the same specimens; their mean is the repeated-validation summary.')
    except (ValueError,KeyError,OSError) as exc:
        st.info(f'Saved validation unavailable: {exc}')


def render_multiple_models(result,origin):
    from .ui_model_diagnostics import render_outcome_score,render_model_diagnostics
    from .validation_diagnostics import common_rows
    points,summary,info=result['points'],result['summary'],result['info']
    if points.empty:
        st.info('No evaluable predictions for this selection. Check the training batches and protocol.')
    else:
        render_outcome_score(common_rows(points,info['models']),origin,f'validation_outcome_{origin}')
    with st.expander('ICP and XRD/STEM model diagnostics',expanded=False):
        st.dataframe(summary,hide_index=True,width='stretch')
        render_model_diagnostics(points,origin,info['models'],key_prefix='weighted_diagnostics',include_score=False)
    st.download_button('Download all model predictions',points.to_csv(index=False).encode(),f'{origin.lower()}_weighted_model_predictions.csv','text/csv',key=f'weighted_models_predictions_{origin}')
    st.download_button('Download model comparison metrics',summary.to_csv(index=False).encode(),f'{origin.lower()}_weighted_model_metrics.csv','text/csv',key=f'weighted_models_metrics_{origin}')
    st.download_button('Download validation methods and folds',json.dumps(info,indent=2,allow_nan=False).encode(),f'{origin.lower()}_weighted_model_methods.json','application/json',key=f'weighted_models_methods_{origin}')


def deployment_settings(root,origin):
    if root is None:return
    base=Path(root)/'sac_gen'/origin.lower();pointer=base/'latest.json'
    if not pointer.exists():return
    try:
        item=json.loads(pointer.read_text());name=item['run_id']
        if Path(name).name!=name or name in {'.','..'}:raise ValueError('Invalid model reference.')
        meta=json.loads((base/name/'sac_gen.json').read_text());st.caption(f"Deployed model: {meta['family']} · {meta['model_id']} · {len(meta['training_ids'])} source specimens")
        key=f'sac_gen_deployed_defaults_{origin}'
        if st.button('Use deployment settings',key=f'sac_gen_use_deployed_{origin}') or st.session_state.get(key)!=meta['model_id']:
            config=meta['settings'];st.session_state.update({f'weighted_icp_family_{origin}':meta['family'],
                f'weighted_aux_members_{origin}':int(config['members']),f'weighted_aux_epochs_{origin}':int(config['epochs_baseline']),
                f'weighted_aux_draws_{origin}':int(config['draws_per_member']),f'weighted_aux_steps_{origin}':int(config['diffusion_steps']),
                f'weighted_seed_{origin}':int(config.get('tuning_selection',{}).get('requested_seed',config['seed'])),key:meta['model_id']})
            st.session_state[f'weighted_refine_{origin}']=config.get('training_variant')=='validated-token-v1'
            validation=item.get('validation_settings',{})
            if validation:
                st.session_state.update({f'weighted_weight_{origin}':float(validation['icp_weight']),
                    f'weighted_delta_{origin}':float(validation['icp_error_scale_wt_percent']),
                    f'weighted_recipe_folds_{origin}':int(validation['folds'])})
            if config.get('crossfit_grouping')=='recipe':
                st.session_state[f'weighted_protocol_{origin}']=RANDOM_RECIPE_CV
        training_selection=base/name/'loading'/'training_selection.json'
        if training_selection.exists():
            with st.expander('Deployment training selection'):
                if st.button('Show training curves',key=f'show_training_curves_{origin}_{name}'):
                    detail=json.loads(training_selection.read_text())
                    trials=detail['trials']
                    st.dataframe(pd.DataFrame([{k:v for k,v in trial.items() if k!='history'} for trial in trials]),hide_index=True,width='stretch')
                    curves=pd.DataFrame([row for trial in trials for row in trial['history']])
                    curves=curves.melt(id_vars=['learning_rate','epoch'],value_vars=['training_loss','validation_loss'],var_name='curve',value_name='loss')
                    chart=alt.Chart(curves).mark_line().encode(x=alt.X('epoch:Q',title='Training epoch'),
                        y=alt.Y('loss:Q',title='Denoising / likelihood loss'),color='curve:N',strokeDash='learning_rate:N',
                        tooltip=['epoch:Q','learning_rate:Q','curve:N','loss:Q']).properties(height=330)
                    st.altair_chart(chart,width='stretch')
                    st.caption('Validation uses fixed diffusion noise and a training-only recipe holdout. Learning rate is selected using decoded ICP MAE; the final model is refitted on all supplied training specimens.')
    except (ValueError,KeyError,OSError) as exc:st.info(f'Model configuration unavailable: {exc}')


def render_results(result,origin):
    points,samples,report,audit,info=result
    numeric=points.dropna(subset=['predicted_w_wt_percent','observed_w_wt_percent']) if len(points) else points
    xrd=next((r for r in info['outcome_metrics'] if r['endpoint']=='xrd_pass'),{})
    from .ui_model_diagnostics import render_outcome_score
    render_outcome_score(points,origin,f'validation_outcome_{origin}')
    with st.expander('ICP regression details',expanded=False):
        mae=float(mean_absolute_error(numeric.observed_w_wt_percent,numeric.predicted_w_wt_percent)) if len(numeric) else None
        rmse=float(np.sqrt(mean_squared_error(numeric.observed_w_wt_percent,numeric.predicted_w_wt_percent))) if len(numeric) else None
        r2=float(r2_score(numeric.observed_w_wt_percent,numeric.predicted_w_wt_percent)) if len(numeric)>1 and np.ptp(numeric.observed_w_wt_percent)>0 else None
        st.subheader('ICP regression: predicted vs measured')
        for col,label,value in zip(st.columns(5),['ICP MAE / wt%','ICP RMSE / wt%','ICP R²','XRD/STEM Brier','XRD/STEM ROC-AUC'],[mae,rmse,r2,xrd.get('brier'),xrd.get('roc_auc')]):col.metric(label,f'{value:.4f}' if value is not None else 'Unavailable')
        st.caption(f"{len(numeric)} ICP pairs · {xrd.get('evaluated_samples',0)} XRD pairs · Mean agreement: {info['sample_weighted_score']:.2f}/100" if info['sample_weighted_score'] is not None else 'No complete agreement pairs.')
        if len(numeric):
            frame=numeric.copy();frame['XRD/STEM']=frame.xrd_pass_observed.map({1:'PASS',0:'FAIL'}).fillna('Unmeasured')
            limit=max(.01,float(frame[['predicted_w_wt_percent','observed_w_wt_percent']].max().max())*1.06)
            values=alt.Data(values=frame.astype(object).where(pd.notna(frame),None).to_dict('records'))
            dots=alt.Chart(values).mark_point(filled=True,size=45,opacity=.65).encode(
                x=alt.X('predicted_w_wt_percent:Q',title='Predicted ICP W / wt%',scale=alt.Scale(domain=[0,limit])),
                y=alt.Y('observed_w_wt_percent:Q',title='Measured ICP W / wt%',scale=alt.Scale(domain=[0,limit])),
                color=alt.Color('XRD/STEM:N',scale=alt.Scale(domain=['PASS','FAIL','Unmeasured'],range=['#268d88','#d46a27','#71808c'])),
                tooltip=['sample_id:N','element:N','batch_id:N','XRD/STEM:N','predicted_w_wt_percent:Q','observed_w_wt_percent:Q'])
            ideal=alt.Chart(pd.DataFrame({'x':[0,limit],'y':[0,limit]})).mark_line(strokeDash=[4,4],color='#8a98a6').encode(x='x:Q',y='y:Q');chart=dots+ideal
            if np.ptp(frame.predicted_w_wt_percent)>1e-12:
                slope,intercept=np.polyfit(frame.predicted_w_wt_percent,frame.observed_w_wt_percent,1)
                line=alt.Chart(pd.DataFrame({'x':[0,limit],'y':[intercept,intercept+slope*limit]})).mark_line(color='#7453a4',clip=True).encode(x='x:Q',y='y:Q')
                chart=chart+line;st.caption(f'OLS: y = {slope:.3f}x {intercept:+.3f}. Dashed line: y = x.')
            st.altair_chart(chart.properties(height=420,title='Held-out ICP predictions vs measurements'),width='stretch')
            st.caption(f'{len(frame)} held-out sample-element pairs from {frame.sample_id.nunique()} specimens. Each point is one measured element in one recipe. Purple: descriptive OLS fit. Dashed: perfect agreement (y = x).')
            if np.ptp(frame.predicted_w_wt_percent)<=1e-12:
                st.info('An OLS line cannot be estimated because the predicted ICP values have no variation.')
        else:
            st.info('No paired numeric ICP predictions and measurements are available in this validation result. Direct PASS/FAIL judgments alone cannot produce a continuous ICP regression plot.')
    if len(points):
        from .ui_prediction_agreement import render
        with st.expander('Additional agreement and XRD/STEM calibration diagnostics',expanded=False):
            render(samples,points)
    else:st.info('No evaluable predictions. Check the selected data and training requirements.')
    st.download_button('Download prediction pairs',points.to_csv(index=False).encode(),f'{origin.lower()}_sac_gen_predictions.csv','text/csv',key=f'weighted_points_{origin}')
    st.download_button('Download sample agreement scores',samples.to_csv(index=False).encode(),f'{origin.lower()}_sac_gen_agreement.csv','text/csv',key=f'weighted_samples_{origin}')
    with st.expander('Methods and evaluation details'):
        st.latex(r'S_W=\frac{100}{1+\langle[(\widehat W-W)/s_W]^2\rangle},\quad S_X=100[1-(p_X-y_X)^2],\quad S=w_WS_W+(1-w_W)S_X')
        st.caption('Outer test batches are excluded from training. XRD/STEM uses inner-fold ICP predictions. Correct PASS and FAIL predictions are rewarded equally.')
        st.dataframe(audit,hide_index=True,width='stretch')
        st.download_button('Download evaluation details',json.dumps(info,indent=2,allow_nan=False).encode(),f'{origin.lower()}_sac_gen_methods.json','application/json',key=f'weighted_json_{origin}')
