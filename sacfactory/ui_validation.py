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
from .model_benchmarks import MODELS


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
            scale=parameters['delta'],agreement_weight=parameters['weight'])
        return {'kind':'multi_model_validation','points':points,'summary':summary,'info':info}


def render_weighted_validation(samples,methods,origin,root=None):
    st.write('**Weighted validation**')
    observed=list(dict.fromkeys(e for s in samples if s.get('origin')==origin for e in s['elements']))
    st.session_state.setdefault(f'weighted_elements_{origin}',observed)
    elements=element_multiselect('Validation elements',ELEMENTS,key=f'weighted_elements_{origin}')
    if st.button('Select all validation models',key=f'weighted_models_all_{origin}'):
        st.session_state[f'weighted_models_{origin}']=['SAC-Gen',*[model for model in MODELS if model!='SAC-Gen']]
    models=st.multiselect('Validation models',['SAC-Gen',*[model for model in MODELS if model!='SAC-Gen']],
        default=None if f'weighted_models_{origin}' in st.session_state else ['SAC-Gen'],key=f'weighted_models_{origin}')
    st.caption('Selected models share the same outer validation masks and outcome-score formula. Conventional estimators use CPU workers.')
    deployment_settings(root,origin)
    cols=st.columns(2);families=['Gaussian residual','Conditional diffusion','Transformer residual','Diffusion Transformer']
    if st.session_state.get(f'weighted_icp_family_{origin}') not in families:st.session_state[f'weighted_icp_family_{origin}']=families[0]
    family=cols[0].selectbox('SAC-Gen architecture',families,key=f'weighted_icp_family_{origin}',disabled='SAC-Gen' not in models)
    protocol=cols[1].selectbox('Validation protocol',PROTOCOLS,key=f'weighted_protocol_{origin}')
    maximum=max(1,len({s.get('batch_id') for s in samples})-2)
    if f'weighted_cv_batches_{origin}' in st.session_state:
        st.session_state[f'weighted_cv_batches_{origin}']=min(maximum,max(1,int(st.session_state[f'weighted_cv_batches_{origin}'])))
    groups=st.number_input('Test batches per fold',1,maximum,(None if f'weighted_cv_batches_{origin}' in st.session_state else max(1,min(maximum,(maximum+6)//5))),key=f'weighted_cv_batches_{origin}')
    with st.expander('Training and scoring settings'):
        machine=hardware();device=st.selectbox('Compute device',['auto','cuda','cpu'],key=f'weighted_device_{origin}')
        cols=st.columns(3)
        members=cols[0].number_input('Ensemble members',1,5,(None if f'weighted_aux_members_{origin}' in st.session_state else 1),key=f'weighted_aux_members_{origin}')
        epochs=cols[1].number_input('Training epochs',1,2000,(None if f'weighted_aux_epochs_{origin}' in st.session_state else 100),key=f'weighted_aux_epochs_{origin}')
        draws=cols[2].number_input('Draws per member',2,512,(None if f'weighted_aux_draws_{origin}' in st.session_state else 32),key=f'weighted_aux_draws_{origin}')
        steps=st.number_input('Diffusion steps',10,200,(None if f'weighted_aux_steps_{origin}' in st.session_state else 30),key=f'weighted_aux_steps_{origin}',disabled='diffusion' not in family.lower())
        cols=st.columns(2)
        threads=cols[0].number_input('CPU training threads',1,machine['cpu_count'],(None if f'weighted_threads_{origin}' in st.session_state else min(4,machine['cpu_count'])),key=f'weighted_threads_{origin}')
        workers=cols[1].number_input('CPU workers',1,machine['cpu_count'],(None if f'weighted_jobs_{origin}' in st.session_state else min(8,machine['cpu_count'])),key=f'weighted_jobs_{origin}')
        cols=st.columns(2)
        weight=cols[0].number_input('ICP agreement weight',.05,.95,(None if f'weighted_weight_{origin}' in st.session_state else .6),step=.05,key=f'weighted_weight_{origin}')
        scale=cols[1].number_input('ICP error scale / wt% points',.0001,100.,(None if f'weighted_delta_{origin}' in st.session_state else .03 if origin=='REAL' else .25),format='%.4f',key=f'weighted_delta_{origin}')
        st.caption(f"XRD weight: {1-weight:.2f}. Agreement is an error score, not percent accuracy. {machine['gpu_name'] or 'CPU'}.")
    settings={**bridge.configuration()['model'],'members':int(members),'epochs_baseline':int(epochs),'epochs_diffusion':int(epochs),
        'draws_per_member':int(draws),'diffusion_steps':int(steps),'token_width':32,'attention_heads':4,'attention_layers':2}
    runtime={'device':device,'threads':int(threads),'cpu_jobs':int(workers),'runtime_version':'device-batched-v1'}
    payload=json.dumps({'samples':samples,'methods':methods,'origin':origin,'targets':elements,'protocol':protocol,
        'profile':'chemistry_pulseicp','batches_per_fold':int(groups),'icp_family':family,'icp_settings':settings,
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
    if root is not None and st.button('Train deployment model',key=f'sac_gen_deploy_{origin}',disabled='SAC-Gen' not in models):
        from .sac_gen import train_deployment
        try:
            with st.spinner('Training deployment model'),execution(runtime):saved=train_deployment(samples,origin,root,family,settings,methods)
            st.success(f"SAC-Gen model {saved['model_id']} is ready.")
        except (ValueError,RuntimeError) as exc:st.error(str(exc))
    saved=st.session_state.get(f'weighted_result_{origin}')
    if saved and saved[0]==signature:
        if isinstance(saved[1],dict):render_multiple_models(saved[1],origin)
        else:render_results(saved[1],origin)
    elif saved:st.info('Settings or data changed. Run validation to refresh the results.')


def render_multiple_models(result,origin):
    from .ui_model_diagnostics import render_outcome_score,render_model_diagnostics
    from .validation_diagnostics import common_rows
    points,summary,info=result['points'],result['summary'],result['info']
    if points.empty:
        st.info('No evaluable predictions for this selection. Check the training batches and protocol.')
    else:
        render_outcome_score(common_rows(points,info['models']),origin,f'validation_outcome_{origin}')
    with st.expander('ICP and XRD model diagnostics',expanded=False):
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
        meta=json.loads((base/name/'sac_gen.json').read_text());st.caption(f"Deployed model: {meta['family']} · {meta['model_id']}")
        key=f'sac_gen_deployed_defaults_{origin}'
        if st.button('Use deployment settings',key=f'sac_gen_use_deployed_{origin}') or not st.session_state.get(key):
            config=meta['settings'];st.session_state.update({f'weighted_icp_family_{origin}':meta['family'],
                f'weighted_aux_members_{origin}':int(config['members']),f'weighted_aux_epochs_{origin}':int(config['epochs_baseline']),
                f'weighted_aux_draws_{origin}':int(config['draws_per_member']),f'weighted_aux_steps_{origin}':int(config['diffusion_steps']),key:True})
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
        for col,label,value in zip(st.columns(5),['ICP MAE / wt%','ICP RMSE / wt%','ICP R²','XRD Brier','XRD ROC-AUC'],[mae,rmse,r2,xrd.get('brier'),xrd.get('roc_auc')]):col.metric(label,f'{value:.4f}' if value is not None else 'Unavailable')
        st.caption(f"{len(numeric)} ICP pairs · {xrd.get('evaluated_samples',0)} XRD pairs · Mean agreement: {info['sample_weighted_score']:.2f}/100" if info['sample_weighted_score'] is not None else 'No complete agreement pairs.')
        if len(numeric):
            frame=numeric.copy();frame['XRD']=frame.xrd_pass_observed.map({1:'PASS',0:'FAIL'}).fillna('Unmeasured')
            limit=max(.01,float(frame[['predicted_w_wt_percent','observed_w_wt_percent']].max().max())*1.06)
            values=alt.Data(values=frame.astype(object).where(pd.notna(frame),None).to_dict('records'))
            dots=alt.Chart(values).mark_point(filled=True,size=45,opacity=.65).encode(
                x=alt.X('predicted_w_wt_percent:Q',title='Predicted ICP W / wt%',scale=alt.Scale(domain=[0,limit])),
                y=alt.Y('observed_w_wt_percent:Q',title='Measured ICP W / wt%',scale=alt.Scale(domain=[0,limit])),
                color=alt.Color('XRD:N',scale=alt.Scale(domain=['PASS','FAIL','Unmeasured'],range=['#268d88','#d46a27','#71808c'])),
                tooltip=['sample_id:N','element:N','batch_id:N','XRD:N','predicted_w_wt_percent:Q','observed_w_wt_percent:Q'])
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
        with st.expander('Additional agreement and XRD calibration diagnostics',expanded=False):
            render(samples,points)
    else:st.info('No evaluable predictions. Check the selected data and training requirements.')
    st.download_button('Download prediction pairs',points.to_csv(index=False).encode(),f'{origin.lower()}_sac_gen_predictions.csv','text/csv',key=f'weighted_points_{origin}')
    st.download_button('Download sample agreement scores',samples.to_csv(index=False).encode(),f'{origin.lower()}_sac_gen_agreement.csv','text/csv',key=f'weighted_samples_{origin}')
    with st.expander('Methods and evaluation details'):
        st.latex(r'S_W=\frac{100}{1+\langle[(\widehat W-W)/s_W]^2\rangle},\quad S_X=100[1-(p_X-y_X)^2],\quad S=w_WS_W+(1-w_W)S_X')
        st.caption('Outer test batches are excluded from training. XRD uses inner-fold ICP predictions. Correct PASS and FAIL predictions are rewarded equally.')
        st.dataframe(audit,hide_index=True,width='stretch')
        st.download_button('Download evaluation details',json.dumps(info,indent=2,allow_nan=False).encode(),f'{origin.lower()}_sac_gen_methods.json','application/json',key=f'weighted_json_{origin}')
