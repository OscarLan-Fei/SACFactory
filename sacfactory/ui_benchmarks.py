"""Publication-facing predictive and acquisition comparisons."""
import json
import altair as alt
import pandas as pd
import streamlit as st
from .ui_elements import element_multiselect
from .model_benchmarks import compare,PUBLIC_MODELS as MODELS
from .acquisition_benchmarks import replay,POLICIES
from .planning import fingerprint
from .compute_runtime import execution,hardware
from . import pulseicp_bridge as bridge


def calculate(payload,progress=None):
    """Keep progress updates out of cached Streamlit message replay."""
    args=json.loads(payload);task=args.pop('task');runtime=args.pop('runtime')
    with execution(runtime,progress):return compare(**args) if task=='models' else replay(**args)


def render_benchmarks(samples,methods,origin):
    from .ui_navigation import preserve_analysis_controls
    preserve_analysis_controls(('benchmark_', 'comparison_'))
    st.write('**Model and acquisition benchmarks**')
    observed=list(dict.fromkeys(e for s in samples for e in s['elements']))
    elements=element_multiselect('Benchmark elements',observed,default=observed,key=f'benchmark_elements_{origin}')
    chosen=[s for s in samples if set(s['elements'])&set(elements)]
    st.caption(f'{len(chosen)} experiments. All selected methods use the same data and evaluation split.')
    predictive,acquisition=st.tabs(['Predictive models','Acquisition strategies'], key=f'benchmark_views_{origin}', on_change='rerun')
    machine=hardware()
    with st.expander('Training configuration'):
        device=st.selectbox('Benchmark device',['auto','cuda','cpu'],key=f'benchmark_device_{origin}')
        family='Diffusion Transformer'
        cols=st.columns(3)
        epochs=cols[0].number_input('Diffusion training epochs',1,2000,100,key=f'benchmark_epochs_{origin}')
        members=cols[1].number_input('Diffusion ensemble members',1,5,1,key=f'benchmark_members_{origin}')
        seed=cols[2].number_input('Benchmark random seed',0,1000000,42,key=f'benchmark_seed_{origin}')
        st.caption(f"{machine['gpu_name'] or 'CPU'}; fixed baseline settings are recorded with the results.")
    settings={**bridge.configuration()['model'],'members':int(members),'epochs_baseline':int(epochs),'epochs_diffusion':int(epochs),
        'draws_per_member':32,'diffusion_steps':30,'token_width':32,'attention_heads':4,'attention_layers':2}
    common={'samples':chosen,'methods':methods,'origin':origin,'sac_family':family,'sac_settings':settings,
            'seed':int(seed),'runtime':{'device':device,'threads':min(4,machine['cpu_count']),'cpu_jobs':min(8,machine['cpu_count'])}}
    if predictive.open:
        with predictive:
            if f'benchmark_models_{origin}' in st.session_state:
                st.session_state[f'benchmark_models_{origin}']=[m for m in st.session_state[f'benchmark_models_{origin}'] if m in MODELS]
            models=st.multiselect('Comparison models',MODELS,default=MODELS,key=f'benchmark_models_{origin}')
            batch_count=len({s.get('batch_id') for s in chosen})
            if f'benchmark_groups_{origin}' in st.session_state:
                st.session_state[f'benchmark_groups_{origin}']=min(max(1,batch_count-2),max(1,int(st.session_state[f'benchmark_groups_{origin}'])))
            groups=st.number_input('Test batches per fold',1,max(1,batch_count-2),(None if f'benchmark_groups_{origin}' in st.session_state else max(1,min(max(1,batch_count-2),(batch_count+4)//5))),key=f'benchmark_groups_{origin}')
            st.caption('Compare ICP MAE/RMSE/R² and XRD/STEM Brier/ROC-AUC on common held-out rows. Bayesian optimization is compared separately as an acquisition strategy.')
            payload=json.dumps({**common,'task':'models','targets':elements,'models':models,'batches_per_fold':int(groups)},sort_keys=True)
            _run(payload,'models',origin,disabled=not models or not chosen)
            saved=st.session_state.get(f'benchmark_models_result_{origin}')
            if saved and saved[0]==fingerprint(payload):
                points,summary,info=saved[1]
                st.dataframe(summary,hide_index=True,width='stretch')
                if len(summary):
                    shown=summary.melt(id_vars='model',value_vars=['ICP_MAE','XRD_Brier'],var_name='metric',value_name='error').dropna()
                    chart=alt.Chart(shown).mark_bar().encode(x=alt.X('error:Q',title='Error (lower is better)'),
                        y=alt.Y('model:N',axis=alt.Axis(labelLimit=220)),color=alt.Color('model:N',legend=None),tooltip=['model','metric','error']).properties(height=alt.Step(28)).facet(column='metric:N').resolve_scale(x='independent')
                    st.altair_chart(chart,width='stretch')
                from .ui_model_diagnostics import render_model_diagnostics
                render_model_diagnostics(points,origin,info.get('models'))
                _exports(points,summary,info,'models',origin)
    if acquisition.open:
        with acquisition:
            policies=st.multiselect('Acquisition policies',POLICIES,default=POLICIES,key=f'benchmark_policies_{origin}')
            cols=st.columns(3)
            initial=cols[0].number_input('Initial observations',4,10000,32,key=f'benchmark_initial_{origin}')
            budget=cols[1].number_input('Total observation budget',4,10000,48,key=f'benchmark_budget_{origin}')
            interval=cols[2].number_input('Queries per iteration',1,32,4,key=f'benchmark_interval_{origin}')
            st.caption('Retrospective replay uses recorded recipes only. All policies share initial observations, budgets and untouched test batches; a common random forest evaluates learning. Full-data SAC-Gen weights are never used in replay.')
            st.caption('GP-BO targets XRD/STEM PASS discovery; SAC-Gen uncertainty targets boundary learning. Compare test error and PASS/FAIL discoveries as separate outcomes.')
            payload=json.dumps({**common,'task':'acquisition','policies':policies,'initial':int(initial),'budget':int(budget),'interval':int(interval)},sort_keys=True)
            _run(payload,'acquisition',origin,disabled=not policies or not chosen)
            saved=st.session_state.get(f'benchmark_acquisition_result_{origin}')
            if saved and saved[0]==fingerprint(payload):
                curves,queries,points,info=saved[1]
                st.dataframe(curves,hide_index=True,width='stretch')
                if len(curves):
                    chart=alt.Chart(curves).mark_line(point=True).encode(x=alt.X('observations:Q',title='Revealed experiments'),
                        y=alt.Y('XRD_Brier:Q',title='Held-out XRD/STEM Brier'),color='policy:N',tooltip=['policy','observations','XRD_Brier','ICP_MAE']).properties(height=330)
                    st.altair_chart(chart,width='stretch')
                _exports(points,curves,info,'acquisition',origin)
                st.download_button('Download query sequence',queries.to_csv(index=False).encode(),f'{origin.lower()}_query_sequence.csv','text/csv',key=f'benchmark_queries_{origin}')


def _run(payload,kind,origin,disabled=False):
    if st.button('Run model comparison' if kind=='models' else 'Run acquisition replay',disabled=disabled,key=f'benchmark_run_{kind}_{origin}'):
        bar=st.progress(0.,text='Preparing comparison')
        def progress(event):
            if 'fold' in event:bar.progress(min(1.,event['fold']/max(1,event['folds'])),text=event.get('stage','Training'))
        try:
            signature=fingerprint(payload)
            previous=st.session_state.get(f'benchmark_{kind}_result_{origin}')
            if not previous or previous[0]!=signature:
                result=calculate(payload,progress)
                st.session_state[f'benchmark_{kind}_result_{origin}']=(signature,result)
            bar.progress(1.,text='Comparison complete')
        except (ValueError,RuntimeError) as exc:bar.empty();st.error(str(exc))


def _exports(points,summary,info,kind,origin):
    st.download_button('Download comparison table',summary.to_csv(index=False).encode(),f'{origin.lower()}_{kind}_comparison.csv','text/csv',key=f'benchmark_summary_{kind}_{origin}')
    st.download_button('Download test predictions',points.to_csv(index=False).encode(),f'{origin.lower()}_{kind}_predictions.csv','text/csv',key=f'benchmark_predictions_{kind}_{origin}')
    st.download_button('Download methods and splits',json.dumps(info,indent=2,allow_nan=False).encode(),f'{origin.lower()}_{kind}_methods.json','application/json',key=f'benchmark_info_{kind}_{origin}')
