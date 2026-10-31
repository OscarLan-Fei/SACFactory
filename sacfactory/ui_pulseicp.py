"""Migrated ICP engine, auditable eligibility, protected validation and forecast UI."""
import json
import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from . import pulseicp_bridge as bridge
from .composition import process_key
from .planning import fingerprint
from .ui_elements import element_multiselect


def render_result(result, origin):
    """One method per default plot, with explicit held-out coverage and provenance."""
    destination, points, comparison, info = result
    rotating = info.get('split_protocol') == bridge.ROTATING_CV
    if rotating:
        columns = st.columns(3)
        columns[0].metric('Recorded specimens in selection', info['recorded_n'])
        columns[1].metric('Eligible quantified specimens', info['eligible_n'])
        columns[2].metric('Out-of-fold predicted specimens', info['evaluated_specimens_n'])
        if info['coverage_fraction'] < 1:
            st.warning('Some eligible specimens have no evaluable training fold. Inspect skipped folds and the coverage audit.')
        st.caption(f"{info['evaluated_target_rows']} unique sample-element predictions; {info['coverage_fraction']:.1%} coverage of eligible targets. Missing or incompatible ICP remains in the audit. Global metrics are calculated from all pooled held-out predictions, rather than averaging fold R-squared values.")
    st.dataframe(comparison, hide_index=True, width='stretch')
    if points.empty:
        st.info('No evaluable predictions. Inspect training-size requirements and skipped folds below.')
    else:
        names = list(dict.fromkeys(points.model.tolist()))
        if len(names) > 1:
            selected = st.selectbox('Prediction method shown', names,
                index=names.index('Gaussian residual') if 'Gaussian residual' in names else 0,
                key=f'picp_plot_method_{origin}')
            compare = st.checkbox('Compare all methods in the plot', value=False, key=f'picp_compare_{origin}')
            shown = points if compare else points.loc[points.model == selected]
        else:
            shown, compare = points, False
            st.caption('Prediction algorithm: ' + str(names[0]))
        limit = max(.01, float(shown[['observed_w_wt_percent', 'predicted_mean_w_wt_percent']].max().max())*1.05)
        tooltip = ['sample_id:N','batch_id:N','element:N','model:N',
            'predicted_mean_w_wt_percent:Q','observed_w_wt_percent:Q','calibration_status:N']
        if 'fold' in shown:
            tooltip += ['fold:O','training_specimens:Q']
        dots = alt.Chart(shown).mark_circle(size=70, opacity=.7, color='#168c8b').encode(
            x=alt.X('predicted_mean_w_wt_percent:Q', title='Predicted ICP W / wt%', scale=alt.Scale(domain=[0,limit])),
            y=alt.Y('observed_w_wt_percent:Q', title='Measured ICP W / wt%', scale=alt.Scale(domain=[0,limit])),
            tooltip=tooltip)
        if compare:
            dots = dots.encode(color='model:N')
        elif shown.element.nunique() > 1:
            dots = dots.encode(color='element:N')
        line = alt.Chart(pd.DataFrame({'value':[0,limit]})).mark_line(strokeDash=[4,4],color='#8795a5').encode(x='value:Q', y='value:Q')
        st.altair_chart((dots+line).properties(height=370), width='stretch')
        st.caption(f'{len(shown)} plotted sample-element rows; {shown.sample_id.nunique()} physical specimens. The dashed line is y = x. Identical points may overlap; all sample identities remain in the table.')
        st.dataframe(shown, hide_index=True, width='stretch')
        st.download_button('Download protected ICP predictions', points.to_csv(index=False).encode(), f'{origin.lower()}_sac_gen_predictions.csv', key=f'picp_points_{origin}')
    if rotating:
        st.caption('Evaluate loading predictions using protected batch splits; physical reference parameters are available below.')
        st.write('**Batch rotation and fold results**')
        st.dataframe(pd.DataFrame(info['fold_metrics']), hide_index=True, width='stretch')
        audit_path = Path(destination)/'eligibility.csv'
        if audit_path.exists():
            coverage = pd.read_csv(audit_path, keep_default_na=False, dtype={'sample_id':str, 'element':str, 'batch_id':str})
            with st.expander('Every recorded target: prediction coverage and exclusions'):
                st.dataframe(coverage, hide_index=True, width='stretch')
                st.download_button('Download ICP cross-validation coverage', coverage.to_csv(index=False).encode(),
                    f'{origin.lower()}_sac_gen_cv_coverage.csv', key=f'picp_cv_audit_{origin}')
    else:
        pass
    st.download_button('Download ICP validation provenance', json.dumps(info,indent=2,allow_nan=False).encode(),
        f'{origin.lower()}_sac_gen_validation.json', key=f'picp_provenance_{origin}')


def render_pulseicp(samples, origin, root):
    st.write('**SAC-Gen: mass-conserving Gaussian and conditional diffusion ensembles**')
    st.caption('Mass-conserving Gaussian and diffusion models predict elemental loading in wt%.')
    contexts = list(dict.fromkeys(process_key(s) for s in samples))
    if not contexts:
        st.info('Add experimental records to begin.')
        return
    from .ui_calculation_batch import select_targets
    context_indices=([0] if len(contexts)==1 else select_targets(
        'Process contexts',list(range(len(contexts))),
        format_func=lambda i:' / '.join(str(part) for part in contexts[i]),
        key=f'picp_contexts_{origin}',default_all=True))
    if not context_indices:
        st.info('Select one or more process contexts. Each context receives a separate protected fit.')
        return
    chosen_contexts=[contexts[i] for i in context_indices]
    context=chosen_contexts[0]
    selected=[s for s in samples if process_key(s) in chosen_contexts]
    config = bridge.configuration()
    settings, physics = config['model'], config['physics']
    with st.expander('Ensemble fitting budget'):
        cols = st.columns(3)
        settings['members'] = cols[0].number_input('ICP ensemble members', 1, 5, int(settings['members']), key=f'picp_members_{origin}')
        settings['epochs_baseline'] = cols[1].number_input('Gaussian training epochs', 1, 2000, int(settings['epochs_baseline']), key=f'picp_gepoch_{origin}')
        settings['epochs_diffusion'] = cols[2].number_input('Diffusion training epochs', 1, 2000, int(settings['epochs_diffusion']), key=f'picp_depoch_{origin}')
    data = bridge.prepare(selected, origin, settings, physics)
    st.caption(f"{len({r['raw']['sample_id'] for r in data.records})} eligible specimens; {len(data.records)} element targets. Targets outside the declared feed/matrix bound are excluded with reasons; recorded ICP is preserved. Zero, missing and censored results are not fabricated.")
    with st.expander('SAC-Gen eligibility audit', expanded=not bool(data.records)):
        st.dataframe(data.audit, hide_index=True, width='stretch')
        st.download_button('Download SAC-Gen eligibility', data.audit.to_csv(index=False).encode(), f'{origin.lower()}_sac_gen_eligibility.csv', key=f'picp_audit_{origin}')
    migration_key = f'picp_rotation_default_v1_{origin}'
    if not st.session_state.get(migration_key):
        st.session_state.pop(f'picp_protocol_{origin}', None)
        st.session_state[migration_key] = True
    protocol = st.selectbox('ICP engine protected split', bridge.PROTOCOLS,
        index=bridge.PROTOCOLS.index(bridge.ROTATING_CV), key=f'picp_protocol_{origin}')
    rotating = protocol == bridge.ROTATING_CV
    family, batches_per_fold = bridge.FAMILIES[0], 1
    if rotating:
        family = st.selectbox('ICP cross-validation algorithm', bridge.FAMILIES, key=f'picp_cv_family_{origin}')
        batch_count = len({str(r['raw']['batch_id']) for r in data.records})
        if batch_count >= 3:
            batches_per_fold = st.number_input('Batches held out per fold', min_value=1, max_value=batch_count-2,
                value=min(batch_count-2, max(1, math.ceil(batch_count/5))), step=1, key=f'picp_cv_batches_{origin}')
            st.caption(f'{batch_count} eligible batches, {math.ceil(batch_count/batches_per_fold)} disjoint test groups. Each group is predicted using the remaining batches, so each eligible target is tested once. Every fit needs sixteen training specimens and at least two training batches. Select the algorithm and grouping before final evaluation.')
        else:
            st.info('Rotating validation needs at least three actual batches. No outcomes or unknown batch names are imputed.')
    held = None
    held_elements=[]
    if protocol == 'Unseen element transfer':
        members=list(dict.fromkeys(e for s in selected for e in s['elements']))
        held_elements=element_multiselect('Entirely held-out ICP elements',members,
            default=members[:1],key=f'picp_held_elements_{origin}')
        held=held_elements[0] if held_elements else None
    if not rotating:
        pass
    multiple=len(chosen_contexts)>1 or len(held_elements)>1
    signature = fingerprint([selected,settings,physics,protocol,held_elements,family,int(batches_per_fold)])
    output = Path(root)/'pulseicp'/origin.lower()
    if st.button('Run rotating batch ICP validation' if rotating else 'Validate Gaussian and diffusion ICP models',
                 key=f'picp_run_{origin}', disabled=rotating and batch_count < 3 or protocol=='Unseen element transfer' and not held_elements):
        try:
            with st.spinner('Fitting models inside protected training groups...'):
                if multiple:
                    results=[]
                    for index,ctx in enumerate(chosen_contexts,1):
                        local=[s for s in selected if process_key(s)==ctx]
                        for target in held_elements if held_elements else [None]:
                            label=f'Context {index}: {ctx}'+(f' / held out {target}' if target else '')
                            try:
                                local_count=len({r['raw']['batch_id'] for r in bridge.prepare(local,origin,settings,physics).records})
                                local_group=min(int(batches_per_fold),max(1,local_count-2))
                                result=bridge.validate(local,origin,settings,physics,output/fingerprint([ctx,target]),
                                    protocol,target,all_samples=samples,family=family,batches_per_fold=local_group)
                                results.append((label,result,None))
                            except (ValueError,TypeError,KeyError,RuntimeError) as exc:
                                results.append((label,None,str(exc)))
                    st.session_state[f'picp_batch_result_{origin}']=(signature,results)
                else:
                    result=_run_single_validation(selected,samples,origin,settings,physics,output,protocol,held,
                        family,batches_per_fold,rotating)
                    st.session_state[f'picp_result_{origin}']=(signature,result)
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            st.error(str(exc))
    saved = st.session_state.get(f'picp_result_{origin}')
    if rotating and not multiple and (not saved or saved[0] != signature):
        from .pulseicp_cross_validation import load_matching_report
        try:
            result = load_matching_report(output, bridge.input_frame(selected,origin,physics), bridge.ORIGINS[origin],
                physics, settings, family, int(batches_per_fold), list(context))
            if result is not None:
                saved = (signature, result)
                st.session_state[f'picp_result_{origin}'] = saved
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if not multiple and saved and saved[0] == signature:
        render_result(saved[1], origin)
    batch_saved=st.session_state.get(f'picp_batch_result_{origin}')
    if multiple and batch_saved and batch_saved[0]==signature:
        st.caption('Every selected context/held-out-element combination was attempted independently. Small contexts may use fewer test batches per fold; each report records its actual grouping. Unavailable contexts remain listed.')
        for label,result,error in batch_saved[1]:
            with st.expander(label,expanded=True):
                if error:st.info(error)
                else:render_result(result,origin+'_'+fingerprint(label))
    with st.expander('Capture, aggregation, redispersion and escape: optional physical diagnostic'):
        ids=select_targets('Structural-prior samples',[s['sample_id'] for s in selected],str,
            key=f'picp_prior_samples_{origin}')
        st.caption('This XRD/STEM prior is a physical assumption; it is separate from the trained probability model.')
        if st.button('Calculate structural prior diagnostic',disabled=not ids,key=f'picp_prior_run_{origin}'):
            try:
                from ._pulseicp.screening_prior import predict_xrd_prior
                by_id={s['sample_id']:s for s in selected}
                result=predict_xrd_prior([bridge.composition(by_id[i],physics) for i in ids],physics,config['screening_prior'])
                if len(result)==len(ids):result.insert(0,'sample_id',ids)
                st.dataframe(result,hide_index=True,width='stretch')
                st.download_button('Download selected structural-prior calculations',result.to_csv(index=False).encode(),
                    f'{origin.lower()}_structural_priors.csv','text/csv',key=f'picp_prior_download_{origin}')
            except (ValueError, TypeError, KeyError) as exc:
                st.error(str(exc))


def _run_single_validation(selected,samples,origin,settings,physics,output,protocol,held,
                           family,batches_per_fold,rotating):
    """Retained single-context report and checkpoint contracts."""
    bar=st.progress(0.) if rotating else None
    try:
        return bridge.validate(selected,origin,settings,physics,output,protocol,held,all_samples=samples,
            family=family,batches_per_fold=int(batches_per_fold),
            progress=(lambda done,total:bar.progress(done/total,text=f'Completed fold {done} / {total}')) if rotating else None)
    finally:
        if bar is not None:bar.empty()
