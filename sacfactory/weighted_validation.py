"""Protected ICP/screening prediction losses, separate from synthesis quality."""
from collections import Counter
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .chemical_descriptors import descriptor_matrix, model_features, reference
from .composition import process_key, transfer_feature
from .domain import ELEMENTS, INVALID_RUN_CONDITIONS, parse_measurement, recipe_key, screen
from .manual import actual_batch_recorded
from .planning import fingerprint
from .precursors import documented, state_reference
from .units import normalize_assay
from .physical_priors import thermal_properties, thermal_reference
from .outcome_validation import (BINARY_ENDPOINTS, outcome_labels, normalized_weights,
    fit_outcomes, binary_metrics, score_samples)
from .validation_groups import RANDOM_RECIPE_CV, recipe_folds, assert_mask

VERSION = 'weighted-validation-v5-sac-gen'
ROTATING_CV = 'Rotating batch cross-validation'
PROTOCOLS = [RANDOM_RECIPE_CV, ROTATING_CV, 'Unseen element holdout', 'Batch holdout', 'New recipe holdout',
             'Chronological batch replay', 'Single to multiple elements']


def weighted_loss(w_pred, w_obs, p_pred, label, weight=.6, delta=.03):
    if not 0 <= weight <= 1 or not math.isfinite(delta) or delta <= 0:
        raise ValueError('Weight must be in [0, 1] and the wt% error scale positive.')
    return weight * ((np.asarray(w_pred)-np.asarray(w_obs))/delta)**2 + (1-weight)*(np.asarray(p_pred)-np.asarray(label))**2


def display_score(loss):
    return 100/(1+loss)


def phase_features(element,temperature):
    phase=thermal_properties(element)
    result=[]
    for field in ('melting_point','boiling_point'):
        value=phase[field]
        flag=phase[field+'_sublimation']
        result.extend([value/5000 if value is not None else 0.,float(value is None),
            (temperature+273.15)/value if value else 0.,float(flag) if flag is not None else 0.,float(flag is None)])
    return np.asarray(result)


def recipe_features(sample, profile, use_thermal=True):
    """Fixed, pre-synthesis controls plus sourced composition moments; no assays."""
    if sample.get('pulse_limit_C') is None or not np.isfinite(float(sample['pulse_limit_C'])):
        raise ValueError('Controlled cutoff missing')
    raw = transfer_feature(sample, profile, for_training=True)
    enriched = model_features([raw], profile)[0]
    dimension = len(ELEMENTS)
    # Identity/log member arrays are replaced by invariant composition moments.
    # Declared-precursor features and missing masks remain in the raw tail.
    result = np.r_[raw[2*dimension:], enriched[len(raw):]]
    if use_thermal:
        weights=np.array([sample['concentrations_mM'][e] for e in sample['elements']]); weights=weights/weights.sum()
        phases=np.array([phase_features(e,sample['pulse_limit_C']) for e in sample['elements']])
        result=np.r_[result,weights@phases]
    if not np.isfinite(result).all():
        raise ValueError('Invalid controls')
    return result


def element_features(sample, element, recipe_x, profile, use_thermal=True):
    from .mixing import component_features
    return np.r_[recipe_x, component_features(sample, element) if profile in {'chemistry_mixing', 'chemistry_pulseicp'} else [], np.log(sample['concentrations_mM'][element]/.05),
                 descriptor_matrix(profile)[ELEMENTS.index(element)],
                 phase_features(element,sample['pulse_limit_C']) if use_thermal else []]


def collect(samples, methods, origin, profile, use_thermal=True):
    records, audit = [], []
    for s in samples:
        if s.get('origin') != origin:
            continue
        reason = ''
        try:
            if not actual_batch_recorded(s):
                raise ValueError('Actual batch missing')
            if any(not documented(s.get(k)) for k in ('precursor_family','support_batch','protocol_id')):
                raise ValueError('Process context missing')
            if s.get('sample_condition') in INVALID_RUN_CONDITIONS:
                raise ValueError('Invalid or aborted run')
            if float(s.get('pulse_count') or 0) < 1 or not float(s['pulse_count']).is_integer():
                raise ValueError('Positive integer pulse count required')
            x = recipe_features(s, profile, use_thermal)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            reason = str(exc)
        audit.append({'sample_id':s['sample_id'],'scope':'Controls','eligible':not reason,'reason':reason})
        if reason:
            continue
        outcome = screen(s, methods)
        label = outcome['label'] if not outcome['review_conflicts'] else None
        assays = {}
        for e in s['elements']:
            matches = [a for a in s.get('icp',[]) if a.get('element') == e]
            why = ''
            try:
                if s.get('sample_condition') == 'BURNED_OUT':
                    raise ValueError('No recovered sample; W undefined')
                if len(matches) != 1:
                    raise ValueError('One numeric ICP result required')
                a = normalize_assay(matches[0]); reading = parse_measurement(a.get('value',''))
                if a.get('flag','MEASURED') != 'MEASURED' or reading['kind'] != 'measured':
                    raise ValueError('Censored or missing ICP; no zero imputation')
                value = reading['value']
                if not 0 <= value <= 100:
                    raise ValueError('W outside 0 to 100 wt%')
                assays[e] = value
            except (ValueError, TypeError, KeyError) as exc:
                why = str(exc)
            audit.append({'sample_id':s['sample_id'],'scope':e,'eligible':not why,'reason':why})
        records.append({'sample':s,'x':x,'label':label,'outcomes':outcome_labels(s,outcome),
            'assays':assays,'recipe':recipe_key(s)})
    return records, pd.DataFrame(audit)


def split_records(records, all_samples, origin, targets, protocol, budget, batches_per_fold=1, seed=42):
    """Construct masks before accessing the withheld numerical/label outcomes."""
    partitions = {}
    for r in records:
        partitions.setdefault(process_key(r['sample']),[]).append(r)
    splits = []
    for key, rows in partitions.items():
        relevant = lambda r: bool(set(r['sample']['elements']) & set(targets))
        if protocol == RANDOM_RECIPE_CV:
            for fold, (source, held) in enumerate(recipe_folds([r['sample'] for r in rows], budget, seed), 1):
                train = [rows[i] for i in source]
                test = [rows[i] for i in held if relevant(rows[i])]
                if test:
                    splits.append((train, test, f'{fingerprint(key)}_RECIPE_OOF_{fold}', targets))
        elif protocol == ROTATING_CV:
            from .pulseicp_cross_validation import batch_groups
            batches = {str(r['sample']['batch_id']) for r in rows}
            if len(batches) < 3:
                continue
            # Keep the declared group size; insufficient contexts remain explicitly uncovered.
            if batches_per_fold > len(batches)-2:
                continue
            for fold, held in enumerate(batch_groups(batches,batches_per_fold,seed),1):
                train = [r for r in rows if str(r['sample']['batch_id']) not in held]
                test = [r for r in rows if str(r['sample']['batch_id']) in held and relevant(r)]
                if test:
                    splits.append((train,test,f'{fingerprint(key)}_OOF_{fold}',targets))
        elif protocol == 'Unseen element holdout':
            for e in targets:
                test = [r for r in rows if r['sample']['elements'] == [e]]
                blocked = {s['batch_id'] for s in all_samples if s.get('origin') == origin and e in s['elements']}
                train = [r for r in rows if len(r['sample']['elements']) == 1 and e not in r['sample']['elements'] and r['sample']['batch_id'] not in blocked]
                if test:
                    splits.append((train,test,f'UNSEEN_{e}',[e]))
        elif protocol == 'Single to multiple elements':
            test = [r for r in rows if len(r['sample']['elements']) > 1 and relevant(r)]
            blocked = {r['sample']['batch_id'] for r in test}
            train = [r for r in rows if len(r['sample']['elements']) == 1 and r['sample']['batch_id'] not in blocked]
            if test:
                splits.append((train,test,'SINGLE_TO_MULTI',targets))
        elif protocol == 'Chronological batch replay':
            times = {}
            for batch in {r['sample']['batch_id'] for r in rows}:
                dates = pd.to_datetime([s.get('result_reported_at') for s in all_samples if s.get('origin') == origin and s['batch_id']==batch],utc=True,errors='coerce')
                if len(dates) and not dates.isna().any():
                    times[batch] = (dates.min(),dates.max())
            batches = sorted({r['sample']['batch_id'] for r in rows if relevant(r) and r['sample']['batch_id'] in times}, key=lambda b:times[b][0])
            if len(batches)>budget:
                batches = [batches[i] for i in np.unique(np.linspace(0,len(batches)-1,budget,dtype=int))]
            for batch in batches:
                test = [r for r in rows if r['sample']['batch_id']==batch and relevant(r)]
                train = [r for r in rows if r['sample']['batch_id'] in times and times[r['sample']['batch_id']][1] < times[batch][0]]
                splits.append((train,test,'TIME_'+batch,targets))
        elif len({r['sample']['batch_id'] for r in rows}) >= 3:
            groups = [r['sample']['batch_id'] for r in rows]
            splitter = GroupKFold(n_splits=min(budget,len(set(groups))))
            for fold,(train_indices,test_indices) in enumerate(splitter.split(rows,groups=groups),1):
                test = [rows[i] for i in test_indices if relevant(rows[i])]
                blocked_recipes = {r['recipe'] for r in test} if protocol=='New recipe holdout' else set()
                train = [rows[i] for i in train_indices if rows[i]['recipe'] not in blocked_recipes]
                if test:
                    splits.append((train,test,f'{fingerprint(key)}_{fold}',targets))
    return splits


def aggregate(points):
    """Count the sample screening error once, regardless of selected metal count."""
    sample_rows = []
    for (fold,sample_id), part in points.groupby(['fold','sample_id']):
        first = part.iloc[0]
        complete = part.loc[part.endpoint_set == 'ICP + screening']
        icp = part.loc[part.icp_loss.notna()]
        loss = float(complete.validation_loss.mean()) if len(complete) else None
        sample_rows.append({'fold':fold,'sample_id':sample_id,'batch_id':first.batch_id,
            'composition':first.composition,'element_count':int(first.element_count),
            'quantified_selected_elements':len(icp),'icp_loss':float(icp.icp_loss.mean()) if len(icp) else None,
            'screening_loss':first.screening_loss,'validation_loss':loss,
            'validation_score':display_score(loss) if loss is not None else None,
            'baseline_loss':float(complete.baseline_loss.mean()) if len(complete) else None})
    return pd.DataFrame(sample_rows)


def weighted_validation(samples, methods, origin, targets, protocol='Unseen element holdout',
                        profile='chemistry_ion', weight=.6, delta=.03, max_folds=3, trees=100, use_thermal=True,
                        icp_family='Random forest', icp_settings=None, batches_per_fold=1, seed=42,
                        endpoint_weights=None, system_weights=None, system_target_wt_percent=2.5,
                        system_model='Direct random forest', system_prior_strength=1.,system_icp_settings=None,
                        agreement_method='component_agreement_v1',include_system_score=True,
                        system_score_definition='capped_attainment_v1',sac_gen=False):
    from . import pulseicp_bridge as bridge
    from . import system_score
    from .prediction_agreement import METHODS, COMPONENT
    from .token_residual import FAMILIES as TOKEN_FAMILIES
    if agreement_method not in METHODS:
        raise ValueError('Select a supported prediction-agreement definition.')
    if icp_family not in ['Random forest', *bridge.FAMILIES, *TOKEN_FAMILIES]:
        raise ValueError('Select a supported weighted-validation ICP family.')
    if sac_gen:
        if icp_family=='Random forest':raise ValueError('SAC-Gen requires a learned neural ICP family.')
        include_system_score=False
        endpoint_weights={'icp':.6,'xrd_pass':.4} if endpoint_weights is None else endpoint_weights
        if set(endpoint_weights)-{'icp','xrd_pass'}:
            raise ValueError('SAC-Gen accuracy uses continuous ICP and XRD probability only.')
    targets = list(dict.fromkeys(targets))
    if not targets or set(targets)-set(ELEMENTS) or protocol not in PROTOCOLS:
        raise ValueError('Select valid elements and a validation protocol.')
    weighted_loss(0,0,0,0,weight,delta)
    score_weights = normalized_weights(endpoint_weights,weight)
    result_weights,result_target=system_score.configuration(system_weights,system_target_wt_percent)
    system_model,system_prior_strength=system_score.prediction_settings(system_model,system_prior_strength)
    from .result_utility import DEFINITIONS
    if system_score_definition not in DEFINITIONS:
        raise ValueError('Select a supported synthesis-result score definition.')
    if isinstance(batches_per_fold,bool) or not isinstance(batches_per_fold,(int,np.integer)) or batches_per_fold < 1:
        raise ValueError('Batches per test fold must be a positive integer.')
    budget = max(2,min(8,int(max_folds)))
    records,audit = collect(samples,methods,origin,profile,use_thermal)
    splits = split_records(records,samples,origin,targets,protocol,budget,batches_per_fold,seed)
    points, provenance, skipped, system_rows = [], [], [], []
    from .compute_runtime import emit, jobs
    for fold_index,(train,test,fold,fold_targets) in enumerate(splits):
        emit({'stage':'Outer validation fold','fold':fold_index+1,'folds':len(splits)})
        train_batches = {r['sample']['batch_id'] for r in train}
        test_batches = {r['sample']['batch_id'] for r in test}
        assert_mask([r['sample'] for r in train], [r['sample'] for r in test], protocol == RANDOM_RECIPE_CV)
        if len(train)<8 or len(train_batches)<2 or len({r['recipe'] for r in train})<3:
            skipped.append({'fold':fold,'reason':'Need 8 training records, 2 batches and 3 recipes'})
            continue
        labeled = [r for r in train if r['label'] is not None]
        heads, head_info = fit_outcomes(train,test,trees,seed,exclude=('icp_pass','screening') if sac_gen else ())
        if sac_gen:
            # The requested workflow has no ICP pass classifier or quality utility.
            for endpoint in ('icp_pass','screening'):
                heads[endpoint]=([None]*len(test),None)
                head_info[endpoint].update(available=False,reason='Disabled in SAC-Gen: ICP is a continuous loading target, XRD is the screening endpoint.')
        result_rows,result_fit=system_score.fit_fold(train,test,result_weights,result_target,trees,seed,
            model=system_model,prior_strength=system_prior_strength,pulse_settings=system_icp_settings,
            score_definition=system_score_definition) if include_system_score else ([],{'available':False,'reason':'Optional synthesis utility analysis disabled'})
        system_rows.extend({**row,'fold':fold} for row in result_rows)
        quantitative = [(r,e,w) for r in train for e,w in r['assays'].items()]
        regressor = None
        if icp_family == 'Random forest' and len(quantitative)>=8 and len({r['sample']['batch_id'] for r,_,_ in quantitative})>=2:
            regressor = RandomForestRegressor(n_estimators=trees,min_samples_leaf=3,max_depth=16,random_state=42,n_jobs=jobs()).fit(
                [element_features(r['sample'],e,r['x'],profile,use_thermal) for r,e,_ in quantitative],[w for _,_,w in quantitative])
        neural_predictions, neural_info = {}, {}
        if sac_gen:
            from .sac_gen import SACGen
            try:
                settings = dict(icp_settings or bridge.configuration()['model'])
                settings['crossfit_grouping'] = 'recipe' if protocol == RANDOM_RECIPE_CV else 'batch'
                trained=SACGen(icp_family,settings,profile).fit([r['sample'] for r in train],origin,methods)
                forecasts=trained.predict([r['sample'] for r in test])
                probabilities=[f['p_xrd_pass'] for f in forecasts]
                heads['xrd_pass']=(probabilities,trained.xrd_mean)
                head_info['xrd_pass']={**trained.xrd_info,'training_n':trained.xrd_info['n'],
                    'classes':sorted({r['outcomes']['xrd_pass'] for r in train if r['outcomes']['xrd_pass'] is not None}),
                    'source_mean':trained.xrd_mean}
                for r,f in zip(test,forecasts):
                    for item in f['icp_predictions']:
                        neural_predictions[(r['sample']['sample_id'],item['element'])]=item['predicted_mean_w_wt_percent']
                neural_info.update(model_name='SAC-Gen',model_id=trained.model_id,settings=trained.settings,
                    training_ids=trained.training_ids,xrd=trained.xrd_info,
                    loading_model_id=trained.loading.model_id,calibration_status='TRAINED_UNCALIBRATED')
            except (ValueError,KeyError,TypeError) as exc:
                heads['xrd_pass']=([None]*len(test),None)
                head_info['xrd_pass'].update(available=False,reason=str(exc))
                neural_info['unavailable_reason']=str(exc)
        elif icp_family != 'Random forest':
            try:
                model, eligibility = bridge.fit([r['sample'] for r in train], origin, icp_family, icp_settings)
                from .compute_runtime import batched_predict
                def forecast(m,s,o):
                    return bridge.predict(m,s,o).to_dict('records')
                def error(sample,exc):
                    neural_info.setdefault('prediction_exclusions', []).append({'sample_id':sample['sample_id'],'reason':str(exc)})
                for item in batched_predict(model,[r['sample'] for r in test],origin,forecast,error):
                    neural_predictions[(item['sample_id'], item['element'])] = item['predicted_mean_w_wt_percent']
                neural_info.update(model_id=model.model_id, training_ids=model.training_ids,
                    physics=model.physics, settings=model.settings, calibration_status='PROVISIONAL_UNCALIBRATED',
                    eligibility=eligibility.astype(object).where(pd.notna(eligibility), None).to_dict('records'))
            except (ValueError, KeyError, TypeError) as exc:
                neural_info['unavailable_reason'] = str(exc)
        p_values = heads['screening'][0]
        baseline_p = float(np.mean([r['label'] for r in labeled])) if labeled else None
        baseline_w = float(np.mean([w for _,_,w in quantitative])) if quantitative else None
        provenance.append({'fold':fold,'train_sample_ids':[r['sample']['sample_id'] for r in train],
            'test_sample_ids':[r['sample']['sample_id'] for r in test],'train_batch_ids':sorted(train_batches),'test_batch_ids':sorted(test_batches),
            'train_elements':sorted({e for r in train for e in r['sample']['elements']}),
            'latest_train_result':max(str(r['sample'].get('result_reported_at') or '') for r in train),
            'first_test_result':min(str(r['sample'].get('result_reported_at') or '') for r in test),
            'quantitative_training_n':len(quantitative),'screening_training_n':len(labeled),
            'screening_model_available':head_info['screening']['available'],'outcome_heads':head_info,
            'loading_model_available':regressor is not None or bool(neural_predictions),
            'icp_family':icp_family, 'pulseicp':neural_info,'system_score_head':result_fit})
        for test_index,(r,p) in enumerate(zip(test,p_values)):
            s=r['sample']; label=r['label']
            observed_total=sum(r['assays'][e] for e in s['elements']) if all(e in r['assays'] for e in s['elements']) else None
            if regressor:
                members=regressor.predict([element_features(s,e,r['x'],profile,use_thermal) for e in s['elements']])
                predicted_total=float(np.sum(members))
            else:
                members=[neural_predictions.get((s['sample_id'],e)) for e in s['elements']]
                predicted_total=float(sum(members)) if all(v is not None for v in members) else None
            binary = {}
            for endpoint in BINARY_ENDPOINTS:
                probability = heads[endpoint][0][test_index]
                actual = r['outcomes'][endpoint]
                mean = heads[endpoint][1]
                paired = probability is not None and actual is not None
                binary.update({f'{endpoint}_predicted':float(probability) if probability is not None else None,
                    f'{endpoint}_observed':actual,
                    f'{endpoint}_loss':float((probability-actual)**2) if paired else None,
                    f'{endpoint}_baseline_loss':float((mean-actual)**2) if paired else None})
            binary['xrd_failure_probability']=1-binary['xrd_pass_predicted'] if binary['xrd_pass_predicted'] is not None else None
            binary['xrd_failure_observed']=1-binary['xrd_pass_observed'] if binary['xrd_pass_observed'] is not None else None
            for e in [e for e in fold_targets if e in s['elements']]:
                w=r['assays'].get(e)
                predicted = float(regressor.predict([element_features(s,e,r['x'],profile,use_thermal)])[0]) if regressor else None
                if icp_family != 'Random forest':
                    predicted = neural_predictions.get((s['sample_id'], e))
                icp_loss = ((predicted-w)/delta)**2 if predicted is not None and w is not None else None
                brier = float((p-label)**2) if p is not None and label is not None else None
                full = icp_loss is not None and brier is not None
                loss = weight*icp_loss+(1-weight)*brier if full else None
                baseline = float(weight*((baseline_w-w)/delta)**2+(1-weight)*(baseline_p-label)**2) if full else None
                points.append({'sample_id':s['sample_id'],'batch_id':s['batch_id'],'element':e,'composition':'+'.join(s['elements']),
                    'element_count':len(s['elements']),'fold':fold,'observed_w_wt_percent':w,'predicted_w_wt_percent':predicted,
                    'observed_total_w_wt_percent':observed_total,'predicted_total_w_wt_percent':predicted_total,
                    'label':label,'p_screen_pass':float(p) if p is not None else None,'icp_loss':icp_loss,'screening_loss':brier,
                    'icp_baseline_loss':float(((baseline_w-w)/delta)**2) if icp_loss is not None else None, **binary,
                    'validation_loss':loss,'validation_score':display_score(loss) if full else None,'baseline_loss':baseline,
                    'endpoint_set':'ICP + screening' if full else 'ICP only' if icp_loss is not None else 'Screening only' if brier is not None else 'Unavailable',
                    'prediction_kind':'RECIPE_OUT_OF_FOLD' if protocol==RANDOM_RECIPE_CV else 'BATCH_OUT_OF_FOLD' if protocol==ROTATING_CV else 'RETROSPECTIVE_HELD_OUT',
                    'target_seen_before':e in {e2 for row in train for e2 in row['sample']['elements']}})
    frame=pd.DataFrame(points)
    sample_scores=score_samples(frame,score_weights,agreement_method) if len(frame) else pd.DataFrame()
    if len(sample_scores) and system_rows:
        sample_scores=sample_scores.merge(pd.DataFrame(system_rows),on=['fold','sample_id','batch_id','composition'],validate='one_to_one')
    if len(frame):
        scores = sample_scores.set_index(['fold','sample_id'])
        for index,row in frame.iterrows():
            result = scores.loc[(row['fold'],row['sample_id'])]
            for column in ('validation_loss','validation_score','baseline_loss','baseline_agreement_score'):
                frame.at[index,column] = result[column]
            frame.at[index,'endpoint_set'] = ' + '.join(score_weights) if pd.notna(result.validation_loss) else 'Incomplete configured outcomes'
    report=[]
    for e in targets:
        part=frame.loc[frame.element==e] if len(frame) else frame
        measured=part.dropna(subset=['predicted_w_wt_percent','observed_w_wt_percent']) if len(part) else part
        full=part.loc[part.validation_loss.notna()] if len(part) else part
        report.append({'element':e,'evaluated_points':len(part),'quantified_pairs':len(measured),'complete_score_pairs':len(full),
            'mae_wt_percent':float(mean_absolute_error(measured.observed_w_wt_percent,measured.predicted_w_wt_percent)) if len(measured) else None,
            'rmse_wt_percent':float(np.sqrt(mean_squared_error(measured.observed_w_wt_percent,measured.predicted_w_wt_percent))) if len(measured) else None,
            'r_squared':float(r2_score(measured.observed_w_wt_percent,measured.predicted_w_wt_percent)) if len(measured)>1 and np.ptp(measured.observed_w_wt_percent)>1e-12 else None,
            'brier':float(part.screening_loss.mean()) if len(part) and part.screening_loss.notna().any() else None,
            'score':float(full.validation_score.mean()) if len(full) and agreement_method==COMPONENT else float(display_score(full.validation_loss.mean())) if len(full) else None,
            'baseline_score':float(full.baseline_agreement_score.mean()) if len(full) and agreement_method==COMPONENT else float(display_score(full.baseline_loss.mean())) if len(full) else None,
            'status':'Evaluated' if len(part) else 'No eligible target fold or sufficient source data'})
    scored=sample_scores.dropna(subset=['validation_loss']) if len(sample_scores) else sample_scores
    by_element=pd.DataFrame(report)
    info={'version':VERSION,'origin':origin,'protocol':protocol,'selected_elements':targets,'profile':profile,'weight_icp':weight,
        'icp_family':icp_family,'model_name':'SAC-Gen' if sac_gen else 'Independent benchmark heads', 'pulseicp_snapshot':bridge.identity() if icp_family != 'Random forest' or profile == 'chemistry_pulseicp' or system_model in system_score.PULSE_MODELS else None,
        'delta_wt_percent':delta,'trees':trees,'fold_budget':None if protocol==ROTATING_CV else budget,
        'batches_per_test_fold':int(batches_per_fold),'split_seed':int(seed),'endpoint_weights':score_weights,
        'outcome_metrics':binary_metrics(frame),'n_points':len(frame),'n_scored_samples':len(scored),
        'agreement_method':agreement_method,'agreement_definition':METHODS[agreement_method],
        'agreement_aggregation':'Mean of complete per-sample component agreement scores' if agreement_method==COMPONENT else 'Inverse of mean weighted error',
        'system_score':{'enabled':bool(include_system_score),'score_definition':system_score_definition,
            'log_target_width':float(np.log(2.)) if system_score_definition=='continuous_target_match_v1' else None,
            'failure_loading_discount':.25 if system_score_definition=='continuous_target_match_v1' else None,
            'version':system_score.VERSION,'weights':result_weights,'total_target_wt_percent':result_target,
            'prediction_model':system_model,'prior_strength':system_prior_strength,
            'pulseicp_settings':system_icp_settings,
            'metrics':system_score.summary(system_rows),'note':'Prespecified operational synthesis-result utility, not a structural proof or validated universal quality scale. A direct, physical-prior residual or cross-fitted learned PulseICP residual head predicts the unchanged experimental score. Learned loading baselines for score training are generated by inner whole-batch masks; outer test loading uses an outer-training-only fit. The selected loading utility is averaged over aligned whole-recipe draws. Prior or learned loading baselines never enter observed scores. Explicit complete burnout defines zero utility without imputing W or assay labels. Missing components retain score bounds; no automatic weight renormalization. All recipe members enter total W, including members not selected as plot targets. Test outcomes never train the score head. Scores and weights require independent validation.'},
        'eligible_selected_samples':sum(bool(set(r['sample']['elements']) & set(targets)) for r in records),
        'predicted_selected_samples':int(frame.sample_id.nunique()) if len(frame) else 0,
        'uncovered_samples':[{'sample_id':r['sample']['sample_id'],'batch_id':r['sample']['batch_id'],
            'reason':'No evaluable fold: insufficient comparable batches, group size or training records/recipes'}
            for r in records if set(r['sample']['elements']) & set(targets) and
            (not len(frame) or r['sample']['sample_id'] not in set(frame.sample_id))],
        'sample_weighted_score':float(scored.validation_score.mean()) if len(scored) and agreement_method==COMPONENT else float(display_score(scored.validation_loss.mean())) if len(scored) else None,
        'baseline_score':float(scored.baseline_agreement_score.mean()) if len(scored) and agreement_method==COMPONENT else float(display_score(scored.baseline_loss.mean())) if len(scored) else None,
        'element_balanced_score':float(np.mean([r['score'] for r in report if r['score'] is not None])) if agreement_method==COMPONENT and any(r['score'] is not None for r in report) else float(display_score(np.mean([100/r['score']-1 for r in report if r['score'] is not None]))) if any(r['score'] is not None for r in report) else None,
        'folds':provenance,'skipped_folds':skipped,'dataset_fingerprint':fingerprint(samples),
        'methods_fingerprint':fingerprint(methods),
        'descriptor_snapshot_sha256':reference()['snapshot_sha256'],'state_snapshot_sha256':state_reference()['snapshot_sha256'],
        'thermal_features_enabled':bool(use_thermal),'thermal_snapshot_sha256':thermal_reference()['snapshot_sha256'] if use_thermal else None,
        'exclusions':dict(Counter(audit.loc[~audit.eligible,'reason'])) if len(audit) else {},
        'note':'Fixed outcome-specific random forests benchmark uncalibrated probabilities; the ICP head is selectable. All heads share whole-batch outer masks. Rotating mode tests every eligible sample once unless an explicit fitting/context gate prevents coverage; no fold-budget truncation is applied. Repeated recipes may remain in training batches. This is a separate retrospective validation engine, not the current Next batch predictor or saved prospective performance. Scores evaluate prediction agreement, not synthesis quality. One fixed normalized endpoint set is required for pooled scores. Missing endpoints are not zero; every selected ICP member must be paired. Binary losses count once per sample. Joint screening and component judgments are correlated; weighting both can double-count evidence. Selecting models/weights using these results requires a new independent final evaluation.'}
    if sac_gen:
        info['note']='SAC-Gen learns mass-conserving ICP distributions and an L2 logistic XRD probability head. XRD features combine recipe chemistry, an elicited mechanistic prior and inner group-held-out ICP summaries. Random recipe mode permits administrative batch overlap while keeping identical recipes and specimens separate. Outer test outcomes never enter either head. No ICP pass probability or material-utility score is used. Correct XRD FAIL forecasts earn high agreement scores. Probabilities are trained but uncalibrated. Fixed physical anchors are prior assumptions, not independently validated mechanisms. Model/weight selection using these development folds requires a new independent final evaluation.'
    info['split_unit']='recipe' if protocol==RANDOM_RECIPE_CV else 'batch'
    emit({'stage':'Validation complete','completed':True,'folds':len(splits)})
    return frame,sample_scores,by_element,audit,info
