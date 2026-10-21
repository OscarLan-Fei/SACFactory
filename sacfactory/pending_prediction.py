"""Audited model forecasts for saved, human-adjusted pending recipes."""
import copy
import json

import numpy as np

from .domain import RULE_VERSION
from .manual import has_observations, actual_batch_recorded, pending_samples
from .planning import fingerprint, fit_model, feature
from .composition import model_feature
from .queueing import controlled_recipe_key
from .storage import now


def item_snapshot(connection):
    return fingerprint(connection.execute('SELECT kind,id,payload FROM items ORDER BY kind,id').fetchall())


def refresh_pending_prediction(store, sample_id, expected):
    """Fit outside the write lock; reject any concurrent dataset/settings change."""
    with store.connect() as connection:
        connection.execute('BEGIN')
        snapshot = item_snapshot(connection)
        entries = connection.execute('SELECT kind,id,payload FROM items').fetchall()
    data = {(kind, ident): json.loads(payload) for kind, ident, payload in entries}
    sample = data.get(('sample', sample_id))
    if not sample or fingerprint(sample) != expected:
        raise ValueError('Pending sample changed. Reload before predicting.')
    samples = [v for (kind, _), v in data.items() if kind == 'sample']
    queues = [v for (kind, _), v in data.items() if kind == 'recommendation']
    methods = {ident: v for (kind, ident), v in data.items() if kind == 'method'}
    if has_observations(sample) or sample.get('result_reported_at') or sample_id not in {s['sample_id'] for s in pending_samples(samples, queues, methods)}:
        raise ValueError('Only an unmeasured pending sample can receive a new pre-result prediction.')
    if sample.get('pulse_limit_C') is None:
        raise ValueError('Enter a controlled cutoff before predicting a pending recipe.')
    from .sac_gen import latest as latest_sac_gen
    deployed=latest_sac_gen(store.root,store.origin,samples)
    if deployed is not None:
        if sample_id in deployed.training_ids:
            raise ValueError('This pending sample belongs to the SAC-Gen training snapshot. Retrain without it before reserving a pre-result forecast.')
        forecast=deployed.predict([sample])[0]
        prediction={'created':now(),'origin':store.origin,'recipe':{k:copy.deepcopy(sample.get(k)) for k in
            ('origin','elements','concentrations_mM','pulse_count','pulse_limit_C','precursor_family','precursors',
             'support_batch','protocol_id','pan_mass_g','solvent_volume_mL','preparation_basis') if k in sample},
            'model_name':'SAC-Gen','model_id':deployed.model_id,'sac_gen_prediction':forecast,
            'p_xrd_pass':forecast['p_xrd_pass'],'p_xrd_fail':forecast['p_xrd_fail'],'p_screen_pass':None,
            'icp_prediction':{'values':forecast['icp_predictions'],'basis':'HUMAN_ADJUSTED_PRE_SYNTHESIS_RECIPE'},
            'prediction_stale':False,'policy':'HUMAN_ADJUSTED_RECIPE','prediction_target':'XRD_PASS',
            'prediction_mode':'MODEL_FORECAST_AFTER_MANUAL_INTERVENTION',
            'evaluation_capture':{'rule_version':RULE_VERSION,'training_batch_ids':sorted({s['batch_id'] for s in deployed.training_samples}),
                'training_records':[{'sample_id':s['sample_id'],'record_uid':s.get('record_uid')} for s in deployed.training_samples],
                'dataset_snapshot':snapshot},
            'note':'Trained SAC-Gen loading and XRD forecasts. ICP pass probability and material-quality penalties are not used.'}
        return persist_prediction(store,sample_id,snapshot,data,sample,prediction,None)
    config = data.get(('meta', 'space'), {}).get('space', {})
    campaign = data.get(('meta', 'element_campaign'), {})
    from .element_campaign import campaign_training_samples
    sources = [s for s in samples if s['sample_id'] != sample_id and
               (not actual_batch_recorded(sample) or s['batch_id'] != sample['batch_id'])]
    sources = campaign_training_samples(sources, campaign.get('enabled', False), sample)
    engine = config.get('model_engine', 'adaptive')
    settings = {'learning_iterations': config.get('learning_iterations', 18),
                'batch_effect': config.get('batch_effect', False), 'skip_validation': True,
                'descriptor_profile': config.get('descriptor_profile', 'chemistry_pulseicp') if engine != 'baseline' else 'periodic'}
    state = fit_model(sources, methods, sample, store.origin, config.get('element_transfer', True), engine, settings)
    if state.model is None:
        raise ValueError('No fitted probability is available: ' + state.info.get('reason', 'Insufficient eligible training data.'))
    x = np.asarray([model_feature(sample, state.model) if state.info.get('feature_kind') == 'COMPOSITION_TRANSFER' else feature(sample)])
    probability = float(state.model.predict_proba(x)[0, 1])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('The fitted model did not return a valid probability.')
    prediction = {'created': now(), 'origin': store.origin, 'recipe': copy.deepcopy(sample),
        'model_id': state.info['model_id'], 'model_info': state.info, 'p_screen_pass': probability,
        'prediction_stale': False, 'policy': 'HUMAN_ADJUSTED_RECIPE',
        'prediction_mode': 'MODEL_FORECAST_AFTER_MANUAL_INTERVENTION',
        'evaluation_capture': {'rule_version': RULE_VERSION,
            'training_batch_ids': sorted({s['batch_id'] for s in state.training}),
            'training_records': [{'sample_id': s['sample_id'], 'record_uid': s.get('record_uid')} for s in state.training],
            'dataset_snapshot': snapshot},
        'note': 'Prediction of the saved manually adjusted recipe. Human intervention is explicit; this is not an untouched algorithm-selected experiment. No target outcomes were available to this fit.'}
    # Store controls only, not recursively nested forecast histories or outcomes.
    prediction['recipe'] = {k: copy.deepcopy(sample.get(k)) for k in
        ('origin', 'elements', 'concentrations_mM', 'pulse_count', 'pulse_limit_C', 'precursor_family', 'precursors', 'support_batch', 'protocol_id', 'pan_mass_g', 'solvent_volume_mL', 'preparation_basis') if k in sample}
    from .mixing import forecast_metadata
    prediction['mixing'] = forecast_metadata(sample, state.info, probability)
    from .pulseicp_bridge import forecast as icp_forecast, identity as icp_identity
    try:
        frame = icp_forecast([sample], store.origin, store.root, current_samples=samples)
        prediction['icp_prediction'] = {'values':frame.astype(object).where(frame.notna(), None).to_dict('records'),
            'snapshot':icp_identity(), 'basis':'HUMAN_ADJUSTED_PRE_SYNTHESIS_RECIPE'}
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        prediction['icp_prediction'] = {'status':'UNAVAILABLE', 'reason':str(exc)}
    return persist_prediction(store,sample_id,snapshot,data,sample,prediction,probability)


def persist_prediction(store,sample_id,snapshot,data,sample,prediction,probability):
    """Commit a pre-result forecast only if the source snapshot is unchanged."""
    with store.connect() as connection:
        connection.execute('BEGIN IMMEDIATE')
        if item_snapshot(connection) != snapshot:
            raise ValueError('Data or planning settings changed while fitting. Retry on the latest saved recipe.')
        updated = copy.deepcopy(sample)
        updated.setdefault('manual_prediction_history', []).append(copy.deepcopy(prediction))
        updated['manual_prediction'] = prediction
        before = json.dumps(sample, ensure_ascii=False, allow_nan=False)
        after = json.dumps(updated, ensure_ascii=False, allow_nan=False)
        connection.execute("UPDATE items SET payload=? WHERE kind='sample' AND id=?", (after, sample_id))
        connection.execute('INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,?,?,?,?,?)',
            (now(), 'predict_manually_adjusted_pending', 'sample', sample_id, before, after))
        link = sample.get('recommendation_id')
        queue = data.get(('recommendation', link))
        if queue:
            amended = copy.deepcopy(queue)
            amended.update(manual_prediction=prediction, p_screen_pass=probability, prediction_stale=False,
                           current_prediction_basis='HUMAN_ADJUSTED_RECIPE')
            if prediction.get('model_name')=='SAC-Gen':
                amended.update(p_xrd_pass=prediction['p_xrd_pass'],p_xrd_fail=prediction['p_xrd_fail'],
                    sac_gen_prediction=prediction['sac_gen_prediction'])
            connection.execute("UPDATE items SET payload=? WHERE kind='recommendation' AND id=?",
                (json.dumps(amended, ensure_ascii=False, allow_nan=False), link))
            connection.execute('INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,?,?,?,?,?)',
                (now(), 'predict_manually_adjusted_pending', 'recommendation', link,
                 json.dumps(queue, ensure_ascii=False, allow_nan=False), json.dumps(amended, ensure_ascii=False, allow_nan=False)))
    return prediction
