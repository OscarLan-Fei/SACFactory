"""Compare fixed forward models and audit their saved predictions after feedback."""
import copy
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .domain import INVALID_RUN_CONDITIONS, parse_measurement, structural_assessment
from .planning import fingerprint
from .queueing import controlled_recipe_key


def forecast_recipe_key(sample):
    loss = sample.get('carbonization_loss_fraction', sample.get('carbonization_loss', .06))
    return controlled_recipe_key(sample), float(loss)


def endpoint_presence(sample):
    icp = any(a.get('element') in sample.get('elements', []) and
              (a.get('flag') in {'ND', 'BLQ'} or parse_measurement(a.get('value', ''))['kind'] != 'missing')
              for a in sample.get('icp', []))
    return {'icp': icp, 'structure': structural_assessment(sample)[0] in {'PASS', 'FAIL'}}


def record_endpoint_times(sample, previous, timestamp):
    """Keep the first recorded result time; do not invent times for old imports."""
    times = copy.deepcopy(previous.get('endpoint_reported_at', {}))
    old, current = endpoint_presence(previous), endpoint_presence(sample)
    for name, present in current.items():
        if present and name not in times:
            known = previous.get('result_reported_at') if old[name] else timestamp
            if known:
                times[name] = known
    if times:
        sample['endpoint_reported_at'] = times
    else:
        sample.pop('endpoint_reported_at', None)


def _capture(model):
    return {'training_membership_status': 'NOT_RELEASED' if getattr(model, 'is_public_release', False) else 'LOCAL_TRAINING_IDENTITY',
            'training_batch_ids': sorted({s.get('batch_id') for s in model.training_samples if s.get('batch_id')}),
            'training_records': [{'sample_id': s['sample_id'], 'record_uid': s.get('record_uid')} for s in model.training_samples]}


def compare_batch(rows, root, origin, names, samples):
    """Forecast the same unmeasured recipes; do not fit or mutate any model/data."""
    from .discovery.common import FIELDS
    from .planning_models import load, prepare_context
    if not rows or not names or len(set(names)) != len(names):
        raise ValueError('Select a nonempty batch and distinct installed models.')
    if any(r.get('origin') != origin for r in rows):
        raise ValueError('Compare recipes within one data space.')
    output, table = {}, []
    created = datetime.now(timezone.utc).isoformat()
    for name in names:
        model = load(root, origin, name, samples)
        recipes = [prepare_context({k: copy.deepcopy(r['recipe'][k]) for k in FIELDS if k in r['recipe']}, model) for r in rows]
        forecasts = model.predict(recipes)
        if len(forecasts) != len(rows):
            raise ValueError('The model returned an incomplete batch.')
        for row, recipe, forecast in zip(rows, recipes, forecasts):
            probability = forecast.get('p_xrd_pass')
            loading = forecast.get('predicted_total_wt_percent')
            if any(v is None or not math.isfinite(v) for v in (probability, loading)) or not 0 <= probability <= 1 or not 0 <= loading <= 100:
                raise ValueError(name+' could not forecast every recipe; inspect inputs and reference coverage.')
            entry = {'created': created, 'recipe': recipe, 'model_id': model.model_id, 'model_name': name,
                     'sac_gen_prediction': forecast, 'evaluation_capture': _capture(model),
                     'policy': 'SAME_RECIPE_MODEL_COMPARISON', 'prediction_stale': False}
            output.setdefault(row['recommendation_id'], []).append(entry)
            table.append({'recommendation_id': row['recommendation_id'], 'model': name, 'model_id': model.model_id,
                          'elements': '+'.join(recipe['elements']), 'p_xrd_pass': probability,
                          'predicted_total_wt_percent': loading, 'outside_support': forecast.get('outside_training_support')})
    return output, pd.DataFrame(table)


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _loading(sample):
    values = {}
    for assay in sample.get('icp', []):
        element = assay.get('element')
        if element not in sample['elements']:
            continue
        if assay.get('unit', 'wt%') != 'wt%':
            raise ValueError('ICP content must use wt%.')
        parsed = parse_measurement(assay.get('value', ''))
        if assay.get('flag', 'MEASURED') not in {'', 'MEASURED'} or parsed['kind'] != 'measured':
            continue
        if element in values:
            raise ValueError('Duplicate measured ICP element.')
        value = float(parsed['value'])
        if not 0 <= value <= 100:
            raise ValueError('ICP content must be between zero and 100 wt%.')
        values[element] = value
    if sum(values.values()) > 100 + 1e-9:
        raise ValueError('Recorded ICP mass fractions exceed 100 wt%.')
    return values


def _forecasts(sample, tasks):
    task = tasks.get(sample.get('recommendation_id'), {})
    containers = [sample.get('proposal') or {}, task.get('original_proposal') or task,
                  *sample.get('manual_prediction_history', []), sample.get('manual_prediction') or {}]
    entries = []
    for item in containers:
        if item.get('sac_gen_prediction'):
            entries.append(item)
        entries.extend(item.get('comparison_predictions', []))
    # A copied current forecast and its immutable history are one forecast.
    unique = {fingerprint([e.get('model_id'), e.get('created'), e.get('recipe')]): e for e in entries}
    return list(unique.values())


def follow_up(samples, origin, recommendations=()):
    """Return endpoint-specific pairs and exclusions; never refit on results."""
    tasks = {r['recommendation_id']: r for r in recommendations}
    audit, pairs, structures = [], [], []
    for sample in samples:
        if sample.get('origin') != origin:
            continue
        entries = _forecasts(sample, tasks)
        # Keep one current-control forecast per sample/model. Older revisions
        # remain in the audit but never duplicate a specimen in pooled metrics.
        matched = {}
        for entry in entries:
            try:
                same = forecast_recipe_key(entry['recipe']) == forecast_recipe_key(sample)
            except (KeyError, TypeError, ValueError):
                same = False
            if same and not entry.get('prediction_stale'):
                key = entry.get('model_id')
                time = pd.to_datetime(entry.get('created'), utc=True, errors='coerce')
                if entry['sac_gen_prediction'].get('model_id') != key:
                    continue
                for endpoint in ('icp', 'structure'):
                    reported = sample.get('endpoint_reported_at', {}).get(endpoint) or sample.get('result_reported_at')
                    observed_time = pd.to_datetime(reported, utc=True, errors='coerce')
                    if pd.notna(observed_time) and pd.notna(time) and time >= observed_time:
                        continue
                    ident = (key, endpoint)
                    if pd.notna(time) and (ident not in matched or time > matched[ident][0]):
                        matched[ident] = (time, entry)
        for entry in entries:
            forecast = entry['sac_gen_prediction']
            model_id = entry.get('model_id') or forecast.get('model_id')
            meta = {'sample_id': sample['sample_id'], 'record_uid': sample.get('record_uid'),
                    'batch_id': sample.get('batch_id'), 'model': forecast.get('model_name') or entry.get('model_name', 'SAC-Gen'),
                    'model_id': model_id, 'prediction_created': entry.get('created')}
            reasons = []
            if not model_id or forecast.get('model_id') != model_id:
                reasons.append('Forecast model identity mismatch')
            if entry.get('origin', sample['origin']) != origin or entry.get('recipe', {}).get('origin') != origin:
                reasons.append('Data-space mismatch')
            if sample.get('sample_condition') in INVALID_RUN_CONDITIONS:
                reasons.append('Invalid or interrupted experiment')
            capture = entry.get('evaluation_capture') or {}
            if sample.get('batch_id') in capture.get('training_batch_ids', []):
                reasons.append('Synthesis batch used for training')
            if any((ref.get('record_uid') and ref['record_uid'] == sample.get('record_uid')) or
                   (not ref.get('record_uid') and ref.get('sample_id') == sample['sample_id']) for ref in capture.get('training_records', [])):
                reasons.append('Sample used for training')
            from .manual import actual_batch_recorded
            overlap_verified = bool(capture.get('training_membership_status') == 'LOCAL_TRAINING_IDENTITY'
                                    and sample.get('record_uid') and actual_batch_recorded(sample)
                                    and capture.get('training_records')
                                    and all(ref.get('record_uid') for ref in capture['training_records']))
            created = pd.to_datetime(entry.get('created'), utc=True, errors='coerce')
            state, _ = structural_assessment(sample)
            try:
                actual = _loading(sample)
                assay_error = None
            except ValueError as exc:
                actual, assay_error = {}, str(exc)
            predicted = {}
            for item in forecast.get('icp_predictions', []):
                element = item.get('element')
                if element in predicted:
                    assay_error = 'Duplicate predicted ICP element'
                predicted[element] = _number(item.get('predicted_mean_w_wt_percent'))
            for endpoint in ('icp', 'structure'):
                excluded = list(reasons)
                if matched.get((model_id, endpoint), (None, None))[1] is not entry:
                    excluded.append('Changed recipe, stale forecast or superseded revision')
                reported = sample.get('endpoint_reported_at', {}).get(endpoint) or sample.get('result_reported_at')
                time = pd.to_datetime(reported, utc=True, errors='coerce')
                if pd.isna(created) or pd.isna(time) or created >= time:
                    excluded.append('Pre-result timing unverified')
                if endpoint == 'icp':
                    common = [e for e in sample['elements'] if e in actual and predicted.get(e) is not None and 0 <= predicted[e] <= 100]
                    if assay_error:
                        excluded.append(assay_error)
                    if not common:
                        excluded.append('No paired quantified ICP; ND/BLQ/missing are excluded')
                    if sample.get('sample_condition') == 'BURNED_OUT':
                        excluded.append('No recovered material for ICP mass fraction')
                    if not excluded:
                        for element in common:
                            pairs.append({**meta, 'element': element, 'observed_wt_percent': actual[element],
                                          'predicted_wt_percent': predicted[element], 'overlap_verified': overlap_verified,
                                          'specimen_weight': 1/len(common)})
                else:
                    probability = _number(forecast.get('p_xrd_pass'))
                    threshold = _number(forecast.get('xrd_pass_threshold', .5))
                    if state not in {'PASS', 'FAIL'}:
                        excluded.append('No resolved structural assessment')
                    if probability is None or not 0 <= probability <= 1 or threshold is None or not 0 <= threshold <= 1:
                        excluded.append('No valid structural forecast')
                    if not excluded:
                        structures.append({**meta, 'label': int(state == 'PASS'), 'p_pass': probability,
                                           'predicted_label': int(probability >= threshold), 'threshold': threshold,
                                           'overlap_verified': overlap_verified})
                audit.append({**meta, 'endpoint': endpoint, 'eligible': not excluded,
                              'training_overlap': 'Verified local snapshot' if overlap_verified else 'Unverified membership',
                              'result_recorded_at': reported, 'reason': '; '.join(dict.fromkeys(excluded))})
    return pd.DataFrame(pairs), pd.DataFrame(structures), pd.DataFrame(audit)


def metrics(pairs, structures):
    summaries = []
    keys = {(r.model, r.model_id) for frame in (pairs, structures) for r in frame.itertuples()}
    for name, ident in sorted(keys):
        row = {'model': name, 'model_id': ident}
        data = pairs.loc[(pairs.model == name) & (pairs.model_id == ident)] if len(pairs) else pairs
        if len(data):
            x, y, w = (data[k].to_numpy(float) for k in ('predicted_wt_percent', 'observed_wt_percent', 'specimen_weight'))
            error = y-x
            denominator = np.sum(w*(y-np.average(y, weights=w))**2)
            row.update(ICP_specimens=data.sample_id.nunique(), ICP_pairs=len(data),
                       ICP_MAE=float(np.average(abs(error),weights=w)), ICP_RMSE=float(np.sqrt(np.average(error**2,weights=w))),
                       ICP_R_squared=float(1-np.sum(w*error**2)/denominator) if denominator > 1e-12 else None)
            if len(np.unique(x)) > 1:
                slope, intercept = np.linalg.lstsq(np.c_[x, np.ones(len(x))]*np.sqrt(w[:,None]), y*np.sqrt(w), rcond=None)[0]
                row.update(fit_slope=float(slope), fit_intercept=float(intercept))
        labels = structures.loc[(structures.model == name) & (structures.model_id == ident)] if len(structures) else structures
        if len(labels):
            recalls = [float((g.predicted_label == target).mean()) for target in (0,1)
                       if len(g := labels.loc[labels.label == target])]
            row.update(structural_specimens=len(labels), structural_Brier=float(((labels.p_pass-labels.label)**2).mean()),
                       structural_balanced_accuracy=float(np.mean(recalls)) if len(recalls)==2 else None,
                       FAIL_recall=float((labels.loc[labels.label == 0].predicted_label == 0).mean()) if (labels.label == 0).any() else None)
        summaries.append(row)
    return pd.DataFrame(summaries)
