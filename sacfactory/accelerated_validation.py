"""Device-aware adapter for protected training/calibration/test artifacts."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json
import numpy as np
import pandas as pd

from ._pulseicp.data import prepare
from ._pulseicp.transfer import protected_split
from .accelerated_ensemble import ResidualEnsemble, FAMILIES
from ._pulseicp.mass_decoder import prior
from ._pulseicp.references import digest, provenance
from ._pulseicp.uncertainty import batch_calibration, metrics


def run_experiment(frame, origin, physics, settings, output, split='Batch holdout', callback=None, held_element=None, source_elements=None):
    data = prepare(frame, physics, origin, settings.get('thermodynamic_features', False))
    if not data.records:
        raise ValueError('No eligible quantified ICP records. Inspect the data audit.')
    if len({r['raw']['protocol_id'] for r in data.records}) != 1:
        raise ValueError('Filter to one comparable synthesis protocol before training.')
    (train, calibration, test), partitions = protected_split(data, int(settings['seed']), split, held_element, source_elements)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
    destination = Path(output)/run_id
    destination.mkdir(parents=True, exist_ok=False)
    data.audit.to_csv(destination/'eligibility.csv', index=False)
    predictions, comparisons, models, stratified = [], [], {}, []

    protocol = data.records[train[0]]['raw']['protocol_id']
    metadata = {'version': '0.1.0', 'run_id': run_id, 'origin': origin, 'protocol': protocol,
        'dataset_fingerprint': digest(frame.astype(object).where(pd.notna(frame), None).to_dict('records')),
        'physics': physics, 'settings': settings, 'reference_provenance': provenance(), 'split_protocol': split,
        'partitions': partitions, 'held_element': held_element, 'source_elements': source_elements,
        'eligible_n': len({r['raw']['sample_id'] for r in data.records}), 'eligible_target_rows': len(data.records), 'excluded_n': int((~data.audit.eligible).sum()),
        'assessment': 'Synthetic implementation evidence only' if origin == 'SIMULATED' else 'Retrospective held-out ICP validation; not proof of atomic dispersion',
        'note': 'Transforms, bootstrap fits and model weights use training only. Calibration uses separate actual batches. Test outcomes are used only in final metrics. Zero/censored/bound-incompatible targets are audited, not fabricated. Interval coverage is conditional on the eligible population.'}
    try:
        for family, folder in zip(FAMILIES, ('gaussian', 'diffusion')):
            model = ResidualEnsemble(family, settings, physics).fit(data, train, callback)
            cal_frame, _ = predict_records(model, data, calibration, origin, protocol)
            model.calibration = batch_calibration(cal_frame.lower_95_w_wt_percent.to_numpy(),
                cal_frame.upper_95_w_wt_percent.to_numpy(), data.w[calibration],
                [str(data.records[i]['raw']['batch_id']) for i in calibration])
            model.save(destination/folder)
            predicted, draws = predict_records(model, data, test, origin, protocol)
            predicted['sample_id'] = [data.records[i]['raw']['sample_id'] for i in test]
            predicted['batch_id'] = [data.records[i]['raw']['batch_id'] for i in test]
            predicted['observed_w_wt_percent'] = data.w[test]
            predictions.append(predicted)
            comparisons.append({'model': family, **metrics(predicted, draws),
                'calibrated_samples': int((predicted.calibration_status=='BATCH_MAX_SPLIT_CALIBRATION').sum()),
                'outside_support_n': int(predicted.outside_training_support.sum())})
            for (composition,element),group in predicted.groupby(['composition_key','element']):
                stratified.append({'model':family,'composition_key':composition,'element':element,**metrics(group,draws[group.index])})
            models[folder] = {'path': folder, 'model_id': model.model_id, 'calibration': model.calibration,
                             'parameter_count_per_member': sum(p.numel() for p in model.members[0].parameters())}
        point = predictions[0].copy()
        point['predicted_mean_w_wt_percent'] = point.physical_prior_w_wt_percent.to_numpy()
        point[['predicted_sd_w_wt_percent', 'within_model_sd_w_wt_percent', 'between_model_sd_w_wt_percent',
               'lower_95_w_wt_percent', 'upper_95_w_wt_percent']] = np.nan
        point['model'] = 'Unfitted physical prior'
        point['thermodynamics_used_by_model'] = False
        point['model_id'] = digest(physics)[:20]
        point['uncertainty_method'] = 'Deterministic unfitted prior; uncertainty not estimated'
        point['diffusion_guard_fraction'] = np.nan
        point['calibration_status'] = 'UNCERTAINTY_NOT_ESTIMATED'
        comparisons.insert(0, {'model': 'Unfitted physical prior', **metrics(point)})
        for (composition,element),group in point.groupby(['composition_key','element']):
            stratified.append({'model':'Unfitted physical prior','composition_key':composition,'element':element,**metrics(group)})
        pd.DataFrame(stratified).to_csv(destination/'stratified_comparison.csv',index=False)
        predictions.insert(0, point)
        combined = pd.concat(predictions, ignore_index=True)
        combined.to_csv(destination/'test_predictions.csv', index=False)
        pd.DataFrame(comparisons).to_csv(destination/'comparison.csv', index=False)
        metadata.update(status='COMPLETE', comparisons=comparisons, models=models)
    except Exception:
        metadata['status'] = 'FAILED_NOT_PROMOTED'
        (destination/'run.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
        raise
    (destination/'run.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    pointer = Path(output)/'latest.json'
    temporary = Path(output)/'latest.tmp'
    temporary.write_text(json.dumps({'run_id': run_id, 'origin': origin, 'protocol': protocol}), encoding='utf-8')
    temporary.replace(pointer)
    return destination, combined, pd.DataFrame(comparisons), metadata


def predict_records(model, data, indices, origin, protocol):
    """Predict each complete composition once, then align component rows exactly."""
    compositions, sample_ids = [], []
    for i in indices:
        ident = data.records[i]['raw']['sample_id']
        if ident not in sample_ids:
            sample_ids.append(ident)
            compositions.append(data.records[i]['composition'])
    frame, draws = model.predict_compositions(compositions, origin=origin, protocol=protocol)
    lookup = {(sample_ids[int(row.composition_index)], row.element): i for i,row in frame.iterrows()}
    ordered = [lookup[(data.records[i]['raw']['sample_id'], data.records[i]['recipe'].element)] for i in indices]
    return frame.iloc[ordered].reset_index(drop=True), draws[ordered]
