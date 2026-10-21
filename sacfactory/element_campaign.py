"""Single-element-first screening coverage and protected cross-element replay."""
import numpy as np
import pandas as pd

from .adaptive import AdaptivePredictor, eligible_records
from .composition import transfer_feature
from .domain import TARGETS, screen
from .manual import actual_batch_recorded
from .validation_research import fitting_ready
from .planning import fingerprint
from .probability_evaluation import probability_evaluation
from .chemical_descriptors import reference
from .precursors import state_reference


def campaign_progress(samples, methods, origin, targets=TARGETS):
    rows = []
    for element in targets:
        singles = [s for s in samples if s.get('origin') == origin and s['elements'] == [element]]
        results = [(s, screen(s, methods)) for s in singles]
        resolved = [(s, r) for s, r in results if r['label'] is not None and not r['review_conflicts']]
        passed = [(s, r) for s, r in resolved if r['label'] == 1]
        pass_batches = {s['batch_id'] for s, _ in passed if actual_batch_recorded(s)}
        rows.append({'element': element, 'single_element_records': len(singles), 'resolved': len(resolved),
            'passed': len(passed), 'failed': sum(r['label'] == 0 for _, r in resolved),
            'pass_batches': len(pass_batches), 'candidate_covered': bool(pass_batches),
            'status': 'Screening candidate obtained' if pass_batches else 'Needs single-element PASS' if singles else 'Not started'})
    return pd.DataFrame(rows)


def campaign_training_samples(samples, enabled, context):
    return [s for s in samples if len(s['elements']) == 1] if enabled and len(context['elements']) == 1 else samples


def enforce_campaign(context, space):
    policy = space.get('_single_element_campaign', {})
    if policy.get('enabled') and len(context['elements']) > 1 and not policy.get('ready'):
        raise ValueError(f"Single-element-first campaign: {policy.get('covered', 0)} / {policy.get('total', 50)} target elements have a single-element screening PASS in a named batch. Complete the singles before proposing mixtures. Progress is under Data analysis > Element campaign.")


def element_transfer_validation(samples, methods, context, origin, target, sources=(),
                                protocol='Strict unseen element', profile='chemistry_ion', iterations=4, max_folds=3):
    if protocol not in {'Strict unseen element', 'Sequential element replay'}:
        raise ValueError('Select a supported element-transfer protocol.')
    singles = [s for s in samples if len(s['elements']) == 1 and
               (not sources or s['elements'][0] in set(sources) | {target})]
    records, exclusions = eligible_records(singles, methods, context, origin, True, profile)
    test = [s for s in records if s['elements'] == [target]]
    splits, missing = [], []
    if protocol == 'Strict unseen element' and test:
        batches = {s['batch_id'] for s in test}
        splits = [([s for s in records if s['elements'] != [target] and s['batch_id'] not in batches], test)]
    elif protocol == 'Sequential element replay':
        times = {}
        for batch in {s['batch_id'] for s in records}:
            dates = pd.to_datetime([s.get('result_reported_at') for s in singles if s.get('origin') == origin and s['batch_id'] == batch], utc=True, errors='coerce')
            if dates.isna().any():
                missing.append(batch)
            else:
                times[batch] = (dates.min(), dates.max())
        for batch in sorted({s['batch_id'] for s in test if s['batch_id'] in times}, key=lambda b: times[b][0]):
            train = [s for s in records if s['batch_id'] in times and times[s['batch_id']][1] < times[batch][0]]
            splits.append((train, [s for s in test if s['batch_id'] == batch]))
        if len(splits) > max_folds:
            splits = [splits[i] for i in np.unique(np.linspace(0, len(splits)-1, int(max_folds), dtype=int))]
    predictions, folds, skipped = [], [], []
    for fold, (train, heldout) in enumerate(splits, 1):
        minimum = 8 if origin == 'REAL' else 3
        if not fitting_ready(train, methods) or len({s['batch_id'] for s in train}) < minimum:
            skipped.append({'fold': fold, 'reason': 'Insufficient source records, independent batches or both-class recipe diversity'})
            continue
        model = AdaptivePredictor(iterations, descriptor_profile=profile).fit(
            np.array([transfer_feature(s, profile, for_training=True) for s in train]),
            np.array([screen(s, methods)['label'] for s in train]), [s['batch_id'] for s in train])
        p = model.predict_proba(np.array([transfer_feature(s, profile, for_training=True) for s in heldout]))[:, 1]
        train_elements = sorted({s['elements'][0] for s in train})
        folds.append({'fold': fold, 'train_elements': train_elements,
            'train_sample_ids': [s['sample_id'] for s in train], 'test_sample_ids': [s['sample_id'] for s in heldout],
            'train_batch_ids': sorted({s['batch_id'] for s in train}), 'test_batch_ids': sorted({s['batch_id'] for s in heldout}),
            'calibration_batch_ids': sorted({train[i]['batch_id'] for i in model.calibration_indices}),
            'latest_train_result': max(str(s.get('result_reported_at') or '') for s in train),
            'first_test_result': min(str(s.get('result_reported_at') or '') for s in heldout)})
        predictions.extend({'sample_id': s['sample_id'], 'batch_id': s['batch_id'], 'target_element': target,
            'label': screen(s, methods)['label'], 'p_screen_pass': float(value), 'fold': fold,
            'training_n': len(train), 'source_elements': ';'.join(train_elements),
            'target_seen_before': target in train_elements} for s, value in zip(heldout, p))
    frame = pd.DataFrame(predictions, columns=['sample_id', 'batch_id', 'target_element', 'label', 'p_screen_pass',
                                             'fold', 'training_n', 'source_elements', 'target_seen_before'])
    info = {'protocol': protocol, 'target': target, 'source_filter': list(sources), 'profile': profile,
        'fit_iterations': int(iterations), 'fold_budget': int(max_folds),
        'descriptor_snapshot_sha256': reference()['snapshot_sha256'], 'state_snapshot_sha256': state_reference()['snapshot_sha256'],
        'origin': origin, 'dataset_fingerprint': fingerprint(samples), 'folds': folds, 'skipped': skipped,
        'missing_batch_timestamps': missing, 'exclusions': exclusions,
        'metrics': probability_evaluation(frame.assign(eligible=True))[0] if not frame.empty else None,
        'note': 'Single-element specimens only. Strict mode removes all target data and shared batches; sequential replay uses only earlier completed batches and may learn from earlier target results. Corrected-data replay is not a saved prospective test. Chemical similarity is a transfer hypothesis, not guaranteed success.'}
    return frame, info
