"""Paired chemistry ablations with batch purges and explicit expansion tests."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .adaptive import AdaptivePredictor, eligible_records, scores
from .composition import composition_name, transfer_feature
from .domain import screen
from .reliable_transfer import ready
from .precursors import PROFILES, precursor_coverage, state_reference
from .chemical_descriptors import reference
from .physics_features import physics_manifest


def chemistry_ablation(samples, methods, context, origin, protocol='single_to_mixture',
                      element=None, iterations=4, max_folds=3, batch_effect=True):
    """All variants use identical training rows, controls, labels and test batches."""
    records, exclusions = eligible_records(samples, methods, context, origin, True, 'chemistry_precursor')
    target = [s for s in records if composition_name(s) == composition_name(context)]
    splits = []
    if protocol in {'single_to_mixture', 'lower_to_higher'}:
        test = target if len(context['elements']) > 1 else [s for s in records if len(s['elements']) > 1]
        stages = sorted({len(s['elements']) for s in test})
        for stage in stages[:max_folds]:
            current = [s for s in test if len(s['elements']) == stage]
            held_out = {s['batch_id'] for s in current}
            train = [s for s in records if (len(s['elements']) == 1 if protocol == 'single_to_mixture' else len(s['elements']) < stage)
                     and s['batch_id'] not in held_out]
            splits.append((train, current))
    elif protocol == 'unseen_element':
        if not element:
            raise ValueError('Choose a held-out element.')
        test = [s for s in records if element in s['elements']]
        held_out = {s['batch_id'] for s in test}
        if test:
            splits = [([s for s in records if element not in s['elements'] and s['batch_id'] not in held_out], test)]
    elif protocol == 'target_batch':
        groups = [s['batch_id'] for s in target]
        if len(set(groups)) >= 3:
            for _, test_indices in GroupKFold(n_splits=min(max_folds, len(set(groups)))).split(target, groups=groups):
                test = [target[i] for i in test_indices]
                held_out = {s['batch_id'] for s in test}
                splits.append(([s for s in records if s['batch_id'] not in held_out], test))
    else:
        raise ValueError('Select a supported chemical-transfer validation protocol.')
    rows, folds, skipped = [], [], []
    for fold, (train, test) in enumerate(splits):
        if not test or not ready(train, methods):
            skipped.append({'fold': fold, 'reason': 'Need measured target outcomes and at least 12 source rows with three recipes per outcome class.'})
            continue
        y = np.array([screen(s, methods)['label'] for s in train])
        groups = [s['batch_id'] for s in train]
        models = []
        for profile in PROFILES:
            x = np.array([transfer_feature(s, profile, for_training=True) for s in train])
            xt = np.array([transfer_feature(s, profile, for_training=True) for s in test])
            model = AdaptivePredictor(iterations, batch_effect=batch_effect, descriptor_profile=profile).fit(x, y, groups)
            p = model.predict_proba(xt)[:, 1]
            models.append({'profile': profile, 'kernel_fit_ids': [train[i]['sample_id'] for i in model.fit_indices],
                           'calibration_ids': [train[i]['sample_id'] for i in model.calibration_indices],
                           'calibration_status': model.calibration_status})
            rows.extend({'profile': profile, 'sample_id': s['sample_id'], 'batch_id': s['batch_id'],
                         'composition': composition_name(s), 'element_count': len(s['elements']),
                         'label': screen(s, methods)['label'], 'p_screen_pass': float(probability), 'fold': fold}
                        for s, probability in zip(test, p))
        folds.append({'fold': fold, 'train_sample_ids': [s['sample_id'] for s in train],
                      'test_sample_ids': [s['sample_id'] for s in test], 'train_batch_ids': sorted(set(groups)),
                      'test_batch_ids': sorted({s['batch_id'] for s in test}),
                      'train_stages': sorted({len(s['elements']) for s in train}),
                      'test_stages': sorted({len(s['elements']) for s in test}), 'models': models})
    frame = pd.DataFrame(rows, columns=['profile', 'sample_id', 'batch_id', 'composition', 'element_count', 'label', 'p_screen_pass', 'fold'])
    comparison, effects = [], []
    for profile, part in frame.groupby('profile'):
        loss = (part['label']-part['p_screen_pass'])**2
        batch_loss = loss.groupby(part['batch_id']).mean()
        comparison.append({'profile': profile, **scores(part['label'].to_numpy(), part['p_screen_pass'].to_numpy()),
                           'equal_batch_brier': float(batch_loss.mean()), 'test_batches': len(batch_loss)})
    if not frame.empty:
        wide = frame.pivot(index=['fold', 'sample_id', 'batch_id', 'label'], columns='profile', values='p_screen_pass').reset_index()
        labels = wide['label'].to_numpy()
        # Mixing adds to thermal inputs, not to the separate empirical-loss variant.
        comparisons = list(zip(list(PROFILES)[:6], list(PROFILES)[1:7])) + [('chemistry_thermal', 'chemistry_mixing'), ('chemistry_mixing', 'chemistry_pulseicp')]
        for first, second in comparisons:
            difference = (labels-wide[first])**2-(labels-wide[second])**2
            batch_effects = difference.groupby(wide['batch_id']).mean().to_numpy()
            low = high = None
            if len(batch_effects) >= 3:
                rng = np.random.default_rng(42)
                draws = rng.choice(batch_effects, (1000, len(batch_effects)), replace=True).mean(axis=1)
                low, high = map(float, np.quantile(draws, [.025, .975]))
            effects.append({'baseline': first, 'added_information': second,
                           'equal_batch_brier_improvement': float(batch_effects.mean()),
                           'bootstrap_lower_95': low, 'bootstrap_upper_95': high, 'test_batches': len(batch_effects),
                           'note': 'Positive favors added information. Exploratory paired actual-batch bootstrap; fixed training fits, not full refitting uncertainty.'})
    chemistry = [r for s in records for r in precursor_coverage(s, for_training=True)]
    info = {'protocol': protocol, 'held_out_element': element if protocol == 'unseen_element' else None,
            'comparison': comparison, 'paired_effects': effects, 'folds': folds, 'skipped': skipped,
            'exclusions': exclusions, 'eligible_records': len(records),
            'precursor_rows_with_source_and_type': sum(r['type_used_by_model'] != 'UNKNOWN' for r in chemistry),
            'ionic_rows_with_exact_state_match': sum(r['ionic_radius_pm'] is not None for r in chemistry),
            'descriptor_snapshot_sha256': reference()['snapshot_sha256'],
            'state_descriptor_snapshot_sha256': state_reference()['snapshot_sha256'],
            'physics_variants': {p: physics_manifest(p) for p in PROFILES},
            'note': 'Identical source/test rows and model family. Support, protocol and control basis stay fixed; precursor families may differ. Singles cannot establish mixture interaction effects. Missing descriptor/state fields remain masked. Retrospective exploratory evidence, not automatic promotion or prospective superiority.'}
    return frame, info
