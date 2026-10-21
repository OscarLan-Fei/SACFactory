"""Independent recorded-outcome heads and explicit prediction-agreement scores."""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

BINARY_ENDPOINTS = {
    'screening': 'Joint ICP/XRD screening pass',
    'icp_pass': 'ICP judgment pass',
    'xrd_pass': 'XRD judgment pass',
    'stem_confirmed': 'STEM confirmation',
    'burnout': 'Complete sample burnout',
}
ENDPOINTS = {'icp': 'ICP W / wt%', **BINARY_ENDPOINTS}


def outcome_labels(sample, outcome):
    """Unknown judgments and unrecorded recovery are never negative labels."""
    condition = sample.get('sample_condition')
    return {
        'screening': outcome['label'] if not outcome['review_conflicts'] else None,
        'icp_pass': {'PASS': 1, 'FAIL': 0}.get(outcome['icp']) if not outcome['review_conflicts'] else None,
        'xrd_pass': {'PASS': 1, 'FAIL': 0}.get(outcome['xrd']),
        'stem_confirmed': {'CONFIRMED': 1, 'NOT_CONFIRMED': 0}.get(outcome['stem']),
        'burnout': 1 if condition == 'BURNED_OUT' else 0 if condition in {'RECOVERED', 'PARTIAL_LOSS'} else None,
    }


def normalized_weights(weights, icp_weight):
    raw = {'icp': icp_weight, 'screening': 1-icp_weight} if weights is None else dict(weights)
    if not raw or set(raw)-set(ENDPOINTS):
        raise ValueError('Select recognized weighted-validation outcomes.')
    if any(isinstance(v, bool) or not math.isfinite(float(v)) or float(v) < 0 for v in raw.values()):
        raise ValueError('Outcome weights must be finite nonnegative numbers.')
    total = sum(float(v) for v in raw.values())
    if total <= 0:
        raise ValueError('At least one outcome weight must be positive.')
    return {k: float(v)/total for k, v in raw.items() if float(v) > 0}


def fit_outcomes(train, test, trees, seed,exclude=()):
    """Every head and source-mean baseline use only the outer training fold."""
    from .compute_runtime import jobs
    predictions, provenance = {}, {}
    for endpoint in BINARY_ENDPOINTS:
        labeled = [r for r in train if r['outcomes'][endpoint] is not None]
        labels = [r['outcomes'][endpoint] for r in labeled]
        available = endpoint not in exclude and len(labels) >= 12 and len(set(labels)) == 2
        if available:
            head = RandomForestClassifier(n_estimators=trees, min_samples_leaf=4,
                max_depth=14, random_state=seed, n_jobs=jobs()).fit([r['x'] for r in labeled], labels)
            values = head.predict_proba([r['x'] for r in test])[:, list(head.classes_).index(1)]
        else:
            values = [None] * len(test)
        baseline = float(np.mean(labels)) if labels else None
        predictions[endpoint] = (values, baseline)
        provenance[endpoint] = {'training_n': len(labels), 'classes': sorted(set(labels)),
            'training_sample_ids': [r['sample']['sample_id'] for r in labeled],
            'available': available, 'source_mean': baseline,
            'reason': '' if available else 'Need 12 training judgments and both outcome classes'}
    return predictions, provenance


def binary_metrics(points):
    """Pooled sample-level metrics; mixtures contribute one binary judgment."""
    result = []
    unique = points.drop_duplicates(['fold', 'sample_id']) if len(points) else points
    for endpoint, name in BINARY_ENDPOINTS.items():
        paired = unique.dropna(subset=[f'{endpoint}_observed', f'{endpoint}_predicted']) if len(unique) else unique
        observed_n = int(unique[f'{endpoint}_observed'].notna().sum()) if len(unique) else 0
        row = {'endpoint': endpoint, 'outcome': name, 'observed_samples': observed_n,
            'predicted_samples': int(unique[f'{endpoint}_predicted'].notna().sum()) if len(unique) else 0,
            'evaluated_samples': len(paired), 'brier': None, 'baseline_brier': None,
            'balanced_accuracy': None, 'roc_auc': None}
        if len(paired):
            y = paired[f'{endpoint}_observed'].to_numpy(dtype=float)
            p = paired[f'{endpoint}_predicted'].to_numpy(dtype=float)
            row.update(brier=float(np.mean((p-y)**2)),
                baseline_brier=float(paired[f'{endpoint}_baseline_loss'].mean()))
            if len(set(y)) == 2:
                row.update(balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)),
                    roc_auc=float(roc_auc_score(y, p)))
        row['status'] = 'Evaluated' if len(paired) else 'No paired predictions; inspect outcome-specific fitting gates and missing judgments'
        result.append(row)
    return result


def score_samples(points, weights, agreement_method='component_agreement_v1'):
    """Require every selected member for ICP, every configured endpoint overall."""
    from .prediction_agreement import combine, component_score
    result = []
    for (fold, ident), part in points.groupby(['fold', 'sample_id']):
        first = part.iloc[0]
        losses, baselines = {}, {}
        if part.icp_loss.notna().all():
            losses['icp'] = float(part.icp_loss.mean())
            baselines['icp'] = float(part.icp_baseline_loss.mean())
        for endpoint in BINARY_ENDPOINTS:
            value = first[f'{endpoint}_loss']
            if pd.notna(value):
                losses[endpoint] = float(value)
                baselines[endpoint] = float(first[f'{endpoint}_baseline_loss'])
        missing = [k for k in weights if k not in losses]
        loss = sum(weights[k]*losses[k] for k in weights) if not missing else None
        baseline = sum(weights[k]*baselines[k] for k in weights) if not missing else None
        result.append({'fold': fold, 'sample_id': ident, 'batch_id': first.batch_id,
            'composition': first.composition, 'element_count': int(first.element_count),
            'xrd_failure_probability':first.get('xrd_failure_probability'),
            'xrd_failure_observed':first.get('xrd_failure_observed'),
            'quantified_selected_elements': int(part.icp_loss.notna().sum()),
            **{f'{k}_loss': losses.get(k) for k in ENDPOINTS},
            **{f'{k}_agreement_score': component_score(k, losses[k]) if k in losses else None for k in ENDPOINTS},
            'missing_score_endpoints': ', '.join(missing), 'validation_loss': loss,
            'validation_score': combine(losses, weights, agreement_method), 'baseline_loss': baseline,
            'baseline_agreement_score': combine(baselines, weights, agreement_method),
            'agreement_method': agreement_method})
    return pd.DataFrame(result)
