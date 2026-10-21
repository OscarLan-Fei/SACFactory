"""Probability quality and decision trade-offs on eligible saved forecasts."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)


def probability_evaluation(records, threshold=.5):
    data = records.loc[records.eligible].dropna(subset=['label', 'p_screen_pass'])
    data = data.loc[data.label.isin([0, 1]) & data.p_screen_pass.between(0, 1)]
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between zero and one.')
    if data.empty:
        return {}, pd.DataFrame()
    y, p = data.label.to_numpy(dtype=int), data.p_screen_pass.to_numpy()
    positive = p >= threshold
    tp, fp = int(((y == 1) & positive).sum()), int(((y == 0) & positive).sum())
    two_classes = len(set(y)) == 2
    metrics = {'n': len(y), 'threshold': threshold, 'brier': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, p, labels=[0, 1])), 'observed_prevalence': float(y.mean()),
        'roc_auc': float(roc_auc_score(y, p)) if two_classes else None,
        'average_precision': float(average_precision_score(y, p)) if two_classes else None,
        'pass_precision': tp / int(positive.sum()) if positive.any() else None,
        'pass_recall': tp / int(y.sum()) if y.sum() else None,
        'false_pass_fraction': fp / int(positive.sum()) if positive.any() else None,
        'predicted_pass_n': int(positive.sum())}
    if two_classes:
        precision, recall, cutoffs = precision_recall_curve(y, p)
        curve = pd.DataFrame({'precision': precision, 'recall': recall,
                              'threshold': np.r_[cutoffs, np.nan]})
    else:
        curve = pd.DataFrame(columns=['precision', 'recall', 'threshold'])
    return metrics, curve
