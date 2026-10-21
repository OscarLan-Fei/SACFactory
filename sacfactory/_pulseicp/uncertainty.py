"""Decoded predictive mixture summaries and empirical distributional metrics."""
import math
import numpy as np


def summarize(draws):
    """Draw shape: ensemble member, recipe, draw within member."""
    samples = np.asarray(draws, dtype=float)
    if samples.ndim != 3 or not np.isfinite(samples).all() or samples.shape[2] < 2:
        raise ValueError('Finite decoded ensemble draws are required.')
    member_means = samples.mean(axis=2)
    mean = member_means.mean(axis=0)
    within = samples.var(axis=2).mean(axis=0)
    between = member_means.var(axis=0)
    pooled = samples.transpose(1, 0, 2).reshape(samples.shape[1], -1)
    low, high = np.quantile(pooled, [.025, .975], axis=1)
    return {'mean': mean, 'sd': np.sqrt(within+between), 'within_sd': np.sqrt(within),
            'between_sd': np.sqrt(between), 'low': low, 'high': high, 'pooled': pooled}


def empirical_crps(draws, observed):
    draws = np.sort(np.asarray(draws), axis=1)
    n = draws.shape[1]
    coefficient = (2*np.arange(1, n+1)-n-1)/(n*n)
    return np.mean(np.abs(draws-np.asarray(observed)[:, None]), axis=1)-draws@coefficient


def metrics(frame, draws=None):
    if frame.empty:
        return {'n': 0}
    y = frame.observed_w_wt_percent.to_numpy()
    p = frame.predicted_mean_w_wt_percent.to_numpy()
    error = p-y
    weights = np.ones(len(frame))
    if 'sample_id' in frame:
        weights = 1/frame.groupby('sample_id').sample_id.transform('size').to_numpy()
    weights = weights/weights.sum()
    denominator = np.sum(weights*(y-np.sum(weights*y))**2)
    result = {'n': len(frame), 'mae_wt_percent': float(np.sum(weights*np.abs(error))),
              'rmse_wt_percent': float(np.sqrt(np.sum(weights*error**2))),
              'r_squared': float(1-np.sum(weights*error**2)/denominator) if len(y)>1 and denominator>1e-12 else None}
    valid = frame.dropna(subset=['lower_95_w_wt_percent', 'upper_95_w_wt_percent'])
    result.update(interval_n=len(valid), nominal_95_coverage=float(((valid.observed_w_wt_percent>=valid.lower_95_w_wt_percent)&
        (valid.observed_w_wt_percent<=valid.upper_95_w_wt_percent)).mean()) if len(valid) else None,
        mean_interval_width_wt_percent=float((valid.upper_95_w_wt_percent-valid.lower_95_w_wt_percent).mean()) if len(valid) else None)
    result['crps_wt_percent'] = float(np.sum(weights*empirical_crps(draws, y))) if draws is not None else None
    result['specimens_n'] = int(frame.sample_id.nunique()) if 'sample_id' in frame else len(frame)
    result['metric_weighting'] = 'Equal physical specimens, equal elements within specimen'
    if len(valid):
        vw = weights[frame.index.get_indexer(valid.index)]
        vw = vw/vw.sum()
        result['nominal_95_coverage'] = float(np.sum(vw*((valid.observed_w_wt_percent>=valid.lower_95_w_wt_percent)&(valid.observed_w_wt_percent<=valid.upper_95_w_wt_percent))))
        result['mean_interval_width_wt_percent'] = float(np.sum(vw*(valid.upper_95_w_wt_percent-valid.lower_95_w_wt_percent)))
    return result


def batch_calibration(low, high, y, batches, coverage=.95):
    scores = np.maximum.reduce([np.asarray(low)-y, y-np.asarray(high), np.zeros(len(y))])
    blocks = [float(np.max(scores[np.asarray(batches)==b])) for b in sorted(set(batches))]
    rank = math.ceil((len(blocks)+1)*coverage)
    if rank > len(blocks):
        return {'gap_wt_percent': None, 'batch_n': len(blocks), 'status': 'INSUFFICIENT_CALIBRATION_BATCHES',
                'note': 'At 95%, at least 19 exchangeable calibration batches are needed for a finite batch-max quantile.'}
    return {'gap_wt_percent': float(np.sort(blocks)[rank-1]), 'batch_n': len(blocks),
            'status': 'BATCH_MAX_SPLIT_CALIBRATION',
            'note': 'Whole-batch maximum residual quantile. Exchangeability and supported input scope are assumptions; held-out coverage is reported. No guarantee for new elements or shifted protocols.'}
