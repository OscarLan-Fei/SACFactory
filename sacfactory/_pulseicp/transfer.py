"""Protected cold-element and single-to-mixture validation protocols."""
import numpy as np
from .data import PreparedData, batch_split

TRANSFER_SPLITS = ['Unseen element transfer', 'Singles to mixtures']


def protected_split(data, seed, protocol, held_element=None, source_elements=None):
    if protocol not in TRANSFER_SPLITS:
        return batch_split(data, seed, protocol)
    if protocol == TRANSFER_SPLITS[0] and not held_element:
        raise ValueError('Choose an entirely held-out target element.')
    if source_elements is not None and not source_elements:
        raise ValueError('Select at least one source element for transfer evaluation.')
    sources = set(source_elements or [])
    target = [i for i,r in enumerate(data.records) if
        (held_element in r['composition'].elements if protocol == TRANSFER_SPLITS[0]
         else len(r['composition'].elements)>1)]
    if not target:
        raise ValueError('No eligible target compositions for this transfer test.')
    target_batches = {str(data.records[i]['raw']['batch_id']) for i in target}
    target_set = set(target)
    candidate = [i for i,r in enumerate(data.records) if i not in target_set and
        (not sources or set(r['composition'].elements)<=sources)]
    source = [i for i in candidate if str(data.records[i]['raw']['batch_id']) not in target_batches]
    if not source:
        raise ValueError('No independent source batches remain after target-batch purging.')
    subset = PreparedData([data.records[i] for i in source], data.audit,
        data.x[source], data.delta[source], data.w[source])
    (train, cal, unused), info = batch_split(subset, seed)
    indices = [np.array([source[i] for i in ix], dtype=int) for ix in (train, cal)]
    indices.append(np.asarray(target, dtype=int))
    info[2]['role'] = 'unused_source_holdout'
    info.append({'role':'test', 'batch_ids':sorted(target_batches),
        'sample_ids':sorted({data.records[i]['raw']['sample_id'] for i in target})})
    purged = sorted(set(candidate)-set(source))
    info.append({'role':'excluded_target_batch_sources',
        'sample_ids':sorted({data.records[i]['raw']['sample_id'] for i in purged}),
        'batch_ids':sorted({str(data.records[i]['raw']['batch_id']) for i in purged})})
    return indices, info
