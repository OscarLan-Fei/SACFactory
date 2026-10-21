"""Shared multi-element regression with whole-batch masks and target descriptors."""
from collections import Counter

import numpy as np
import pandas as pd

from .chemical_descriptors import descriptor_matrix,reference
from .domain import ELEMENTS,recipe_key
from .loading import LoadingRegressor,MODELS,PROTOCOLS,loading_records,regression_metrics,PREDICTION_COLUMNS
from .planning import fingerprint
from .precursors import state_reference

VERSION='pooled-target-aware-ICP-v1'


def collect(samples,origin,elements,profile):
    targets=list(dict.fromkeys(elements))
    if not targets or set(targets)-set(ELEMENTS):
        raise ValueError('Select at least one valid ICP target element.')
    rows,audits=[],[]
    matrix=descriptor_matrix(profile)
    for element in targets:
        eligible,audit=loading_records(samples,origin,element,profile)
        audits.append(audit)
        identity=np.zeros(len(ELEMENTS));identity[ELEMENTS.index(element)]=1.
        for sample,w,x in eligible:
            target=np.r_[identity,matrix[ELEMENTS.index(element)],np.log(sample['concentrations_mM'][element]/.05)]
            rows.append({'sample':sample,'element':element,'w':w,'x':np.r_[x,target],'recipe':recipe_key(sample)})
    audit=pd.concat(audits,ignore_index=True) if audits else pd.DataFrame()
    totals={}
    for row in rows:totals[row['sample']['sample_id']]=totals.get(row['sample']['sample_id'],0.)+row['w']
    invalid={ident for ident,total in totals.items() if total>100+1e-9}
    if invalid:
        audit.loc[audit.sample_id.isin(invalid),['eligible','exclusion_reason']]=[False,'Quantified selected metal fractions sum above 100 wt%']
        rows=[r for r in rows if r['sample']['sample_id'] not in invalid]
    return rows,audit


def pooled_loading_validation(samples,origin,elements,family='Gaussian process',profile='chemistry_ion',
                              protocol='Batch holdout',max_folds=3,physics=None):
    from .weighted_validation import split_records
    from .physics_features import loading_prior,material_assumptions,compare_loading_predictions,physics_manifest
    if family not in MODELS or protocol not in PROTOCOLS:
        raise ValueError('Select a supported pooled ICP model and protocol.')
    targets=list(dict.fromkeys(elements))
    rows,audit=collect(samples,origin,targets,profile)
    priors={}
    if physics is not None:
        material_assumptions(physics)
        kept=[]
        for row in rows:
            ident=row['sample']['sample_id'];element=row['element']
            try:
                priors[(ident,element)]=loading_prior(row['sample'],element,physics)
                kept.append(row)
            except (ValueError,TypeError,KeyError,OverflowError) as exc:
                mask=(audit.sample_id==ident)&(audit.element==element)
                audit.loc[mask,['eligible','exclusion_reason']]=[False,'Physical prior: '+str(exc)]
        rows=kept
    mapped='Chronological batch replay' if protocol=='Chronological replay' else protocol
    splits=split_records(rows,samples,origin,targets,mapped,max(2,min(6,int(max_folds))))
    points,folds,skipped=[],[],[]
    for train,test,fold,_ in splits:
        training_ids={r['sample']['sample_id'] for r in train}
        training_batches={r['sample']['batch_id'] for r in train}
        test_batches={r['sample']['batch_id'] for r in test}
        assert not training_batches & test_batches
        if len(training_ids)<8 or len(training_batches)<2 or len({r['recipe'] for r in train})<3:
            skipped.append({'fold':fold,'reason':'Need 8 training specimens, 2 batches and 3 recipes'})
            continue
        x=np.asarray([r['x'] for r in train])
        if not np.any(np.ptp(x,axis=0)>1e-12):
            skipped.append({'fold':fold,'reason':'Training controls and enabled target descriptors are constant'})
            continue
        y=[r['w'] for r in train];xt=[r['x'] for r in test]
        direct=LoadingRegressor(family).fit(x,y)
        p,low,high,outside=direct.predict(xt)
        direct_values=p.copy()
        if physics is not None:
            train_prior=[priors[(r['sample']['sample_id'],r['element'])] for r in train]
            test_prior=[priors[(r['sample']['sample_id'],r['element'])] for r in test]
            fitted=LoadingRegressor(family).fit(x,y,train_prior)
            p,low,high,outside=fitted.predict(xt,test_prior)
        source_elements=sorted({r['element'] for r in train})
        folds.append({'fold':fold,'train_sample_ids':sorted(training_ids),
            'test_sample_ids':sorted({r['sample']['sample_id'] for r in test}),
            'train_batch_ids':sorted(training_batches),'test_batch_ids':sorted(test_batches),
            'train_elements':source_elements,'test_elements':sorted({r['element'] for r in test}),
            'training_target_pairs':len(train),'training_specimens':len(training_ids),
            'train_pairs':[[r['sample']['sample_id'],r['element']] for r in train],
            'test_pairs':[[r['sample']['sample_id'],r['element']] for r in test]})
        for index,(r,predicted,lo,hi,out) in enumerate(zip(test,p,low,high,outside)):
            sample=r['sample'];element=r['element']
            row={'sample_id':sample['sample_id'],'batch_id':sample['batch_id'],'element':element,
                'composition':'+'.join(sample['elements']),'element_count':len(sample['elements']),
                'fold':fold,'model':family,'observed_w_wt_percent':r['w'],'predicted_w_wt_percent':float(predicted),
                'lower_95_w_wt_percent':float(lo) if np.isfinite(lo) else None,
                'upper_95_w_wt_percent':float(hi) if np.isfinite(hi) else None,
                'training_n':len(training_ids),'training_target_pairs':len(train),
                'outside_training_range':bool(out),'target_seen_in_training':element in source_elements}
            if physics is not None:
                row.update(direct_predicted_w_wt_percent=float(direct_values[index]),
                    physical_prior_w_wt_percent=priors[(sample['sample_id'],element)])
            points.append(row)
    frame=pd.DataFrame(points) if points else pd.DataFrame(columns=PREDICTION_COLUMNS+['model','target_seen_in_training'])
    if len(frame) and frame.duplicated(['sample_id','element']).any():
        raise RuntimeError('A pooled ICP target was evaluated more than once.')
    per_element=[]
    for element in targets:
        part=frame.loc[frame.element==element]
        eligible=sum(r['element']==element for r in rows)
        per_element.append({'element':element,'eligible_targets':eligible,'evaluated_targets':len(part),
            **regression_metrics(part),'status':'Evaluated' if len(part) else 'No evaluable targets; inspect audit and skipped folds'})
    report=pd.DataFrame(per_element)
    seen=set(zip(frame.sample_id,frame.element))
    if len(audit):
        audit['evaluated']=[(r.sample_id,r.element) in seen for r in audit.itertuples()]
        audit['coverage_reason']=['Held-out prediction' if r.evaluated else r.exclusion_reason or 'No evaluable training fold' for r in audit.itertuples()]
    metrics=regression_metrics(frame)
    metrics['physical_specimens']=int(frame.sample_id.nunique())
    metrics['equal_element_mae_wt_percent']=float(report.mae_wt_percent.dropna().mean()) if report.mae_wt_percent.notna().any() else None
    if len(frame):
        errors=frame.assign(absolute_error=(frame.predicted_w_wt_percent-frame.observed_w_wt_percent).abs())
        metrics['equal_specimen_mae_wt_percent']=float(errors.groupby('sample_id').absolute_error.mean().mean())
    info={'version':VERSION,'origin':origin,'elements':targets,'model':family,'profile':profile,'protocol':protocol,
        'metrics':metrics,'eligible_targets':len(rows),'eligible_specimens':len({r['sample']['sample_id'] for r in rows}),
        'recorded_specimens':sum(s.get('origin')==origin for s in samples),
        'folds':folds,'skipped_folds':skipped,'dataset_fingerprint':fingerprint(samples),
        'descriptor_snapshot_sha256':reference()['snapshot_sha256'],'state_snapshot_sha256':state_reference()['snapshot_sha256'],
        'physics':physics_manifest(profile),'physical_residual':physics,
        'paired_comparison':compare_loading_predictions(frame),'per_element':per_element,
        'exclusions':dict(Counter(audit.loc[~audit.eligible,'exclusion_reason'])) if len(audit) else {},
        'note':'One shared target-aware statistical regressor per model, process context and outer fold. All selected element targets train together. Whole batches, including every mixture member, are held out together; scalers and active features fit training only. Compatible contexts stay separate. Each measured target has at most one held-out prediction. Pooled R-squared reflects the combined loading range; inspect per-element errors and unseen-target flags. log1p(W) predictions are transformed centers. These independent conditional targets do not enforce joint mixture mass closure; PulseICP provides a coupled mass decoder. Missing/ND/BLQ and burnout W are not imputed.'}
    return frame,audit,info
