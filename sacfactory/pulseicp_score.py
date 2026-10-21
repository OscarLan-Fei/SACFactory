"""Cross-fitted learned ICP contributions for the unchanged synthesis utility."""
import numpy as np
from sklearn.model_selection import GroupKFold

from . import pulseicp_bridge as bridge


def loading_baselines(train,test,origin,family,settings,weight,target,strength,score_definition='capped_attainment_v1'):
    """No score-training recipe or its batch enters its own loading model."""
    settings=bridge.configuration()['model'] if settings is None else dict(settings)
    groups=[r['sample']['batch_id'] for r in train]
    batches=set(groups)
    values={};exclusions=[];folds=[]
    info={'family':family,'settings':settings,'inner_folds':folds,'exclusions':exclusions,
        'outer_model':None,'training_baseline':'Whole-batch out-of-fold predictions',
        'utility':'100 * ICP weight * strength * mean_draw loading_utility(total decoded W)',
        'score_definition':score_definition,
        'uncertainty':'Loading draws are provisional; no calibrated system-score interval is claimed.'}
    def rejected(audit):
        return audit.loc[~audit.eligible,['sample_id','batch_id','element','exclusion_reason']].to_dict('records')
    def predict(model,rows,scope):
        from .compute_runtime import batched_predict, emit
        emit({'stage':scope,'samples':len(rows)})
        def forecast(m,s,o):
            return bridge.predict_attainment(m,s,o,target,
                **({} if score_definition=='capped_attainment_v1' else {'score_definition':score_definition}))
        def error(sample,exc):
            exclusions.append({'sample_id':sample['sample_id'],'scope':scope,'reason':str(exc)})
        for v in batched_predict(model,[r['sample'] for r in rows],origin,forecast,error):
            values[v['sample_id']]={**v,'contribution':100*weight*strength*v['expected_attainment']}
    if len(batches)<3:
        info['reason']='Learned score fusion needs at least 3 outer-training batches for protected inner fitting.'
        return values,info
    for index,(source,held) in enumerate(GroupKFold(n_splits=min(3,len(batches))).split(train,groups=groups),1):
        from .compute_runtime import emit
        emit({'stage':f'Inner loading fold {index} / {min(3,len(batches))}'})
        source_rows=[train[i] for i in source];held_rows=[train[i] for i in held]
        record={'inner_fold':index,'train_sample_ids':[r['sample']['sample_id'] for r in source_rows],
            'held_sample_ids':[r['sample']['sample_id'] for r in held_rows],
            'train_batches':sorted({groups[i] for i in source}),
            'held_batches':sorted({groups[i] for i in held})}
        assert not set(record['train_batches'])&set(record['held_batches'])
        try:
            model,audit=bridge.fit([r['sample'] for r in source_rows],origin,family,settings)
            record.update(model_id=model.model_id,actual_training_ids=model.training_ids,
                model_settings=model.settings,eligible_target_rows=int(audit.eligible.sum()),excluded_training_targets=rejected(audit))
            predict(model,held_rows,'Inner held-out training baseline')
        except (ValueError,TypeError,KeyError,OverflowError,RuntimeError) as exc:
            record['reason']=str(exc)
        folds.append(record)
    try:
        model,audit=bridge.fit([r['sample'] for r in train],origin,family,settings)
        info['outer_model']={'model_id':model.model_id,'training_ids':model.training_ids,
            'physics':model.physics,'model_settings':model.settings,'eligible_target_rows':int(audit.eligible.sum()),
            'excluded_training_targets':rejected(audit)}
        predict(model,test,'Outer held-out prediction')
    except (ValueError,TypeError,KeyError,OverflowError,RuntimeError) as exc:
        info['reason']=str(exc)
    info['covered_training_samples']=sum(r['sample']['sample_id'] in values for r in train)
    info['covered_test_samples']=sum(r['sample']['sample_id'] in values for r in test)
    return values,info
