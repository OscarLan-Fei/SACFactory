"""Independent observed/predicted outcome indices, not prediction-error scores."""
import math
import numpy as np
import pandas as pd

VERSION='icp-xrd-outcome-index-v1'


def loading_coordinate(w,reference=2.5):
    """Smooth coordinate on physical mass percent [0,100], without a hard cap."""
    if not math.isfinite(reference) or reference<=0:raise ValueError('Loading reference must be positive.')
    values=np.asarray(w,dtype=float)
    return values/(values+reference)*(100+reference)/100


def score_pairs(points,weight=.8,reference=2.5):
    """Use the same fixed formula separately on predicted and measured endpoints.

    The prediction is a plug-in index of OOF means/probabilities, not an
    estimate of E[index] integrated over joint predictive draws. Whole-recipe
    totals require every member. Missing endpoints never trigger reweighting.
    """
    if not math.isfinite(weight) or not 0<=weight<=1:raise ValueError('ICP outcome weight must lie in [0,1].')
    loading_coordinate(0,reference)
    frame=points.copy()
    if frame.empty:return pd.DataFrame(columns=['model','fold','sample_id','batch_id','predicted_outcome_score','observed_outcome_score','status'])
    if 'model' not in frame:frame['model']='SAC-Gen'
    compositions={}
    if 'composition' in frame:
        for row in frame[['fold','sample_id','composition']].itertuples(index=False):
            if isinstance(row.composition,str) and row.composition.strip():
                compositions[(row.fold,row.sample_id)]=row.composition
    output=[]
    for (model,fold,ident),part in frame.groupby(['model','fold','sample_id'],sort=False):
        row=part.iloc[0]
        composition=row.get('composition')
        if not isinstance(composition,str) or not composition.strip():
            composition=compositions.get((fold,ident),'+'.join(part.element) if 'composition' not in frame else '')
        expected=set(composition.split('+')) if composition else set(part.element)
        actual=set(part.element)
        totals={}
        for side in ('predicted','observed'):
            total_key=side+'_total_w_wt_percent'
            values=part.get(total_key,pd.Series(dtype=float)).dropna()
            if len(values) and values.nunique()==1:total=float(values.iloc[0])
            elif actual==expected and not part.element.duplicated().any() and part[side+'_w_wt_percent'].notna().all():
                total=float(part[side+'_w_wt_percent'].sum())
            else:total=None
            totals[side]=total
        p=row.get('xrd_pass_predicted');y=row.get('xrd_pass_observed')
        state='PAIRED';predicted=observed=None
        if any(v is None or not np.isfinite(v) for v in [totals['predicted'],totals['observed'],p,y]):state='INCOMPLETE_ENDPOINTS'
        elif not all(0<=v<=100 for v in totals.values()) or not 0<=p<=1 or y not in (0,1):state='OUTSIDE_PHYSICAL_BOUNDS'
        elif part.xrd_pass_predicted.dropna().nunique()>1 or part.xrd_pass_observed.dropna().nunique()>1:state='CONFLICTING_XRD_ENDPOINTS'
        else:
            predicted=100*(weight*float(loading_coordinate(totals['predicted'],reference))+(1-weight)*float(p))
            observed=100*(weight*float(loading_coordinate(totals['observed'],reference))+(1-weight)*float(y))
        output.append({'model':model,'fold':fold,'sample_id':ident,'batch_id':row.get('batch_id',''),
            'composition':composition,'predicted_outcome_score':predicted,'observed_outcome_score':observed,
            'predicted_total_w_wt_percent':totals['predicted'],'observed_total_w_wt_percent':totals['observed'],
            'xrd_pass_predicted':p,'xrd_pass_observed':y,'recorded_xrd':'PASS' if y==1 else 'FAIL' if y==0 else 'Unknown',
            'status':state,'score_version':VERSION,'icp_weight':weight,'loading_reference_wt_percent':reference})
    return pd.DataFrame(output)


def shared_score_support(scores):
    """Flag the same complete physical specimens for multi-model comparisons."""
    frame=scores.copy()
    if frame.empty:
        frame['comparison_common']=pd.Series(dtype=bool)
        return frame
    support=[]
    for _,part in frame.groupby('model',sort=False):
        valid=part.loc[part.status=='PAIRED']
        support.append(set(zip(valid.fold,valid.sample_id)))
    common=set.intersection(*support)
    frame['comparison_common']=[(row.fold,row.sample_id) in common for row in frame.itertuples()]
    return frame
