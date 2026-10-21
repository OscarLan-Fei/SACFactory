"""Held-out regression fits, sample-balanced errors and probability diagnostics."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve


def regression_summary(frame,predicted,observed,groups=('model',)):
    rows=[]
    if frame.empty:return pd.DataFrame()
    for keys,part in frame.groupby(list(groups),sort=False):
        key=keys if isinstance(keys,tuple) else (keys,)
        row=dict(zip(groups,key));valid=part.dropna(subset=[predicted,observed]).copy()
        valid=valid.loc[np.isfinite(valid[predicted])&np.isfinite(valid[observed])]
        row.update(n=len(valid),specimens=valid.sample_id.nunique(),mae=None,rmse=None,predictive_r2=None,
                   bias=None,fit_slope=None,fit_intercept=None,fit_r2=None)
        if len(valid):
            counts=valid.groupby(['fold','sample_id']).sample_id.transform('size')
            w=(1/counts).to_numpy();p=valid[predicted].to_numpy();y=valid[observed].to_numpy();error=y-p
            row.update(mae=float(np.average(abs(error),weights=w)),rmse=float(np.sqrt(np.average(error**2,weights=w))),
                       bias=float(np.average(error,weights=w)))
            if len(y)>1 and np.ptp(y)>1e-12:row['predictive_r2']=float(r2_score(y,p,sample_weight=w))
            if len(y)>1 and np.ptp(p)>1e-12:
                design=np.column_stack([p,np.ones(len(p))]);slope,intercept=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
                row.update(fit_slope=float(slope),fit_intercept=float(intercept))
                if np.ptp(y)>1e-12:row['fit_r2']=float(r2_score(y,intercept+slope*p,sample_weight=w))
        rows.append(row)
    return pd.DataFrame(rows)


def parity_data(frame,summary,predicted,observed,limits=None):
    valid=frame.dropna(subset=[predicted,observed]).copy()
    valid=valid.loc[np.isfinite(valid[predicted])&np.isfinite(valid[observed])]
    if valid.empty:return pd.DataFrame()
    low,high=limits or (min(0.,float(valid[[predicted,observed]].min().min())),max(.01,float(valid[[predicted,observed]].max().max())*1.04))
    output=[]
    for row in valid.to_dict('records'):
        output.append({**row,'prediction':row[predicted],'measurement':row[observed],'kind':'Observation'})
    for row in summary.to_dict('records'):
        if not row['n']:continue
        for x in (low,high):
            output.append({'model':row['model'],'prediction':x,'measurement':x,'kind':'Identity'})
            if pd.notna(row['fit_slope']):output.append({'model':row['model'],'prediction':x,
                'measurement':row['fit_intercept']+row['fit_slope']*x,'kind':'OLS'})
    return pd.DataFrame(output)


def common_rows(points,models=None):
    """Use identical paired endpoint sets across all models shown."""
    if points.empty:return points.copy()
    frame=points.copy();models=list(frame.model.unique()) if models is None else list(models);sets=[];binary=[]
    for name in models:
        part=frame.loc[frame.model==name]
        p=part.dropna(subset=['predicted_w_wt_percent','observed_w_wt_percent'])
        sets.append(set(zip(p.fold,p.sample_id,p.element)))
        p=part.dropna(subset=['xrd_pass_predicted','xrd_pass_observed'])
        binary.append(set(zip(p.fold,p.sample_id)))
    paired=set.intersection(*sets) if sets else set();xrd=set.intersection(*binary) if binary else set()
    for i,row in frame.iterrows():
        if (row.fold,row.sample_id,row.element) not in paired:
            frame.loc[i,['predicted_w_wt_percent','observed_w_wt_percent']]=np.nan
            for name in ('predicted_total_w_wt_percent','observed_total_w_wt_percent'):
                if name in frame:frame.at[i,name]=np.nan
        if (row.fold,row.sample_id) not in xrd:frame.loc[i,['xrd_pass_predicted','xrd_pass_observed']]=np.nan
    return frame


def probability_diagnostics(points,bins=10):
    calibration=[];curves=[];summaries=[]
    if points.empty:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    data=points.drop_duplicates(['model','fold','sample_id']).dropna(subset=['xrd_pass_predicted','xrd_pass_observed'])
    for model,part in data.groupby('model',sort=False):
        p=part.xrd_pass_predicted.to_numpy();y=part.xrd_pass_observed.to_numpy();valid=(p>=0)&(p<=1)&np.isin(y,[0,1])
        p=p[valid];y=y[valid]
        if not len(y):continue
        summaries.append({'model':model,'n':len(y),'PASS':int(sum(y)),'FAIL':int(sum(1-y)),
            'brier':float(np.mean((p-y)**2)),'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'average_precision':float(average_precision_score(y,p)) if len(set(y))==2 else None})
        if len(set(y))==2:
            fpr,tpr,_=roc_curve(y,p)
            curves.extend({'model':model,'curve':'ROC','x':float(x),'y':float(z),'order':i} for i,(x,z) in enumerate(zip(fpr,tpr)))
            precision,recall,_=precision_recall_curve(y,p)
            curves.extend({'model':model,'curve':'Precision-recall','x':float(x),'y':float(z),'order':i} for i,(x,z) in enumerate(zip(recall,precision)))
        ids=np.minimum((p*bins).astype(int),bins-1)
        for index in sorted(set(ids)):
            mask=ids==index;n=int(mask.sum());rate=float(y[mask].mean());z=1.96
            center=(rate+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
            calibration.append({'model':model,'bin':int(index),'predicted_probability':float(p[mask].mean()),
                'observed_fraction':rate,'n':n,'lower':float(center-half),'upper':float(center+half)})
    return pd.DataFrame(summaries),pd.DataFrame(calibration),pd.DataFrame(curves)
