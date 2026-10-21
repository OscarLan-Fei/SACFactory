"""Reusable multi-target selection and auditable independent calculation reports."""
import json

import pandas as pd
import streamlit as st


def select_targets(label,options,format_func,key,default_all=False):
    if st.button('Select all: '+label,key=key+'_all'):
        st.session_state[key]=list(options)
    if key in st.session_state:
        st.session_state[key]=[v for v in st.session_state[key] if v in options]
    return st.multiselect(label,options,default=None if key in st.session_state else list(options if default_all else options[:1]),
        format_func=format_func,key=key)


def detail(results,key):
    """Every selected job is calculated; detail selection only changes inspection."""
    if isinstance(results,tuple):
        return results
    if len(results)==1:
        return next(iter(results.values()))
    summaries=[];frames=[]
    for label,(frame,info) in results.items():
        summaries.append({'calculation':label,'prediction_rows':len(frame),
            'status':info.get('batch_status','Evaluated' if len(frame) else 'No evaluable rows'),
            'reason':info.get('reason','')})
        if len(frame):frames.append(frame.assign(calculation=label))
    st.write('**All selected calculations**')
    st.dataframe(pd.DataFrame(summaries),hide_index=True,width='stretch')
    if frames:
        st.download_button('Download all selected calculation results',pd.concat(frames,ignore_index=True).to_csv(index=False).encode(),
            key+'_predictions.csv','text/csv',key=key+'_csv')
    st.download_button('Download all selected calculation provenance',json.dumps(
        {name:info for name,(_,info) in results.items()},indent=2,allow_nan=False).encode(),
        key+'_provenance.json','application/json',key=key+'_json')
    label=st.selectbox('Inspect calculated target',list(results),key=key+'_inspect')
    return results[label]


def run_jobs(jobs,callback):
    """Preserve per-target failures instead of aborting or hiding other targets."""
    results={}
    bar=st.progress(0.)
    try:
        for index,(label,args) in enumerate(jobs,1):
            try:
                results[label]=callback(args)
            except (ValueError,KeyError,TypeError,RuntimeError) as exc:
                results[label]=(pd.DataFrame(),{'batch_status':'Unavailable','reason':str(exc)})
            bar.progress(index/max(1,len(jobs)),text=f'Completed target {index} / {len(jobs)}')
    finally:
        bar.empty()
    return next(iter(results.values())) if len(results)==1 else results
