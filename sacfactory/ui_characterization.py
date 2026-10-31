"""Reviewable SEM/XRD/STEM evidence inside the discovery feedback loop."""
import hashlib
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from .discovery.common import digest,identity
from .discovery.characterization import (OBSERVATIONS,read_image,xrd_analysis,microscopy_analysis,
    register_artifact,review_artifact,evidence_summary,commit_assessment)
from .domain import ELEMENTS
from .ui_elements import element_multiselect


def render_characterization(store,samples,ledger,context):
    prefix='discovery_characterization_'+store.origin
    st.write('**SEM, XRD and STEM evidence**')
    st.caption('Upload measurements, inspect candidate features, review observations, then return a structural screening assessment to the experiment loop.')
    selected=element_multiselect('Characterization elements',ELEMENTS,default=context.get('elements',[]),key=prefix+'_elements')
    choices=[s for s in samples if set(s['elements'])&set(selected)]
    if not choices:st.info('Select elements with recorded samples.');return
    ident=st.selectbox('Characterization sample',[s['sample_id'] for s in choices],key=prefix+'_sample')
    sample=next(s for s in choices if s['sample_id']==ident)
    modality=st.selectbox('Measurement type',['XRD','STEM','SEM'],key=prefix+'_modality')
    types=['csv','txt','xy','dat'] if modality=='XRD' else ['png','tif','tiff','jpg','jpeg']
    uploaded=st.file_uploader('Measurement export',type=types,key=prefix+'_file_'+modality)
    st.caption('XRD: two numeric columns (2theta in degrees, intensity). Images: exported single frames; crop out scale bars, labels and detector overlays. Native instrument formats are not decoded here.')
    params={};references=[]
    if modality=='XRD':
        cols=st.columns(3)
        params['prominence']=cols[0].number_input('Relative peak prominence',.001,1.,.05,format='%.3f',key=prefix+'_prominence')
        params['min_width_deg']=cols[1].number_input('Minimum peak width / deg',.001,10.,.05,format='%.3f',key=prefix+'_width')
        params['tolerance_deg']=cols[2].number_input('Reference tolerance / deg',.001,2.,.2,format='%.3f',key=prefix+'_tolerance')
        wavelength_verified=st.checkbox('X-ray wavelength verified from measurement metadata',key=prefix+'_wavelength_verified')
        params['wavelength_angstrom']=st.number_input('Measurement wavelength / angstrom',.01,10.,1.5406,format='%.5f',key=prefix+'_wavelength') if wavelength_verified else None
        text=st.text_area('Optional sourced reference peaks (CSV)',value='phase,two_theta_deg,source,wavelength_angstrom\n',key=prefix+'_references')
        frame=pd.read_csv(io.StringIO(text))
        if len(frame):
            required=['phase','two_theta_deg','source','wavelength_angstrom']
            if not set(required)<=set(frame.columns):raise ValueError('Reference columns: phase, two_theta_deg, source, wavelength_angstrom.')
            if frame[required].isna().any().any():raise ValueError('Reference values, sources and wavelengths cannot be missing.')
            references=frame.to_dict('records')
        params['references']=references
    else:
        calibrated=st.checkbox('Pixel size calibrated from instrument metadata or scale bar',key=prefix+'_calibrated')
        params['nm_per_pixel']=st.number_input('Pixel size / nm per pixel',.00001,10000.,.02 if modality=='STEM' else 1.,format='%.5f',key=prefix+'_scale_'+modality) if calibrated else None
        cols=st.columns(2)
        params['sigma_px']=cols[0].number_input('Candidate feature scale / pixels',.5,20.,1.5,key=prefix+'_sigma_'+modality)
        params['threshold_sigma']=cols[1].number_input('Contrast threshold / noise scale',1.,20.,4.,key=prefix+'_threshold_'+modality)
        params['min_area_px']=st.number_input('Minimum SEM feature area / pixels',1,100000,3,key=prefix+'_minarea')
        params['max_area_px']=st.number_input('Maximum SEM feature area / pixels',1,1000000,10000,key=prefix+'_maxarea')
        if uploaded:
            image=read_image(uploaded.getvalue());height,width=image.shape
            roi_text=st.text_input('Analysis ROI: x0,y0,x1,y1 (blank = entire image)',key=prefix+'_roi_'+modality)
            params['roi']=[int(v.strip()) for v in roi_text.split(',')] if roi_text.strip() else [0,0,width,height]
            display=(image-image.min())/max(float(np.ptp(image)),1e-12)
            st.image(display,caption='Measurement export (display contrast normalized; detection uses the original intensities)',clamp=True,width='stretch')
    acquisition=st.text_input('Acquisition date / instrument / field identifier (optional)',key=prefix+'_acquisition')
    content=uploaded.getvalue() if uploaded else b''
    signature=digest([ident,modality,hashlib.sha256(content).hexdigest(),params,acquisition])
    if st.button('Extract candidate features',disabled=not uploaded,key=prefix+'_analyze'):
        if modality=='XRD':analysis,plot=xrd_analysis(content,**params)
        else:analysis=microscopy_analysis(content,modality,**params);plot=None
        st.session_state[prefix+'_result']={'signature':signature,'analysis':analysis,'plot':plot}
    saved=st.session_state.get(prefix+'_result')
    if saved and saved['signature']==signature:
        analysis=saved['analysis'];st.info(analysis['interpretation'])
        if saved['plot']:
            plot=pd.DataFrame(saved['plot']);st.line_chart(plot.set_index('two_theta_deg'))
            st.dataframe(pd.DataFrame(analysis['peaks']),hide_index=True,width='stretch')
            if analysis['reference_matches']:st.dataframe(pd.DataFrame(analysis['reference_matches']),hide_index=True,width='stretch')
        else:
            st.metric('Candidate features',analysis['candidate_count'])
            features=pd.DataFrame(analysis['features'])
            if len(features):
                st.scatter_chart(features,x='x_px',y='y_px',height=360)
                st.caption('Pixel coordinates are tabulated below; this scatter is not an elemental map. Image y increases downward.')
                st.dataframe(features,hide_index=True,width='stretch')
        st.download_button('Export candidate analysis',json.dumps(analysis,indent=2).encode(),'characterization_candidates.json','application/json',key=prefix+'_export')
        already=st.session_state.get(prefix+'_saved')==signature
        if st.button('Save source and analysis for review',disabled=already,key=prefix+'_save'):
            register_artifact(store,ledger,ident,modality,uploaded.name,content,analysis,{'acquisition':acquisition})
            st.session_state[prefix+'_saved']=signature;st.success('Saved locally; no experimental label was changed.')
            sample=store.get('sample',ident)
    artifacts=[a for a in ledger.records('characterization') if a['record_key']==identity(sample)]
    st.divider();st.write('**Review observations**')
    if not artifacts:st.info('No characterization files are linked to this sample.');return
    lookup={a['id']:a for a in artifacts}
    artifact_id=st.selectbox('Saved evidence',list(lookup),format_func=lambda key:lookup[key]['modality']+' · '+lookup[key]['file_name']+' · '+lookup[key]['created'],key=prefix+'_artifact')
    artifact=lookup[artifact_id]
    observation=st.selectbox('Reviewed observation',OBSERVATIONS[artifact['modality']],key=prefix+'_observation_'+artifact['modality'],index=len(OBSERVATIONS[artifact['modality']])-1)
    reviewer=st.text_input('Reviewer',key=prefix+'_reviewer')
    notes=st.text_area('Observation notes and limitations',key=prefix+'_notes')
    if st.button('Record reviewed observation',key=prefix+'_review'):
        review_artifact(store,ledger,artifact_id,observation,reviewer,notes);st.success('Observation recorded. Review the combined evidence below.')
    summary=evidence_summary(sample,ledger)
    if summary['invalid_artifact_ids']:st.warning('Some reviewed source files are missing or changed; they are excluded from the combined evidence.')
    st.write('**Combined reviewed evidence**')
    st.write('Modalities: '+', '.join(summary['modalities'])+' · Suggested screening state: '+summary['recommended_screening_state'])
    st.write(', '.join(summary['observations']) or 'Awaiting review')
    if summary['conflict']:st.warning('Conflicting structural observations require an explicit interpretation or additional characterization.')
    if summary['morphology_alert']:st.warning('SEM aggregates were reported. Review their identity and obtain targeted structural evidence; SEM alone does not resolve the XRD/STEM label.')
    st.caption(summary['interpretation'])
    state=st.selectbox('Final screening assessment',['PENDING','PASS','FAIL'],key=prefix+'_state')
    criteria=st.text_area('Assessment rationale / screening criteria',key=prefix+'_criteria')
    override=st.checkbox('Explicitly resolve conflicting evidence or replace an existing judgment',key=prefix+'_override')
    if st.button('Commit reviewed assessment to Loop',disabled=not summary['review_ids'],key=prefix+'_commit'):
        event=commit_assessment(store,ledger,ident,state,reviewer,criteria,override)
        st.success('Committed '+event['state']+'. Evolution and subsequent model fits use this reviewed outcome. Existing forecasts remain frozen.')
        st.caption('Run protected validation and train a versioned candidate under Set model & updates when ready; training is not started by this button.')
