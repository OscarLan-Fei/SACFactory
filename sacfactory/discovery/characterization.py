"""Source-preserving characterization triage and reviewed experimental feedback.

The local detectors extract candidate features. They do not certify atomic
dispersion or provide calibrated chemical-class probabilities.
"""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
from uuid import uuid4

import numpy as np
from scipy import ndimage, signal
from scipy.spatial import cKDTree

from .common import digest, identity, query, recipe_id, stamp

ALGORITHM='characterization-triage-v1'
OBSERVATIONS={
    'SEM':['MORPHOLOGY_RECORDED','AGGREGATES_OBSERVED','INCONCLUSIVE'],
    'XRD':['NO_RESOLVED_TARGET_CRYSTALLINE_PEAKS','TARGET_CRYSTALLINE_PHASE_DETECTED','INCONCLUSIVE'],
    'STEM':['ISOLATED_SITE_CANDIDATES_CONFIRMED','CLUSTERS_OBSERVED','INCONCLUSIVE']}


def positive(value,name):
    number=float(value)
    if not math.isfinite(number) or number<=0:raise ValueError(name+' must be positive and finite.')
    return number


def read_pattern(content):
    """Read exported two-column 2theta/intensity, not proprietary instrument data."""
    if len(content)>20_000_000:raise ValueError('Use an exported pattern smaller than 20 MB.')
    text=content.decode('utf-8-sig');rows=[];started=False
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith(('#',';','!')):continue
        fields=re.split(r'[,;\s]+',line)
        try:
            if len(fields)!=2:raise ValueError()
            pair=tuple(float(v) for v in fields)
        except ValueError:
            if started:raise ValueError('Use exactly two numeric columns: two_theta_deg and intensity.') from None
            continue
        rows.append(pair);started=True
    array=np.asarray(rows,dtype=float)
    if len(rows)<8 or len(rows)>200000 or not np.isfinite(array).all():raise ValueError('A pattern needs 8-200000 finite points.')
    if np.any(np.diff(array[:,0])<=0) or np.any((array[:,0]<=0)|(array[:,0]>=180)):
        raise ValueError('2theta must increase strictly between 0 and 180 degrees.')
    return array


def xrd_analysis(content,prominence=.05,min_width_deg=.05,references=None,tolerance_deg=.2,wavelength_angstrom=None):
    array=read_pattern(content);x,y=array.T
    prominence=positive(prominence,'Relative prominence')
    if prominence>1:raise ValueError('Relative prominence must be <= 1.')
    min_width_deg=positive(min_width_deg,'Minimum peak width')
    tolerance_deg=positive(tolerance_deg,'Reference tolerance')
    wavelength=None if wavelength_angstrom is None else positive(wavelength_angstrom,'X-ray wavelength')
    # Uniform resampling is used only for descriptive signal processing.
    grid=np.linspace(x[0],x[-1],len(x));raw=np.interp(grid,x,y)
    baseline=ndimage.percentile_filter(raw,percentile=15,size=min(501,max(5,2*(len(x)//50)+1)))
    corrected=np.maximum(raw-baseline,0)
    smooth=ndimage.gaussian_filter1d(corrected,.75)
    span=float(np.ptp(raw));step=float(grid[1]-grid[0])
    peaks,properties=signal.find_peaks(smooth,prominence=prominence*span,width=max(1,min_width_deg/step)) if span>0 else ([],{})
    rows=[]
    for i,p in enumerate(peaks):
        rows.append({'two_theta_deg':float(grid[p]),'prominence':float(properties['prominences'][i]),
                     'width_deg':float(properties['widths'][i]*step),'intensity':float(raw[p])})
    matches=[]
    for reference in references or []:
        phase=str(reference.get('phase','')).strip();source=str(reference.get('source','')).strip()
        position=positive(reference['two_theta_deg'],'Reference 2theta')
        if not phase or not source or position>=180:raise ValueError('Reference peaks require a phase, 2theta and source.')
        reference_wavelength=reference.get('wavelength_angstrom')
        if wavelength is None or reference_wavelength is None or not math.isclose(wavelength,positive(reference_wavelength,'Reference wavelength'),rel_tol=1e-5):
            raise ValueError('2theta matching requires explicitly verified, matching measurement and reference wavelengths.')
        closest=min(rows,key=lambda r:abs(r['two_theta_deg']-position),default=None)
        distance=abs(closest['two_theta_deg']-position) if closest else None
        matches.append({'phase':phase,'reference_two_theta_deg':position,'source':source,
                        'matched':distance is not None and distance<=tolerance_deg,
                        'nearest_peak_deg':closest['two_theta_deg'] if closest else None,
                        'distance_deg':distance})
    result={'modality':'XRD','algorithm':ALGORITHM,'method':'Baseline subtraction and prominence-based candidate peaks',
            'parameters':{'relative_prominence':prominence,'min_width_deg':min_width_deg,'tolerance_deg':tolerance_deg,'wavelength_angstrom':wavelength},
            'point_count':len(x),'scan_min_deg':float(x[0]),'scan_max_deg':float(x[-1]),
            'peaks':rows,'reference_matches':matches,'probability':None,'suggested_state':'PENDING',
            'interpretation':'Peak proximity is not phase identification. No resolved target peaks does not prove atomic dispersion.'}
    return result,{'two_theta_deg':grid.tolist(),'intensity':raw.tolist(),'baseline':baseline.tolist()}


def read_image(content):
    from PIL import Image
    if len(content)>40_000_000:raise ValueError('Use an exported image smaller than 40 MB.')
    with Image.open(io.BytesIO(content)) as image:
        if image.width*image.height>16_000_000:raise ValueError('Use a crop with at most 16 million pixels.')
        if getattr(image,'n_frames',1)!=1:raise ValueError('Export one microscopy frame at a time.')
        array=np.asarray(image)
        if array.ndim==3:
            array=np.asarray(image.convert('L'))
        array=np.asarray(array,dtype=float)
    if array.ndim!=2 or min(array.shape)<16 or not np.isfinite(array).all():raise ValueError('Use a finite 2D microscopy image of at least 16 x 16 pixels.')
    return array


def microscopy_analysis(content,modality,nm_per_pixel=None,roi=None,sigma_px=1.5,threshold_sigma=4.,
                        min_area_px=3,max_area_px=10000,neighbor_distance_nm=None):
    if modality not in {'SEM','STEM'}:raise ValueError('Select SEM or STEM.')
    image=read_image(content);height,width=image.shape
    if roi is None:roi=[0,0,width,height]
    if len(roi)!=4 or any(int(v)!=v for v in roi):raise ValueError('ROI must contain four integer pixel bounds.')
    x0,y0,x1,y1=map(int,roi)
    if not 0<=x0<x1<=width or not 0<=y0<y1<=height or min(x1-x0,y1-y0)<16:raise ValueError('Select a valid ROI of at least 16 x 16 pixels.')
    crop=image[y0:y1,x0:x1];sigma_px=positive(sigma_px,'Feature scale')
    threshold_sigma=positive(threshold_sigma,'Detection threshold')
    if not 1<=int(min_area_px)<=int(max_area_px):raise ValueError('Use 1 <= minimum area <= maximum area.')
    scale=None if nm_per_pixel is None else positive(nm_per_pixel,'nm per pixel')
    lo,hi=np.percentile(crop,[1,99.9]);normalized=(crop-lo)/max(hi-lo,1e-12)
    filtered=ndimage.gaussian_filter(normalized,sigma_px)
    response=filtered-ndimage.gaussian_filter(normalized,max(3.,sigma_px*5)) if modality=='SEM' else -ndimage.gaussian_laplace(normalized,sigma_px)*sigma_px**2
    center=float(np.median(response));mad=float(np.median(abs(response-center)))
    noise=max(1.4826*mad,float(np.std(response))*.05,1e-9)
    cutoff=center+threshold_sigma*noise
    rows=[]
    if modality=='SEM':
        labels,count=ndimage.label(response>cutoff)
        objects=ndimage.find_objects(labels)
        for label,slices in enumerate(objects,1):
            if slices is None:continue
            mask=labels[slices]==label;area=int(mask.sum())
            if not int(min_area_px)<=area<=int(max_area_px):continue
            coords=np.argwhere(mask)+np.array([slices[0].start,slices[1].start]);y,x=coords.mean(0)
            diameter=float(2*np.sqrt(area/np.pi))
            rows.append({'x_px':float(x+x0),'y_px':float(y+y0),'area_px':area,'equivalent_diameter_px':diameter,
                         'equivalent_diameter_nm':None if scale is None else diameter*scale})
    else:
        size=max(3,2*int(np.ceil(sigma_px))+1)
        maxima=(response==ndimage.maximum_filter(response,size=size))&(response>cutoff)
        border=int(np.ceil(sigma_px*2));maxima[:border]=False;maxima[-border:]=False;maxima[:,:border]=False;maxima[:,-border:]=False
        labels,_=ndimage.label(maxima)
        for y,x in ndimage.center_of_mass(maxima,labels,range(1,int(labels.max())+1)):
            rows.append({'x_px':float(x+x0),'y_px':float(y+y0),'response':float(response[round(y),round(x)])})
    if len(rows)>10000:raise ValueError('More than 10000 candidates; crop the image or increase the threshold.')
    coordinates=np.array([[r['x_px'],r['y_px']] for r in rows])
    distances=cKDTree(coordinates).query(coordinates,k=2)[0][:,1] if len(rows)>1 else np.full(len(rows),np.nan)
    neighbor=positive(neighbor_distance_nm,'Neighbor distance') if neighbor_distance_nm is not None else None
    if neighbor is not None and scale is None:raise ValueError('Physical neighbor distances require calibrated nm per pixel.')
    for row,distance in zip(rows,distances):
        row['nearest_neighbor_nm']=None if scale is None or not np.isfinite(distance) else float(distance*scale)
        row['within_selected_neighbor_distance']=None if neighbor is None or row['nearest_neighbor_nm'] is None else row['nearest_neighbor_nm']<=neighbor
    return {'modality':modality,'algorithm':ALGORITHM,
            'method':'Bright-feature connected components' if modality=='SEM' else 'Laplacian-of-Gaussian bright-spot candidates',
            'parameters':{'nm_per_pixel':scale,'roi':roi,'sigma_px':sigma_px,'threshold_sigma':threshold_sigma,
                          'min_area_px':int(min_area_px),'max_area_px':int(max_area_px),'neighbor_distance_nm':neighbor},
            'image_width_px':width,'image_height_px':height,'candidate_count':len(rows),'features':rows,
            'sampled_area_nm2':None if scale is None else float(crop.size*scale**2),
            'probability':None,'suggested_state':'PENDING',
            'interpretation':'Contrast candidates require review. Bright spots are not identified elements; SEM cannot resolve isolated atoms.'}


def register_artifact(store,ledger,sample_id,modality,name,content,analysis,metadata=None):
    sample=store.get('sample',sample_id)
    if not sample or sample.get('origin')!=ledger.origin or ledger.origin!=store.origin:raise ValueError('Choose a sample in this data space.')
    if modality not in OBSERVATIONS or analysis.get('modality')!=modality or analysis.get('suggested_state')!='PENDING':raise ValueError('Use aligned unreviewed characterization analysis.')
    extension=Path(name).suffix.lower()
    allowed={'.csv','.txt','.xy','.dat'} if modality=='XRD' else {'.png','.tif','.tiff','.jpg','.jpeg'}
    if extension not in allowed:raise ValueError('Use a supported exported characterization file.')
    sha=hashlib.sha256(content).hexdigest()
    folder=store.root/'characterization'/store.origin.lower();folder.mkdir(parents=True,exist_ok=True)
    path=folder/(sha+extension)
    if not path.exists():path.write_bytes(content)
    # Older stores can predate stable physical-sample identities. Bind once so
    # later display-ID compaction cannot attach evidence to another specimen.
    with store.connect() as db:
        db.execute('ATTACH DATABASE ? AS discovery',(str(ledger.path.resolve()),))
        db.execute('BEGIN IMMEDIATE')
        encoded=db.execute("SELECT payload FROM items WHERE kind='sample' AND id=?",(sample_id,)).fetchone()
        if encoded is None:raise ValueError('Sample no longer exists.')
        current=json.loads(encoded[0])
        if recipe_id(current)!=recipe_id(sample):raise ValueError('Recipe changed during source registration.')
        if not current.get('record_uid'):
            old_key=identity(current);current['record_uid']=uuid4().hex
            updated=json.dumps(current,allow_nan=False)
            db.execute("UPDATE items SET payload=? WHERE kind='sample' AND id=?",(updated,sample_id))
            db.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'bind_evidence_identity','sample',?,?,?)",(stamp(),sample_id,encoded[0],updated))
            binding={'id':uuid4().hex,'created':stamp(),'origin':store.origin,'previous_key':old_key,'record_key':current['record_uid']}
            db.execute('INSERT INTO discovery.events VALUES(?,?,?,?)',(binding['id'],'identity_binding',binding['created'],json.dumps(binding)))
            epoch={'id':'data-epoch','origin':store.origin,'value':uuid4().hex}
            db.execute("INSERT OR REPLACE INTO items VALUES('meta','data-epoch',?)",(json.dumps(epoch),))
        event={'id':uuid4().hex,'created':stamp(),'origin':store.origin,'record_key':identity(current),'sample_id':sample_id,
            'recipe_key':recipe_id(current),'modality':modality,'file_name':Path(name).name,'sha256':sha,
            'relative_path':str(path.relative_to(store.root)),'analysis':copy.deepcopy(analysis),'metadata':metadata or {},'status':'AWAITING_REVIEW'}
        db.execute('INSERT INTO discovery.events VALUES(?,?,?,?)',(event['id'],'characterization',event['created'],json.dumps(event,allow_nan=False)))
    return event


def review_artifact(store,ledger,artifact_id,observation,reviewer,notes=''):
    if store.origin!=ledger.origin:raise ValueError('Evidence review cannot cross data spaces.')
    artifact=next((a for a in ledger.records('characterization') if a['id']==artifact_id),None)
    if artifact is None:raise ValueError('Unknown characterization artifact.')
    sample=next((s for s in store.all('sample') if identity(s)==artifact['record_key']),None)
    if sample is None or recipe_id(sample)!=artifact['recipe_key']:raise ValueError('The sample recipe changed or was deleted; acquire or register evidence for the actual recipe.')
    path=(store.root/artifact['relative_path']).resolve()
    if not path.is_relative_to(store.root.resolve()) or not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']:
        raise ValueError('Original evidence is missing or its checksum changed.')
    if observation not in OBSERVATIONS[artifact['modality']] or not reviewer.strip():raise ValueError('Choose a supported observation and name the reviewer.')
    return ledger.add('characterization_review',{'origin':store.origin,'artifact_id':artifact_id,
        'record_key':artifact['record_key'],'recipe_key':artifact['recipe_key'],'modality':artifact['modality'],
        'observation':observation,'reviewer':reviewer.strip(),'notes':notes.strip(),'evidence_sha256':artifact['sha256']})


def evidence_summary(sample,ledger):
    if sample.get('origin')!=ledger.origin:raise ValueError('Evidence fusion cannot cross data spaces.')
    reviews={r['artifact_id']:r for r in ledger.records('characterization_review') if r['record_key']==identity(sample)}
    artifacts={a['id']:a for a in ledger.records('characterization')}
    active=[];invalid=[]
    for key,review in reviews.items():
        if key not in artifacts or review['recipe_key']!=recipe_id(sample):continue
        artifact=artifacts[key];path=(ledger.path.parent/artifact['relative_path']).resolve()
        if (not path.is_relative_to(ledger.path.parent.resolve()) or not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest()!=review['evidence_sha256']):
            invalid.append(key);continue
        active.append(review)
    observed={r['observation'] for r in active}
    positive_stem='ISOLATED_SITE_CANDIDATES_CONFIRMED' in observed
    negative=bool(observed&{'TARGET_CRYSTALLINE_PHASE_DETECTED','CLUSTERS_OBSERVED'})
    conflict=positive_stem and negative
    label='PENDING' if conflict else 'FAIL' if negative else 'PASS' if positive_stem or 'NO_RESOLVED_TARGET_CRYSTALLINE_PEAKS' in observed else 'PENDING'
    return {'record_key':identity(sample),'recipe_key':recipe_id(sample),'review_ids':[r['id'] for r in active],
            'artifact_ids':[r['artifact_id'] for r in active],'observations':sorted(observed),'recommended_screening_state':label,
            'conflict':conflict,'morphology_alert':'AGGREGATES_OBSERVED' in observed,
            'invalid_artifact_ids':invalid,'modalities':sorted({r['modality'] for r in active}),
            'interpretation':'A reviewed screening assessment; XRD non-detection does not establish atomic dispersion. SEM supplies morphology context.'}


def commit_assessment(store,ledger,sample_id,state,reviewer,reason,override=False):
    """Atomically commit reviewed feedback and its discovery event across both DBs."""
    from ..domain import structural_assessment,validate_sample
    if ledger.origin!=store.origin or state not in {'PASS','FAIL','PENDING'} or not reviewer.strip() or not reason.strip():
        raise ValueError('Select the outcome and provide reviewer and screening criteria.')
    with store.connect() as db:
        db.execute('ATTACH DATABASE ? AS discovery',(str(ledger.path.resolve()),))
        db.execute('BEGIN IMMEDIATE')
        row=db.execute("SELECT payload FROM items WHERE kind='sample' AND id=?",(sample_id,)).fetchone()
        if row is None:raise ValueError('Sample no longer exists.')
        sample=json.loads(row[0]);summary=evidence_summary(sample,ledger)
        if summary['invalid_artifact_ids']:raise ValueError('Reviewed evidence is missing or changed; restore the original source before committing.')
        if not summary['review_ids'] or not set(summary['modalities'])&{'XRD','STEM'}:raise ValueError('A structural decision requires reviewed XRD or STEM evidence; SEM alone remains morphology context.')
        if state!='PENDING' and (summary['conflict'] or state!=summary['recommended_screening_state']) and not override:
            raise ValueError('Conflicting or divergent evidence requires an explicit override and explanation.')
        old_state,_=structural_assessment(sample)
        if old_state in {'PASS','FAIL'} and old_state!=state and not override:raise ValueError('Replacing an existing resolved judgment requires an explicit override.')
        prior=db.execute("SELECT payload FROM discovery.events WHERE id=? AND kind='characterization_assessment'",(sample.get('characterization_assessment_id',''),)).fetchone()
        if prior:
            previous=json.loads(prior[0])
            if (previous['state']==state and previous['reviewer']==reviewer.strip() and previous['criteria']==reason.strip()
                    and previous['evidence']['review_ids']==summary['review_ids']):return previous
        event={'id':uuid4().hex,'created':stamp(),'origin':store.origin,'record_key':identity(sample),
               'sample_id':sample_id,'recipe_key':summary['recipe_key'],'state':state,'reviewer':reviewer.strip(),
               'criteria':reason.strip(),'override':bool(override),'evidence':summary,'mode':'REVIEWED_CHARACTERIZATION'}
        sample['characterization_assessment_id']=event['id']
        sample['characterization_reported_at']=event['created']
        sample['xrd']={'mode':'AUTO','state':state,'source':'Characterization assessment '+event['id'],
                       'criteria':reason.strip(),'reviewer':reviewer.strip(),'assessment_basis':'REVIEWED_XRD_STEM'}
        # Keep existing ICP measurement timing. A structural return can resolve a new pending specimen.
        if state!='PENDING' and not sample.get('result_reported_at'):sample['result_reported_at']=event['created']
        if state=='PENDING':sample['stem']={'state':'INCONCLUSIVE','source':'Characterization assessment '+event['id'],'reviewer':reviewer.strip()}
        sample=validate_sample(sample)
        encoded=json.dumps(sample,allow_nan=False)
        db.execute("UPDATE items SET payload=? WHERE kind='sample' AND id=?",(encoded,sample_id))
        db.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'characterization_review','sample',?,?,?)",(event['created'],sample_id,row[0],encoded))
        db.execute('INSERT INTO discovery.events VALUES(?,?,?,?)',(event['id'],'characterization_assessment',event['created'],json.dumps(event,allow_nan=False)))
        epoch={'id':'data-epoch','origin':store.origin,'value':uuid4().hex}
        db.execute("INSERT OR REPLACE INTO items VALUES('meta','data-epoch',?)",(json.dumps(epoch),))
    return event
