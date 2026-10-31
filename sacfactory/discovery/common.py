"""Pre-synthesis contracts shared by discovery tools."""
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
import numpy as np
from ..domain import ELEMENTS, INVALID_RUN_CONDITIONS, parse_measurement, structural_assessment

FIELDS = ('origin','elements','concentrations_mM','pulse_count','pulse_limit_C','precursor_family',
          'support_batch','protocol_id','precursors','pan_mass_g','solvent_volume_mL',
          'carbonization_loss_fraction','carbonization_loss','voltage_V','current_A')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,default=str).encode()).hexdigest()


def query(sample):
    result = {k:copy.deepcopy(sample[k]) for k in FIELDS if k in sample}
    if result.get('origin') not in {'REAL','DEMO'}:
        raise ValueError('Select REAL or DEMO.')
    elements = result.get('elements',[])
    if not elements or len(set(elements))!=len(elements) or set(elements)-set(ELEMENTS):
        raise ValueError('Select unique known elements.')
    result['elements'] = sorted(elements)
    concentrations = result.get('concentrations_mM',{})
    if set(concentrations)!=set(elements) or any(not math.isfinite(float(v)) or float(v)<=0 for v in concentrations.values()):
        raise ValueError('Each target needs a positive finite precursor concentration.')
    result['concentrations_mM']={e:float(concentrations[e]) for e in result['elements']}
    cutoff=float(result.get('pulse_limit_C',float('nan')))
    if not math.isfinite(cutoff) or not 0<cutoff<4000:
        raise ValueError('A controlled cutoff between 0 and 4000 C is required.')
    result['pulse_limit_C']=cutoff
    pulses = float(result.get('pulse_count',0))
    if not math.isfinite(pulses) or pulses<1 or not pulses.is_integer():
        raise ValueError('Pulse count must be a positive integer.')
    result['pulse_count'] = int(pulses)
    for key,default in [('pan_mass_g',2.),('solvent_volume_mL',20.)]:
        result[key] = float(result.get(key,default))
        if not math.isfinite(result[key]) or result[key]<=0:
            raise ValueError('PAN mass and solvent volume must be positive and finite.')
    for key,default in [('voltage_V',32.),('current_A',80.)]:
        result[key]=float(result.get(key) or default)
        if not math.isfinite(result[key]) or result[key]<=0:raise ValueError('Electrical settings must be positive and finite.')
    loss=float(result.get('carbonization_loss_fraction',result.get('carbonization_loss',.06)))
    if not math.isfinite(loss) or not 0<=loss<1:raise ValueError('Carbonization loss must be a fraction in [0,1).')
    result.pop('carbonization_loss',None);result['carbonization_loss_fraction']=loss
    result['sample_id'] = 'DISCOVERY_QUERY'
    return result


def recipe_id(sample):
    result=query(sample);result.pop('sample_id')
    return digest(result)


def identity(sample):
    return sample.get('record_uid') or sample.get('sample_id')


def outcomes(sample):
    if sample.get('sample_condition') in INVALID_RUN_CONDITIONS:
        return {},None
    values={}
    for assay in sample.get('icp',[]):
        if assay.get('unit','wt%')!='wt%':raise ValueError('Normalize ICP to W / wt% before discovery analysis.')
        if assay.get('element') not in sample['elements']:raise ValueError('ICP element is outside the declared recipe.')
        parsed=parse_measurement(assay.get('value',''))
        if assay.get('flag','MEASURED')=='MEASURED' and parsed['kind']=='measured':
            value=float(parsed['value'])
            if not 0<=value<=100:raise ValueError('ICP W must be 0-100 wt%.')
            if assay['element'] in values:raise ValueError('Ambiguous duplicate ICP element.')
            values[assay['element']]=value
    state,_=structural_assessment(sample)
    return values,1 if state=='PASS' else 0 if state=='FAIL' else None


def values(prediction):
    return {r['element']:float(r['predicted_mean_w_wt_percent']) for r in prediction.get('icp_predictions',[])
            if r.get('predicted_mean_w_wt_percent') is not None}


def prediction_metrics(samples,predictions):
    errors=[];squares=[];observed=[];predicted=[];brier=[]
    for sample,p in zip(samples,predictions):
        actual,label=outcomes(sample);forecast=values(p)
        common=sorted(set(actual)&set(forecast))
        if common:
            errors.append(np.mean([abs(actual[e]-forecast[e]) for e in common]))
            squares.append(np.mean([(actual[e]-forecast[e])**2 for e in common]))
        if label is not None and p.get('p_xrd_pass') is not None:
            brier.append((label-p['p_xrd_pass'])**2)
            observed.append(label);predicted.append(int(p['p_xrd_pass']>=p.get('xrd_pass_threshold',.5)))
    fail=[guess==0 for label,guess in zip(observed,predicted) if label==0]
    passed=[guess==1 for label,guess in zip(observed,predicted) if label==1]
    return {'quantified_specimens':len(errors),'ICP_MAE_wt_percent':float(np.mean(errors)) if errors else None,
            'ICP_RMSE_wt_percent':float(np.sqrt(np.mean(squares))) if squares else None,
            'structural_specimens':len(brier),'structural_Brier':float(np.mean(brier)) if brier else None,
            'FAIL_recall':float(np.mean(fail)) if fail else None,'PASS_recall':float(np.mean(passed)) if passed else None,
            'structural_balanced_accuracy':float((np.mean(fail)+np.mean(passed))/2) if fail and passed else None}
