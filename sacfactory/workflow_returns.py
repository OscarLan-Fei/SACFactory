"""Structured operator returns and execution contracts for laboratory work cards."""
import json
import math


def validate_return(checkpoint,data):
    data={} if data is None else data
    if not isinstance(data,dict):raise ValueError('Returned data must be a JSON object.')
    try:encoded=json.dumps(data,allow_nan=False)
    except (ValueError,TypeError) as exc:raise ValueError('Return finite JSON values only.') from exc
    if len(encoded)>24000:raise ValueError('A checkpoint return is limited to 24 KB; register larger raw files separately.')
    missing=[k for k in checkpoint.get('return_fields',[]) if k not in data or data[k] is None or data[k]=='']
    if missing:raise ValueError('Record the required return fields: '+', '.join(missing))
    if checkpoint.get('approval_required') and data.get('approved') is not True:raise ValueError('Explicit operator approval is required before dispatch.')
    for key in ('pan_mass_g','solvent_volume_mL','recovered_mass_g','duration_min','fiber_mass_g','electrode_area_cm2'):
        if key in data and (type(data[key]) not in (int,float) or not math.isfinite(data[key]) or data[key]<=0):raise ValueError(key+' must be a positive finite number.')
    if 'precursor_amounts_mmol' in data:
        amounts=data['precursor_amounts_mmol']
        if not isinstance(amounts,dict) or not amounts or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in amounts.values()):raise ValueError('Record positive finite precursor amounts in mmol.')
    return data


def pipeline_for(operation):
    return {
        'inverse.design':['Bounded candidate generation','Precursor / unit resolution','Loading and structural forecasts','Diversity-aware ranking'],
        'model.train':['Eligible records','Feature construction / physical prior','Residual-model fit','Checkpoint reload verification'],
        'validation.run':['Recipe-group masks','Fold-local fitting','Held-out predictions','Pooled error statistics'],
        'characterization.analyze':['Verify acquisition receipts','Parse raw exports','Element mass-content import','Structural evidence analysis'],
        'records.training_pool':['Selected historical records','Current-group returns','Eligibility filtering','Deduplicated training pool'],
    }.get(operation,[])


def model_names(output,parameters):
    name=(output or {}).get('_execution',{}).get('model') or parameters.get('model') or parameters.get('family')
    if name:return name
    metrics=(output or {}).get('metrics',[])
    if isinstance(metrics,list):return ', '.join(dict.fromkeys(r['model'] for r in metrics if isinstance(r,dict) and r.get('model'))) or None
    return None


def result_summary(output):
    if not output:return 'Awaiting execution'
    if output.get('awaiting_manual_action'):return 'Awaiting explicit plan approval' if output.get('approval_required') else 'Awaiting operator return'
    if 'returned_data' in output:return 'Recorded: '+', '.join(output['returned_data'])
    if 'design' in output:return str(len(output['design']['rows']))+' ranked recipes · '+str(output.get('model_id',''))
    if 'candidate' in output:return str(output.get('training_specimens',0))+' training records · '+str(output['candidate'].get('model_id',''))
    if 'eligible_count' in output:return str(output['eligible_count'])+' eligible / '+str(output['source_count'])+' source records'
    if 'sample_ids' in output:return str(len(output['sample_ids']))+' specimens'
    if 'acquisition' in output:return 'Acquisition received: '+str(output['acquisition'].get('job_id',''))
    if 'receipt' in output:return 'Synthesis receipt received'
    if 'evidence' in output:return 'Evidence available for review'
    return ', '.join(k for k in output if not k.startswith('_'))[:180]
