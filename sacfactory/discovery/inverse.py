"""Bounded target-conditioned recipe search with explicit physical units."""
import copy
import math
import numpy as np
from scipy.stats import qmc
from .common import query,recipe_id,values
from ..precursor_catalog import resolve_planning_precursors


def candidates(context,elements,bounds_mmol,temperature=(800,1600),pulses=(1,2,4),count=128,seed=42,max_total_mmol=None):
    elements=sorted(set(elements))
    if not elements or set(bounds_mmol)!=set(elements):raise ValueError('Provide amount bounds for every selected element.')
    lower,upper=map(float,temperature)
    if not all(math.isfinite(v) for v in (lower,upper)) or not 0<lower<=upper<4000:raise ValueError('Invalid controlled cutoff bounds.')
    grid_low=math.ceil(lower/10)*10;grid_high=math.floor(upper/10)*10
    if grid_low>grid_high:raise ValueError('No 10 C setpoint inside the declared range.')
    if not pulses or any(type(p) is not int or p<1 for p in pulses):raise ValueError('Use positive integer pulse options.')
    if not 1<=count<=4096:raise ValueError('Use 1-4096 candidate recipes.')
    for lo,hi in bounds_mmol.values():
        if not all(math.isfinite(float(v)) for v in (lo,hi)) or not 0<float(lo)<=float(hi):raise ValueError('Invalid positive mmol bounds.')
    if max_total_mmol is not None and (not math.isfinite(max_total_mmol) or max_total_mmol<=0):raise ValueError('Invalid total precursor limit.')
    points=qmc.Sobol(len(elements)+2,scramble=True,seed=seed).random_base2(math.ceil(math.log2(count)))[:count]
    result=[];seen=set()
    for point in points:
        amounts={e:float(bounds_mmol[e][0]+point[j]*(bounds_mmol[e][1]-bounds_mmol[e][0])) for j,e in enumerate(elements)}
        if max_total_mmol is not None and sum(amounts.values())>max_total_mmol:continue
        recipe={k:copy.deepcopy(v) for k,v in context.items() if k not in ['icp','xrd','stem','result_reported_at','icp_review']}
        recipe['elements']=elements;volume=float(recipe.get('solvent_volume_mL',20.))
        recipe['concentrations_mM']={e:n*1000/volume for e,n in amounts.items()}
        recipe['precursors']=resolve_planning_precursors({},context,elements)
        recipe['pulse_limit_C']=float(np.clip(round((grid_low+point[-2]*(grid_high-grid_low))/10)*10,grid_low,grid_high))
        recipe['pulse_count']=pulses[min(len(pulses)-1,int(point[-1]*len(pulses)))]
        recipe=query(recipe);key=recipe_id(recipe)
        if key not in seen:seen.add(key);result.append(recipe)
    return result


def rank(recipes,predictions,target_total=(0.,100.),min_probability=0.,batch_size=8,allow_extrapolation=False,disagreements=None,strategy='Target attainment'):
    if len(recipes)!=len(predictions):raise ValueError('Recipes and forecasts must align.')
    lo,hi=map(float,target_total)
    if not all(math.isfinite(v) for v in (lo,hi,min_probability)) or not 0<=lo<=hi<=100 or not 0<=min_probability<=1:
        raise ValueError('Use a physical ICP target interval and probability threshold.')
    if strategy not in {'Target attainment','Model disagreement','Balanced learning'}:raise ValueError('Unknown discovery strategy.')
    rows=[]
    for i,(recipe,forecast) in enumerate(zip(recipes,predictions)):
        w=values(forecast);p=forecast.get('p_xrd_pass');total=forecast.get('predicted_total_wt_percent')
        if set(w)!=set(recipe['elements']) or total is None or not math.isfinite(total):continue
        if min_probability>0 and (p is None or p<min_probability):continue
        if not allow_extrapolation and forecast.get('outside_training_support',True):continue
        distance=max(lo-total,0,total-hi)/max(hi-lo,.25)
        disagreement=(disagreements[i].get('ICP_model_SD_wt_percent') or 0) if disagreements else 0
        benefit=1/(1+distance)
        utility=benefit if strategy=='Target attainment' else disagreement if strategy=='Model disagreement' else benefit+.5*disagreement
        rows.append({'recipe':query(recipe),'prediction':forecast,'recipe_key':recipe_id(recipe),
                     'predicted_total_wt_percent':float(total),'p_xrd_pass':p,'target_distance':distance,
                     'model_disagreement':disagreement,'utility':float(utility),'strategy':strategy})
    chosen=[]
    while rows and len(chosen)<int(batch_size):
        def feature(row):
            r=row['recipe'];return np.r_[[np.log1p(r['concentrations_mM'][e]) for e in sorted(r['elements'])],r['pulse_limit_C']/1000,r['pulse_count']/10]
        def score(row):
            diversity=min(np.linalg.norm(feature(row)-feature(s)) for s in chosen) if chosen else 0
            return row['utility']+.1*diversity/(1+diversity)
        item=max(rows,key=score);chosen.append(item);rows.remove(item)
    return chosen
