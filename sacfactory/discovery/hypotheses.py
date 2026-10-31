"""Pre-registered one-factor synthesis contrasts, separate from causal claims."""
import copy
from datetime import datetime
from .common import digest,query,recipe_id,values,stamp


def contrasts(base,factor='temperature',values_to_test=(900.,1200.)):
    baseline=query(base);recipes=[baseline]
    if factor not in {'temperature','pulse','amount'}:raise ValueError('Select temperature, pulse or amount.')
    for value in values_to_test:
        r=copy.deepcopy(baseline)
        if factor=='temperature':r['pulse_limit_C']=float(value)
        elif factor=='pulse':r['pulse_count']=value
        else:
            if len(r['elements'])!=1:raise ValueError('Amount contrasts currently change one single-element feed only.')
            r['concentrations_mM'][r['elements'][0]]=float(value)*1000/r['solvent_volume_mL']
        r=query(r)
        if recipe_id(r) not in {recipe_id(s) for s in recipes}:recipes.append(r)
    if len(recipes)<2:raise ValueError('Choose a value different from the baseline.')
    return recipes


def register(recipes,predictions,factor,statement):
    if not statement.strip() or len(recipes)<2 or len(recipes)!=len(predictions):raise ValueError('A testable statement and aligned contrasts are required.')
    baseline=recipes[0];anchor=copy.deepcopy(query(baseline))
    for recipe in recipes[1:]:
        a=copy.deepcopy(anchor);b=query(recipe)
        if factor=='temperature':a.pop('pulse_limit_C');b.pop('pulse_limit_C')
        elif factor=='pulse':a.pop('pulse_count');b.pop('pulse_count')
        elif factor=='amount':a.pop('concentrations_mM');b.pop('concentrations_mM')
        else:raise ValueError('Select a supported intervention.')
        if a!=b:raise ValueError('A controlled contrast must change only the selected factor.')
    base_values=values(predictions[0]);effects=[]
    for prediction in predictions[1:]:
        current=values(prediction)
        effects.append({e:current[e]-base_values[e] for e in set(current)&set(base_values)})
    return {'hypothesis_id':digest([recipes,statement,stamp()])[:20],'created':stamp(),'statement':statement,
            'factor':factor,'recipes':[query(s) for s in recipes],'recipe_keys':[recipe_id(s) for s in recipes],
            'predictions':predictions,'predicted_component_changes_wt_percent':effects,
            'interpretation':'Pre-registered model hypothesis; no causal claim before controlled experimental feedback'}


def assess(card,samples):
    import numpy as np
    from .common import outcomes
    def timestamp(value):
        try:
            parsed=datetime.fromisoformat(str(value).replace('Z','+00:00'))
            return parsed if parsed.tzinfo is not None else None
        except (ValueError,TypeError):return None
    registered=timestamp(card['created'])
    if registered is None:raise ValueError('Registration requires a timezone-aware timestamp.')
    groups={key:[] for key in card['recipe_keys']}
    for sample in samples:
        try:key=recipe_id(sample)
        except ValueError:continue
        reported=timestamp(sample.get('result_reported_at'))
        if key in groups and reported is not None and reported>registered:
            actual,_=outcomes(sample)
            if actual:groups[key].append(actual)
    baseline=groups[card['recipe_keys'][0]];result=[]
    for key,effect in zip(card['recipe_keys'][1:],card['predicted_component_changes_wt_percent']):
        observed=groups[key]
        for element,forecast_delta in effect.items():
            a=[v[element] for v in baseline if element in v];b=[v[element] for v in observed if element in v]
            delta=float(np.mean(b)-np.mean(a)) if a and b else None
            result.append({'recipe_key':key,'element':element,'baseline_n':len(a),'intervention_n':len(b),
                           'predicted_change_wt_percent':forecast_delta,'observed_change_wt_percent':delta,
                           'direction_agreement':None if delta is None else bool(np.sign(delta)==np.sign(forecast_delta)),
                           'status':'POST_REGISTRATION_DESCRIPTIVE_CONTRAST' if delta is not None else 'AWAITING_NEW_MEASUREMENTS'})
    return result
