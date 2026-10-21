"""Frozen synthetic designs with physics-linked mass, ICP and XRD observations."""
import copy
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sacfactory.domain import RULE_VERSION,screen,recipe_key,validate_sample
from sacfactory.chemical_descriptors import reference
from sacfactory.demo_physics import mass_balance,dose_factor,thermal_reference
from sacfactory.demo_observation import observation_settings,apply_batch_retention,aggregation_parameters,observe_assays
from sacfactory.thermal_loss import loss_settings,loss_reference,elemental_loss_reference
from sacfactory.planning import fingerprint
SEED=20261013


def prediction_features(s,atoms):
    """Pre-synthesis controls/references; no observed or latent outcomes."""
    elements=s['elements']; c=np.array([s['concentrations_mM'][e] for e in elements]); weights=c/c.sum()
    thermal=[elemental_loss_reference(e) for e in elements]
    result=[np.log(c.sum()),np.log(len(elements)),s['pulse_limit_C']/1000,np.log(s['pulse_count']),
        float(sum(w for e,w in zip(elements,weights) if dose_factor(e)==1.8))]
    properties=[([atoms[e].get('en_pauling') for e in elements],1.),
                ([atoms[e].get('covalent_radius_cordero') for e in elements],100.)]
    properties += [([r[field] for r in thermal],scale) for field,scale in
        [('melting_point_K',2000.),('boiling_point_K',3000.),('heat_of_formation_kJ_mol',400.)]]
    for raw,scale in properties:
        known=np.array([v is not None and np.isfinite(v) for v in raw])
        coverage=float(weights[known].sum())
        if coverage>0:
            values=np.array([v/scale for v,k in zip(raw,known) if k]); w=weights[known]/coverage
            mean=float(w@values); variance=float(w@((values-mean)**2))
        else:
            mean=variance=0.  # Numerical placeholders accompanied by missing-weight=1.
        result.extend([mean,variance,1-coverage])
    return result


def make_demo():
    raw=(ROOT/'examples/demo_campaign_design.json').read_bytes(); design=json.loads(raw)
    settings=observation_settings(); atoms={r['symbol']:r for r in reference()['elements']}
    samples=[]; phases=[]; batch_rng=np.random.default_rng(SEED+10)
    batch_factors={b:math.exp(float(batch_rng.normal(0,settings['batch_log_retention_sd']))) for b in dict.fromkeys(s['batch_id'] for s in design['rows'])}
    process_rng=np.random.default_rng(SEED+20); assay_rng=np.random.default_rng(SEED+30); xrd_rng=np.random.default_rng(SEED+40)
    for elements,group in groupby(design['rows'],key=lambda s:tuple(s['elements'])):
        recipes=list(group); prior=[s for s in samples if screen(s,{})['label'] is not None]
        if len(prior)>144:
            prior=[prior[i] for i in np.unique(np.linspace(0,len(prior)-1,144,dtype=int))]
        model=None
        if len(prior)>=12 and len({screen(s,{})['label'] for s in prior})==2:
            model=RandomForestClassifier(n_estimators=60,min_samples_leaf=4,random_state=SEED,n_jobs=1).fit(
                [prediction_features(s,atoms) for s in prior],[screen(s,{})['label'] for s in prior])
        capture={'rule_version':RULE_VERSION,'training_batch_ids':sorted({s['batch_id'] for s in prior}),
            'training_records':[{'sample_id':s['sample_id'],'record_uid':s['record_uid']} for s in prior],
            'feature_version':'CONTROL_AND_KNOWN_REFERENCE_MOMENTS_WITH_MASKS_V6',
            'loss_settings_sha256':loss_settings()['settings_sha256'],'observation_settings_sha256':settings['settings_sha256']}
        for recipe in recipes:
            s=copy.deepcopy(recipe); case=s.pop('workflow_case')
            s.update(record_uid=f"demo-v6-{s['sample_id']}",origin='DEMO',precursor_family='DEMO_CATALOG_SALTS_V6',
                support_batch='DEMO_PAN_2G_V6',protocol_id='DEMO_DMF_20ML_IR_V6')
            balance=mass_balance(s['concentrations_mM'],s['temperature_C'],s['pulse_count'],process_rng)
            apply_batch_retention(balance,batch_factors[s['batch_id']])
            balance.update(temperature_basis='SIMULATED_RECORDED_PEAK',used_temperature_C=s['temperature_C'])
            latent=aggregation_parameters(balance,s['temperature_C'],s['pulse_count'])
            assays,icp,details=observe_assays(balance,assay_rng)
            latent['simulated_xrd_draw']=float(xrd_rng.random())
            xrd='FAIL' if latent['simulated_xrd_draw']<latent['xrd_fail_probability'] else 'PASS'
            condition='RECOVERED'; basis='SIMULATED_OBSERVATIONS'
            if case in ('EQUIPMENT_ABORT','AWAITING_RESULTS','BURNED_OUT'):
                assays=[]; icp=xrd='PENDING'; details={}; latent={}
                condition=case if case!='AWAITING_RESULTS' else 'RECOVERED'
                basis='ILLUSTRATIVE_WORKFLOW_EXCEPTION_NO_ASSAY'
                if case=='BURNED_OUT':
                    balance.update(carrier_after_pulse_g=0.,carrier_loss_fraction=1.,recovered_sample_g=0.,total_retained_metal_g=0.)
                    for row in balance['elements'].values():
                        row.update(retained_metal_g=0.,retention_fraction=0.,lost_metal_g=row['feed_metal_g'],true_w_wt_percent=None)
            s.update(sample_condition=condition,icp=assays,
                icp_review={'state':icp,'source':'SIMULATED all-target detection criterion; not real analytical limits','reviewer':'DEMO generator'},
                xrd={'state':xrd,'mode':'MANUAL','source':'SIMULATED crystalline-phase observation; not an XRD spectrum','reviewer':'DEMO generator'},
                stem={'state':'NOT_TESTED'},notes='SIMULATED physics-linked campaign. Thermal priors, assay limits/noise and XRD surrogates are assumptions; no experimental curve fitting.',
                demo_mass_balance=balance,demo_observation={'version':settings['version'],'settings_sha256':settings['settings_sha256'],
                    'result_basis':basis,'workflow_case':case,'assays':details,'screening_latent':latent})
            from sacfactory.demo_preparation import restore_inputs
            s=restore_inputs(s)
            s=validate_sample(s)
            if model is not None:
                p=float(model.predict_proba([prediction_features(s,atoms)])[0,list(model.classes_).index(1)])
                planned={k:copy.deepcopy(s[k]) for k in ('origin','elements','concentrations_mM','pulse_count','pulse_limit_C','precursor_family','support_batch','protocol_id','precursors','pan_mass_g','solvent_volume_mL','carbonization_loss_fraction')}
                s['proposal']={'recipe':planned,'p_screen_pass':p,'created':s['result_reported_at'].split('T')[0]+'T00:00:00+00:00',
                    'model_id':fingerprint({'capture':capture,'seed':SEED,'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}),
                    'policy':'SIMULATED_SEQUENTIAL_STAGE_RF','evaluation_capture':capture,
                    'purpose':'Source-only stage forecast; fixed 144-record budget, not a BO benchmark'}
            samples.append(s)
        phase=samples[-len(recipes):]
        phases.append({'elements':list(elements),'n_samples':len(phase),'first_batch':phase[0]['batch_id'],'last_batch':phase[-1]['batch_id']})
        if len(elements)==1:
            assert any(screen(s,{})['label']==1 for s in phase),f'No candidate in fixed simulated stage {elements}'
    counts={}
    for s in samples:
        result=screen(s,{}); assert not result['review_conflicts']
        counts[result['status']]=counts.get(result['status'],0)+1
    context=copy.deepcopy(next(s for s in samples if s['elements']==['Fe'] and s['pulse_limit_C']==1000 and s['concentrations_mM']['Fe']==50))
    return {'demo_version':'synthetic-physics-linked-campaign-v6','loss_model_version':loss_settings()['version'],
        'loss_settings_sha256':loss_settings()['settings_sha256'],'loss_reference_sha256':loss_reference()['snapshot_sha256'],
        'observation_settings':settings,'design_sha256':hashlib.sha256(raw).hexdigest(),'random_seed':SEED,
        'unique_recipes':len({recipe_key(s) for s in samples}),'description':'Frozen 1476 simulated recipes. Temperature-dependent mass retention, batch/process variability, noisy/censored ICP and retained-molar-burden XRD surrogate. Explicit workflow exceptions; no sample-index-triggered loss.',
        'status_counts':counts,'phase_summaries':phases,'methods':[],'samples':samples,'thermal_reference_sha256':thermal_reference()['snapshot_sha256'],
        'planning_context':context,'planning_space':{'concentration_bounds_mM':{'Fe':{'min_mM':15.,'max_mM':150.}},'max_total_mM':150.,
            'pulses':[1,2,3,4,6,10],'pulse_limit_min_C':850.,'pulse_limit_max_C':2000.,'verified':True,
            'range_source':'SIMULATED 20 mL conversion; not recommended safe limits','element_transfer':True,
            'model_engine':'reliability','descriptor_profile':'chemistry_ion','learning_iterations':4,'batch_effect':True,'baseline_repeat':True}}


def main():
    content=make_demo()
    assert len(content['samples'])==1476 and content['unique_recipes']>1000
    (ROOT/'examples/demo_seed.json').write_text(json.dumps(content,allow_nan=False,separators=(',',':')),encoding='utf-8')
    print(json.dumps({'demo_version':content['demo_version'],'samples':len(content['samples']),
        'unique_recipes':content['unique_recipes'],'counts':content['status_counts'],
        'saved_predictions':sum('proposal' in s for s in content['samples']),'stages':len(content['phase_summaries'])}))


if __name__=='__main__':
    main()
