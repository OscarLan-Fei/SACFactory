"""Reusable, parameterized workflow patterns; building a preview has no side effects."""
import copy
import math

from .device_catalog import PROFILES,modalities_of
from .domain import ELEMENTS
from .workflow_engine import validate_spec


def _role(key,profile,capability=None,modalities=None,automatic=False):
    p=PROFILES[profile]
    return dict(role=key,profile=profile,name=p['name'],capability=capability or p['capabilities'][0],
                modalities=modalities if modalities is not None else p['modalities'],automatic=automatic)


ACQUISITION=[_role('heater','pulsed_heater',automatic=True),_role('icp','icp_oes',automatic=True),
             _role('xrd','xrd',automatic=True),_role('stem','stem',automatic=True),_role('sem','sem',automatic=True)]
MANUAL_ACQUISITION=[{**r,'automatic':False} for r in ACQUISITION]


def _entry(ident,name,category,description,roles=(),origins=('REAL','DEMO'),needs_records=False,needs_artifacts=False):
    return dict(id=ident,name=name,category=category,description=description,roles=list(roles),
                origins=list(origins),needs_records=needs_records,needs_artifacts=needs_artifacts)


TEMPLATES=[
    _entry('simulation_loop','Complete simulated SAC-Gen loop','Closed loop',
           'The existing 16-step design, independent acquisition, reviewed fixture feedback, validation and candidate-update loop.',ACQUISITION,('DEMO',)),
    _entry('simulated_screening','Simulated synthesis and characterization','Synthesis & characterization',
           'Design and freeze a batch, then synthesize and collect independent ICP/XRD/STEM/SEM fixture results without training.',ACQUISITION,('DEMO',)),
    _entry('inverse_screening','Recipe design to Pending','Design & learning',
           'Retrieve evidence, propose target-conditioned recipes, freeze forecasts and wait for operator review.'),
    _entry('multielement_bridge','Multielement bridge experiments','Design & learning',
           'Propose mixtures from at least two selected elements and require a documented bridge-experiment review before synthesis.'),
    _entry('data_benchmark','Selected-data validation and comparison','Design & learning',
           'Run protected recipe-group validation and a separate model-opinion diagnostic on the selected records.',needs_records=True),
    _entry('candidate_update','Validate and train a candidate','Design & learning',
           'Validate selected records, wait for review, fit a versioned candidate and compare predictions. Primary deployment is unchanged.',needs_records=True),
    _entry('precursor_preparation','Precursor preparation and electrospinning','Synthesis & characterization',
           'Operator-led weighing, dispensing, mixing and fiber preparation using the laboratory\'s approved methods.',
           [_role('balance','balance'),_role('dispenser','liquid_handler'),_role('mixer','stirrer'),_role('spinner','electrospinning')]),
    _entry('manual_synthesis','Operator-led pulsed synthesis','Synthesis & characterization',
           'Design and freeze specimens, prepare precursors and document a manually performed synthesis.',
           [_role('balance','balance'),_role('heater','pulsed_heater')]),
    _entry('reviewed_characterization','Review registered characterization evidence','Synthesis & characterization',
           'Collect already registered artifacts, wait for explicit evidence review and commit only the reviewed assessment.',
           needs_records=True,needs_artifacts=True),
    _entry('characterization_panel','Extended characterization panel','Synthesis & characterization',
           'Operator-led ICP, XRD, STEM, SEM, XPS, XAS, Raman and BET acquisition. Unimplemented modalities retain manual notes.',
           [_role('icp','icp_oes'),_role('xrd','xrd'),_role('stem','stem'),_role('sem','sem'),_role('xps','xps'),
            _role('xas','xas'),_role('raman','raman'),_role('bet','bet')],needs_records=True),
    _entry('electrochemical_performance','Electrochemical catalyst evaluation','Performance testing',
           'Document the approved method, electrode preparation, electrochemical measurements, durability and product analysis.',
           [_role('balance','balance'),_role('potentiostat','potentiostat'),_role('rrde','rrde'),_role('products','gc')],needs_records=True),
    _entry('gas_catalysis','Gas-phase catalyst evaluation','Performance testing',
           'Operator-led reactor setup, product analysis, blank/reference controls and material-balance review.',
           [_role('reactor','gas_reactor'),_role('products','gc')],needs_records=True),
    _entry('real_feedback_loop','Operator-assisted synthesis feedback loop','Closed loop',
           'Freeze a designed batch, perform synthesis and characterization manually, review registered evidence and evaluate the frozen forecasts.',
           [_role('balance','balance')]+MANUAL_ACQUISITION),
    _entry('confirmed_synthesis_loop','Confirmed material synthesis and learning loop','Closed loop',
           'Review a calculated plan before dispatch; prepare, weigh, dispense, mix, electrospin, heat, characterize and test performance; update a candidate from historical and returned records and propose the next group.',
           [_role('balance','balance'),_role('dispenser','liquid_handler'),_role('mixer','stirrer'),_role('spinner','electrospinning'),_role('potentiostat','potentiostat')]+MANUAL_ACQUISITION),
]
INDEX={t['id']:t for t in TEMPLATES}
OPTION_KEYS={'name','elements','batch_size','bounds_mmol','temperature','pulses','model','candidate_family','epochs',
             'folds','comparison_models','sample_ids','artifact_ids','bindings','register_resources','allow_extrapolation',
             'pan_mass_g','solvent_volume_mL','voltage_V','current_A','carbonization_loss_fraction','scheduling','composition_mode','performance_protocol'}
PREPARATION_DEFAULTS={'pan_mass_g':2.,'solvent_volume_mL':20.,'voltage_V':32.,'current_A':80.,'carbonization_loss_fraction':.06}
DEFAULT_DEVICES={'heater':'demo_heater','icp':'demo_icp','xrd':'demo_xrd','stem':'demo_stem','sem':'demo_sem'}


def catalog(origin=None):
    return copy.deepcopy([t for t in TEMPLATES if origin is None or origin in t['origins']])


def compatible(role,device,origin):
    if role['capability'] not in device.get('capabilities',[]):return False
    if any(m not in modalities_of(device) for m in role['modalities']):return False
    if not role['modalities'] and device.get('equipment_type','custom') not in {'custom',role['profile']}:return False
    if role['automatic']:return origin=='DEMO' and device['mode']=='SIMULATOR'
    return origin=='DEMO' or device['mode']!='SIMULATOR'


def build(template_id,origin,options=None,selected_ids=None):
    if template_id not in INDEX:raise ValueError('Unknown workflow template.')
    if options is not None and not isinstance(options,dict):raise ValueError('Template options must be an object.')
    template=INDEX[template_id];o=copy.deepcopy(options or {})
    if origin not in template['origins']:raise ValueError('This template is available in '+', '.join(template['origins'])+' only.')
    if not isinstance(o,dict) or set(o)-OPTION_KEYS:raise ValueError('Unexpected workflow-template options.')
    elements=o.get('elements',['Fe'])
    if not isinstance(elements,list) or not elements or any(e not in ELEMENTS for e in elements) or len(set(elements))!=len(elements):raise ValueError('Choose distinct supported elements.')
    if template_id=='multielement_bridge' and len(elements)<2:raise ValueError('Choose at least two elements for bridge experiments.')
    selected=list(selected_ids or [])
    ids=o.get('sample_ids',selected if template['needs_records'] else [])
    if not isinstance(ids,list) or any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids) or set(ids)-set(selected):raise ValueError('Template specimens must belong to the selected source records.')
    if template['needs_records'] and not ids:raise ValueError('Select source experiment records for this template.')
    artifacts=o.get('artifact_ids',[])
    if not isinstance(artifacts,list) or any(not isinstance(a,str) or not a for a in artifacts) or len(set(artifacts))!=len(artifacts):raise ValueError('Choose distinct registered artifact IDs.')
    if template['needs_artifacts'] and not artifacts:raise ValueError('Select registered characterization artifacts first.')
    bindings=o.get('bindings',{})
    if not isinstance(bindings,dict) or set(bindings)-{r['role'] for r in template['roles']} or any(not isinstance(d,str) or not d for d in bindings.values()):raise ValueError('Bind known equipment roles to instrument IDs.')
    if 'register_resources' in o and type(o['register_resources']) is not bool:raise ValueError('register_resources must be boolean.')
    if 'allow_extrapolation' in o and type(o['allow_extrapolation']) is not bool:raise ValueError('allow_extrapolation must be boolean.')
    if o.get('composition_mode','independent') not in {'independent','equimolar'}:raise ValueError('Choose independent or equimolar precursor feeds.')
    if 'performance_protocol' in o and (not isinstance(o['performance_protocol'],str) or not o['performance_protocol'].strip() or len(o['performance_protocol'])>4000):raise ValueError('Provide a nonempty performance protocol (up to 4000 characters).')
    for field,default in PREPARATION_DEFAULTS.items():
        value=o.get(field,default)
        if type(value) not in (int,float) or not math.isfinite(value) or (not 0<=value<1 if field=='carbonization_loss_fraction' else value<=0):raise ValueError('Invalid preparation parameter: '+field)
    bounds=o.get('bounds_mmol',{e:[.2,2.4] for e in elements})
    if not isinstance(bounds,dict) or set(bounds)!=set(elements) or any(not isinstance(v,list) or len(v)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in v) or not 0<v[0]<=v[1] for v in bounds.values()):raise ValueError('Provide positive finite mmol bounds for each selected element.')
    model=o.get('model','Fixed prior');family=o.get('candidate_family','SAC-Gen-Set')
    for key,default,low,high in [('batch_size',6,1,48),('epochs',20,1,500),('folds',5,2,10)]:
        value=o.get(key,default)
        if type(value) is not int or not low<=value<=high:raise ValueError(f'{key} requires an integer in [{low},{high}].')
    temperature=o.get('temperature',[800,1600]);pulses=o.get('pulses',[1,2,4,8,12])
    if not isinstance(temperature,list) or len(temperature)!=2 or any(type(t) not in (int,float) or not math.isfinite(t) for t in temperature) or not 0<temperature[0]<=temperature[1]<4000:raise ValueError('Temperature bounds must be within 0-4000 C.')
    if not isinstance(pulses,list) or not pulses or any(type(p) is not int or not 1<=p<=1000 for p in pulses):raise ValueError('Pulse counts must be positive integers up to 1000.')
    if not isinstance(model,str) or not model.strip():raise ValueError('Choose a model.')
    if family not in {'SAC-Gen-Set','SAC-Gen-GB','SAC-Gen-Diffusion','Calibrated discovery ensemble'}:raise ValueError('Choose a supported candidate family.')
    steps=[]
    def add(key,op,params,deps=()):
        steps.append(dict(id=key,operation=op,parameters=params,depends_on=list(deps)))
        return key
    def ref(path):return {'$ref':path}
    def manual(key,instruction,deps=(),role=None,samples=None):
        params={'instruction':instruction}
        if samples is not None:params['sample_ids']=samples
        if role:
            requirement=next(r for r in template['roles'] if r['role']==role)
            params.update(capability=requirement['capability'],equipment_type=requirement['profile'])
            if requirement['modalities']:params['modalities']=requirement['modalities']
            if role in bindings:params['device_id']=bindings[role]
        return add(key,'manual.checkpoint',params,deps)
    def design():
        add('memory','knowledge.search',{'query':' '.join(elements)+' precursor retention aggregation synthesis evidence'})
        add('design','inverse.design',{'elements':elements,'bounds_mmol':bounds,'temperature':o.get('temperature',[800,1600]),
             'pulses':o.get('pulses',[1,2,4,8,12]),'batch_size':o.get('batch_size',6),'count':96,'model':model,'allow_extrapolation':o.get('allow_extrapolation',False)},['memory'])
        add('pending','campaign.plan',{'design':ref('design.design')},['design'])
        return ref('pending.sample_ids')
    if template_id=='confirmed_synthesis_loop':
        add('memory','knowledge.search',{'query':' '.join(elements)+' precursor retention aggregation synthesis evidence','include_history':True})
        search={'elements':elements,'bounds_mmol':bounds,'temperature':temperature,'pulses':pulses,
                'batch_size':o.get('batch_size',6),'count':96,'model':model,
                'composition_mode':o.get('composition_mode','independent'),'allow_extrapolation':o.get('allow_extrapolation',False)}
        add('design','inverse.design',search,['memory'])
        protocol=o.get('performance_protocol')
        add('confirm_plan','manual.checkpoint',{'instruction':'Review recipes, schedule, equipment bindings and the performance protocol. Confirm explicitly before creating Pending specimens or dispatching equipment.',
            'approval_required':True,'return_fields':['approved','performance_protocol'],
            'plan':{'design':ref('design.design'),'preparation':{k:o.get(k,v) for k,v in PREPARATION_DEFAULTS.items()},
                    'performance_protocol':protocol,'composition_mode':search['composition_mode'],'equipment':bindings}},['design'])
        add('pending','campaign.plan',{'design':ref('design.design')},['confirm_plan'])
        target=ref('pending.sample_ids')
        manual('prepare','Check the approved recipe, reagent identity, hydrate, stock concentration and preparation SOP.',['pending'],samples=target)
        steps[-1]['parameters']['return_fields']=['method_id','reagent_check']
        previous='prepare'
        for key,role,instruction,fields in [
            ('weigh','balance','Weigh the approved precursor amounts and PAN. Return actual quantities; deviations require review.',['pan_mass_g','precursor_amounts_mmol']),
            ('dispense','dispenser','Dispense solvent and record the actual volume.',['solvent_volume_mL']),
            ('mix','mixer','Mix under the approved method; return the duration and observations.',['method_id','duration_min']),
            ('electrospin','spinner','Electrospin using the approved instrument SOP; return settings and recovered fiber mass.',['method_id','settings','fiber_mass_g'])]:
            previous=manual(key,instruction,[previous],role,target);steps[-1]['parameters']['return_fields']=fields
        if origin=='DEMO':
            add('heating','device.synthesize',{'device_id':bindings.get('heater','demo_heater'),'sample_ids':target},[previous])
            acquisitions=[]
            for modality in ['ICP','XRD','STEM','SEM']:
                ident=modality.lower();add(ident,'device.characterize',{'device_id':bindings.get(ident,'demo_'+ident),'synthesis':ref('heating.receipt'),'modalities':[modality]},['heating']);acquisitions.append(ident)
            add('recognition','characterization.analyze',{'measurements':{'acquisitions':[ref(i+'.acquisition') for i in acquisitions]}},acquisitions)
            manual('review','Review the independent raw exports and analysis before committing the returned fixture evidence.',['recognition'],samples=target)
            add('commit','characterization.fixture_review',{'evidence':ref('recognition.evidence')},['review'])
        else:
            manual('heating','Execute the approved heating SOP. Record actual controls, electrical trace and specimen recovery.',[previous],'heater',target)
            previous='heating'
            for modality in ['ICP','XRD','STEM','SEM']:
                previous=manual(modality.lower(),'Acquire '+modality+' raw exports and register the data in Characterization / Samples & data. A completion note does not import measurements.',[previous],modality.lower(),target)
            manual('review','Review registered structural evidence and ICP measurements before committing assessments.',[previous],samples=target)
            add('commit','characterization.commit',{'sample_ids':target,'state':'FROM_REVIEWED_EVIDENCE','reviewer':'WORKFLOW_OPERATOR','criteria':'Reviewed registered structural evidence.'},['review'])
        manual('performance','Perform only the explicitly approved performance protocol. Return method, units and results; no activity is inferred from ICP or a structural PASS.',['commit'],'potentiostat',target)
        steps[-1]['parameters']['return_fields']=['protocol_id','results','units','raw_data_reference']
        steps[-1]['parameters']['plan']={'approved_performance_protocol':ref('confirm_plan.returned_data.performance_protocol')}
        add('feedback','campaign.evaluate',{},['performance'])
        add('training_pool','records.training_pool',{'sample_ids':target,'include_history':True},['feedback'])
        add('validation','validation.run',{'sample_ids':ref('training_pool.sample_ids'),'folds':o.get('folds',5),'epochs':o.get('epochs',20),'include_sac':False},['training_pool'])
        add('update','model.train',{'sample_ids':ref('training_pool.sample_ids'),'family':family,'epochs':o.get('epochs',20)},['validation'])
        add('next_design','inverse.design',{**search,'model':family,'seed':43},['update'])
    elif template_id in {'simulation_loop','simulated_screening'}:
        from .workflow_service import simulation_template
        spec=simulation_template(elements,o.get('batch_size',32),o.get('epochs',10))
        steps=spec['steps']
        for node in steps:
            if node['id']=='design':node['parameters'].update(bounds_mmol=bounds,model=model,temperature=o.get('temperature',[800,1600]),pulses=o.get('pulses',[1,2,4,8,12]))
            if node['operation'].startswith('device.'):
                role='heater' if node['operation']=='device.synthesize' else node['parameters']['modalities'][0].lower()
                node['parameters']['device_id']=bindings.get(role,DEFAULT_DEVICES[role])
        if template_id=='simulated_screening':steps=steps[:next(i+1 for i,n in enumerate(steps) if n['id']=='feedback')]
    elif template_id in {'inverse_screening','multielement_bridge'}:
        target=design()
        manual('review','Review the frozen recipes, support compatibility and allowed operating range before performing synthesis. Bridge experiments test transfer; single-element success does not establish mixture success.',['pending'],samples=target)
    elif template_id=='data_benchmark':
        add('validation','validation.run',{'sample_ids':ids,'folds':o.get('folds',5),'epochs':o.get('epochs',20),'include_sac':False})
        add('comparison','model.compare',{'sample_ids':ids,'models':o.get('comparison_models',[model])},['validation'])
    elif template_id=='candidate_update':
        add('validation','validation.run',{'sample_ids':ids,'folds':o.get('folds',5),'epochs':o.get('epochs',20),'include_sac':False})
        manual('review','Review fold eligibility, held-out errors and evidence provenance before training a deployment candidate.',['validation'],samples=ids)
        add('update','model.train',{'sample_ids':ids,'family':family,'epochs':o.get('epochs',50)},['review'])
        add('comparison','model.compare',{'sample_ids':ids,'models':list(dict.fromkeys([model,family]))},['update'])
    elif template_id=='precursor_preparation':
        previous=[]
        for key,role,instruction in [('weigh','balance','Weigh precursors and PAN; record reagent identity, hydrate, purity and actual masses.'),
              ('dispense','dispenser','Dispense DMF and record actual volume. Convert mmol to mM using the recorded solution volume.'),
              ('mix','mixer','Mix using the approved laboratory method. Record method ID and deviations.'),
              ('spin','spinner','Prepare fibers using the approved electrospinning SOP. Record settings, traceability and recovery.')]:
            previous=[manual(key,instruction,previous,role,ids)]
    elif template_id in {'manual_synthesis','real_feedback_loop'}:
        target=design()
        manual('prepare','Prepare the frozen recipes, verify precursor quantities and record actual preparation parameters.',['pending'],'balance',target)
        manual('synthesis','Perform synthesis under the approved equipment SOP. Record actual electrical settings, controlled temperature, pulses and specimen recovery.',['prepare'],'heater',target)
        if template_id=='real_feedback_loop':
            previous='synthesis'
            for modality in ['ICP','XRD','STEM','SEM']:
                previous=manual(modality.lower(),'Acquire '+modality+' measurements for the frozen specimens. Enter ICP W / wt% in Samples & data; register structural raw exports in Characterization. A completion note does not import measurements.',[previous],modality.lower(),target)
            manual('review','Review the registered XRD/STEM evidence and record a supported assessment in Characterization. Acknowledgement does not infer PASS/FAIL.',[previous],samples=target)
            add('commit','characterization.commit',{'sample_ids':target,'state':'FROM_REVIEWED_EVIDENCE','reviewer':'WORKFLOW_OPERATOR','criteria':'Recorded, reviewed structural evidence; raw measurements remain separate.'},['review'])
            add('feedback','campaign.evaluate',{},['commit'])
    elif template_id=='reviewed_characterization':
        add('analysis','characterization.analyze',{'measurements':{'artifact_ids':artifacts}})
        target=ref('analysis.evidence.sample_ids')
        manual('review','Review the registered evidence in Characterization and record observations before continuing.',['analysis'],samples=target)
        add('commit','characterization.commit',{'sample_ids':target,'state':'FROM_REVIEWED_EVIDENCE','reviewer':'WORKFLOW_OPERATOR','criteria':'Previously reviewed registered evidence.'},['review'])
    elif template_id=='characterization_panel':
        previous=[]
        for role in template['roles']:
            previous=[manual(role['role'],'Acquire '+role['name']+' data under the approved method. Preserve raw exports and record method/calibration details. Only supported raw formats can be analyzed automatically.',previous,role['role'],ids)]
        manual('review','Cross-check chemistry, structure and composition using the recorded evidence. Document limitations and conflicts; do not substitute a completion note for measurement data.',previous,samples=ids)
    elif template_id=='electrochemical_performance':
        manual('method','Record the approved reaction, electrolyte, reference calibration, electrode geometry, loading and protocol ID.',samples=ids)
        manual('electrode','Prepare electrodes and document catalyst and binder masses, area and drying history.',['method'],'balance',ids)
        manual('electrochemistry','Acquire raw electrochemical curves using the approved protocol. Preserve reference scale, units and compensation settings.',['electrode'],'potentiostat',ids)
        manual('transport','Perform the planned rotating-electrode measurements and document hydrodynamic conditions and controls.',['electrochemistry'],'rrde',ids)
        manual('products','Measure products as appropriate to the selected reaction; document calibration, background and detection limits.',['transport'],'products',ids)
        manual('review','Review activity, selectivity, durability, controls and uncertainty. Store raw files and measured values separately from these workflow notes.',['products'],samples=ids)
    elif template_id=='gas_catalysis':
        manual('method','Record the approved reaction protocol, gas identities, catalyst mass, flow calibration and operating limits.',samples=ids)
        manual('reaction','Perform the approved catalytic reactor run and retain actual temperature, flow and time traces.',['method'],'reactor',ids)
        manual('products','Acquire calibrated product analysis with blank/reference controls and raw chromatograms.',['reaction'],'products',ids)
        manual('review','Review conversion, selectivity, material balance and stability using measured records and uncertainty. Workflow notes do not generate performance labels.',['products'],samples=ids)
    scheduling=o.get('scheduling',{})
    if not isinstance(scheduling,dict) or set(scheduling)-{n['id'] for n in steps}:raise ValueError('Scheduling keys must name steps in this template.')
    from .workflow_schedule import validate_schedule
    for node in steps:
        if node['id'] in scheduling:node['schedule']=validate_schedule(scheduling[node['id']])
    spec={'name':o.get('name',template['name']),'version':1,'description':template['description'],'steps':steps}
    return validate_spec(spec)
