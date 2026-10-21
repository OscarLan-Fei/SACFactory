"""Finite-pool acquisition replay with an untouched batch evaluation set."""
import copy
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .weighted_validation import collect
from .model_benchmarks import fitted_baseline,metrics
from .domain import recipe_key
from .compute_runtime import emit
from .planning import fingerprint

POLICIES=['Random','Space filling','GP-BO expected improvement','GP-BO upper confidence bound','SAC-Gen uncertainty']
VERSION='recorded-pool-acquisition-replay-v1'


def controls(sample):
    result={k:copy.deepcopy(sample.get(k)) for k in ('sample_id','origin','elements','concentrations_mM','pulse_count',
        'pulse_limit_C','precursor_family','precursors','support_batch','protocol_id','pan_mass_g',
        'solvent_volume_mL','carbonization_loss_fraction','carbonization_loss') if k in sample}
    result.update(icp=[],xrd={'state':'PENDING'},icp_review={'state':'PENDING'})
    return result


def replay(samples,methods,origin,policies=None,initial=32,budget=48,interval=4,seed=42,
           profile='chemistry_pulseicp',sac_family='Diffusion Transformer',sac_settings=None):
    policies=list(dict.fromkeys(policies or POLICIES))
    if not policies or set(policies)-set(POLICIES):raise ValueError('Choose acquisition policies.')
    if not 4<=initial<=budget or interval<1:raise ValueError('Use an initial budget of at least four and a total budget no smaller than the initial budget.')
    records,audit=collect(samples,methods,origin,profile)
    # The first stored record is chosen by ID, never by its outcome.
    unique={}
    for r in sorted(records,key=lambda r:r['sample']['sample_id']):
        if r['outcomes']['xrd_pass'] is not None:unique.setdefault(recipe_key(r['sample']),r)
    records=list(unique.values())
    if len({r['sample']['batch_id'] for r in records})<4:raise ValueError('Acquisition replay needs at least four recorded batches.')
    source,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(records,
        groups=[r['sample']['batch_id'] for r in records]))
    pool=[records[i] for i in source];held=[records[i] for i in test]
    if budget>len(pool):raise ValueError(f'The candidate pool has {len(pool)} unique recipes after holding out batches; reduce the budget.')
    x=np.asarray([r['x'] for r in pool]);distance_x=StandardScaler().fit_transform(x)
    start=np.random.default_rng(seed).choice(len(pool),int(initial),replace=False).tolist()
    queries=[];curves=[];predictions=[];fallbacks=[]
    all_progress=len(policies)*max(1,int(np.ceil((budget-initial)/interval))+1);progress=0
    for policy in policies:
        selected=start.copy();rng=np.random.default_rng(seed)
        while True:
            revealed=[pool[i] for i in selected]
            emit({'stage':f'{policy}: {len(selected)} observations','fold':progress+1,'folds':all_progress});progress+=1
            values,detail=fitted_baseline(revealed,held,'Random forest',profile,seed)
            frame=pd.DataFrame(values).assign(fold='FIXED_BATCH_HOLDOUT')
            result=metrics(frame)
            curves.append({'policy':policy,'observations':len(selected),**result,
                'observed_XRD_PASS':sum(r['outcomes']['xrd_pass']==1 for r in revealed),
                'observed_XRD_FAIL':sum(r['outcomes']['xrd_pass']==0 for r in revealed),
                'observed_elements':len({e for r in revealed for e in r['sample']['elements']})})
            predictions.extend({**v,'policy':policy,'observations':len(selected),'fold':'FIXED_BATCH_HOLDOUT'} for v in values)
            if len(selected)>=budget:break
            remaining=[i for i in range(len(pool)) if i not in selected]
            batch=min(interval,budget-len(selected),len(remaining))
            diversity=cdist(distance_x[remaining],distance_x[selected]).min(axis=1)
            reason=None
            if policy=='Random':score=rng.random(len(remaining))
            elif policy=='Space filling':score=diversity
            elif policy.startswith('GP-BO'):
                gp=make_pipeline(StandardScaler(),GaussianProcessRegressor(
                    kernel=Matern(length_scale=np.sqrt(x.shape[1]),nu=2.5)+WhiteKernel(.05),
                    optimizer=None,normalize_y=True,alpha=1e-5,random_state=seed))
                y=np.asarray([pool[i]['outcomes']['xrd_pass'] for i in selected],dtype=float)
                gp.fit(x[selected],y)
                mean,sd=gp.predict(x[remaining],return_std=True)
                if policy=='GP-BO upper confidence bound':score=mean+2*sd
                else:
                    improvement=mean-float(y.max())-.01
                    z=improvement/np.maximum(sd,1e-12)
                    score=improvement*norm.cdf(z)+sd*norm.pdf(z)
            else:
                from .sac_gen import SACGen
                try:
                    # No full-data checkpoint is loaded into a replay.
                    model=SACGen(sac_family,sac_settings,profile).fit([r['sample'] for r in revealed],origin,methods)
                    forecasts=model.predict([controls(pool[i]['sample']) for i in remaining])
                    probability=np.asarray([v['p_xrd_pass'] if v['p_xrd_pass'] is not None else np.nan for v in forecasts])
                    p=np.clip(probability,1e-12,1-1e-12)
                    score=-(p*np.log(p)+(1-p)*np.log1p(-p))
                    if not np.isfinite(score).all():reason='Missing trained probabilities; use space filling for unsupported candidates.'
                    score=np.where(np.isfinite(score),score,diversity/(1+diversity))
                except (ValueError,KeyError,TypeError,RuntimeError) as exc:
                    score=diversity;reason=str(exc)
            if reason:fallbacks.append({'policy':policy,'observations':len(selected),'reason':reason})
            # Greedy diversity within the proposed group; no unobserved outcome is revealed early.
            chosen=[]
            for slot in range(batch):
                adjustment=np.zeros(len(remaining)) if not chosen else cdist(distance_x[remaining],distance_x[chosen]).min(axis=1)
                if policy=='Random':combined=score.copy()
                elif policy=='Space filling':combined=score if not chosen else np.minimum(score,adjustment)
                else:combined=score+(0 if not chosen else .05*adjustment/(1+adjustment))
                combined=np.where(np.isin(remaining,chosen),-np.inf,combined)
                chosen.append(remaining[int(np.argmax(combined))])
            for ident in chosen:
                queries.append({'policy':policy,'query_number':len(selected)+1,'sample_id':pool[ident]['sample']['sample_id'],
                    'batch_id':pool[ident]['sample']['batch_id']})
                selected.append(ident)
    return pd.DataFrame(curves),pd.DataFrame(queries),pd.DataFrame(predictions),{
        'version':VERSION,'origin':origin,'dataset_fingerprint':fingerprint(samples),'seed':seed,
        'profile':profile,'candidate_ids':[r['sample']['sample_id'] for r in pool],
        'test_ids':[r['sample']['sample_id'] for r in held],
        'candidate_batches':sorted({r['sample']['batch_id'] for r in pool}),
        'test_batches':sorted({r['sample']['batch_id'] for r in held}),
        'initial_ids':[pool[i]['sample']['sample_id'] for i in start],'initial':initial,'budget':budget,'interval':interval,
        'policies':policies,'fallbacks':fallbacks,'sac_family':sac_family,'sac_settings':sac_settings,
        'common_evaluator':'Fixed random forest fitted on revealed observations only',
        'BO_objective':'Recorded XRD PASS indicator; EI xi=0.01, UCB beta=2; fixed GP kernel, no test tuning',
        'SAC_Gen_objective':'XRD predictive entropy; refit on revealed observations, no full-data weights',
        'note':'Retrospective finite recorded-pool replay, not prospective closed-loop experimental evidence. All policies share initial observations, candidate pool, observation budget, fixed test batches and evaluator. BO targets XRD PASS discovery; entropy targets boundary learning. Report held-out predictive error and PASS/FAIL discoveries separately, without declaring unlike objectives interchangeable. Recipe duplicates are collapsed by ID order before splitting. No unmeasured recipe receives a simulated ground-truth outcome.'}
