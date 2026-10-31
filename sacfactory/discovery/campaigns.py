"""Immutable pre-result forecast rounds in a separate namespace database."""
import copy
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4
from .common import digest,identity,outcomes,prediction_metrics,query,recipe_id,stamp


class CampaignLedger:
    def __init__(self,root,origin):
        if origin not in {'REAL','DEMO'}:raise ValueError('Invalid namespace.')
        self.origin=origin;self.path=Path(root)/f'discovery_{origin.lower()}.sqlite3'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, kind TEXT, created TEXT, payload TEXT)')

    def add(self,kind,payload):
        if payload.get('origin')!=self.origin:raise ValueError('Cross-namespace discovery data are forbidden.')
        row={**copy.deepcopy(payload),'id':uuid4().hex,'created':stamp()}
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('INSERT INTO events VALUES(?,?,?,?)',(row['id'],kind,row['created'],json.dumps(row,allow_nan=False)))
        return row

    def records(self,kind):
        with closing(sqlite3.connect(self.path)) as db, db:
            return [json.loads(r[0]) for r in db.execute('SELECT payload FROM events WHERE kind=? ORDER BY created,id',(kind,))]

    def freeze(self,samples,model_metadata):
        if any(s.get('origin')!=self.origin for s in samples):raise ValueError('Do not mix data spaces.')
        return self.add('round',{'origin':self.origin,'snapshot':copy.deepcopy(samples),
            'dataset_fingerprint':digest(samples),'training_keys':[identity(s) for s in samples],
            'model_metadata':copy.deepcopy(model_metadata),'state':'FROZEN'})

    def propose(self,round_id,recipes,predictions,training_keys=(),rationales=None):
        frozen=next((r for r in self.records('round') if r['id']==round_id),None)
        if frozen is None or len(recipes)!=len(predictions) or not recipes:
            raise ValueError('Select a frozen round and aligned nonempty forecasts.')
        for s in recipes:
            if s.get('origin')!=self.origin:raise ValueError('Namespace differs.')
            if outcomes(s)!=( {},None) or s.get('result_reported_at'):
                raise ValueError('Only pre-result recipes can receive prospective forecasts.')
        import math
        from .common import values
        for s,p in zip(recipes,predictions):
            content=values(p);probability=p.get('p_xrd_pass')
            if set(content)!=set(s['elements']) or any(not math.isfinite(v) or not 0<=v<=100 for v in content.values()):
                raise ValueError('Freeze complete finite physical ICP forecasts only.')
            if probability is not None and (not math.isfinite(probability) or not 0<=probability<=1):
                raise ValueError('A structural probability must be between zero and one.')
        return self.add('proposal',{'origin':self.origin,'round_id':round_id,'recipes':[query(s) for s in recipes],
            'recipe_keys':[recipe_id(s) for s in recipes],'predictions':copy.deepcopy(predictions),
            'training_keys':list(training_keys),'rationales':rationales or [],'mode':'PRE_RESULT_FORECAST'})

    def link(self,proposal_id,samples):
        proposal=next((p for p in self.records('proposal') if p['id']==proposal_id),None)
        if proposal is None or len(samples)!=len(proposal['recipes']):raise ValueError('Proposal and pending rows must align.')
        if any(s.get('origin')!=self.origin or recipe_id(s)!=k or outcomes(s)!=( {},None) or s.get('result_reported_at')
               or identity(s) in proposal['training_keys'] for s,k in zip(samples,proposal['recipe_keys'])):
            raise ValueError('Link unchanged unmeasured pending records outside the training snapshot.')
        if any(e['proposal_id']==proposal_id for e in self.records('link')):raise ValueError('Proposal already linked.')
        return self.add('link',{'origin':self.origin,'proposal_id':proposal_id,'record_keys':[identity(s) for s in samples]})

    def evaluate(self,samples):
        current={identity(s):s for s in samples if s.get('origin')==self.origin};results=[]
        aliases={r['previous_key']:r['record_key'] for r in self.records('identity_binding')}
        proposals={p['id']:p for p in self.records('proposal')}
        for link in self.records('link'):
            p=proposals[link['proposal_id']];actual=[];pred=[];deviated=missing=0
            for key,recipe,forecast in zip(link['record_keys'],p['recipe_keys'],p['predictions']):
                sample=current.get(aliases.get(key,key))
                if sample is None:missing+=1;continue
                if recipe_id(sample)!=recipe:deviated+=1;continue
                if not outcomes(sample)[0] and outcomes(sample)[1] is None:continue
                actual.append(sample);pred.append(forecast)
            results.append({'proposal_id':p['id'],'round_id':p['round_id'],'created':p['created'],
                            'changed_recipes_excluded':deviated,'missing_records':missing,**prediction_metrics(actual,pred)})
        return results
