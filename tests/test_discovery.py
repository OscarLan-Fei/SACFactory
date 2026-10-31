"""Discovery scientific contracts, immutable feedback and leakage regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_pulseicp_bridge import specimens
from sacfactory.compute_runtime import execution
from sacfactory.storage import Store
from sacfactory.discovery.common import query,recipe_id,outcomes,identity,digest
from sacfactory.discovery.campaigns import CampaignLedger
from sacfactory.discovery.knowledge import document,search,history_documents,public_documents
from sacfactory.discovery.calibration import Calibration,disagreement,learn_weights,protected_weights,fuse
from sacfactory.discovery.inverse import candidates,rank
from sacfactory.discovery.hypotheses import contrasts,register,assess
from sacfactory.discovery.set_model import SetResidual
from sacfactory.discovery.decision import DecisionEnsemble
from sacfactory.discovery.workflows import train_candidate,load_model,pending_from_proposal
from sacfactory.discovery.validation import validate,replay


def forecast(rows,value=1.,p=.7):
    return [{'model_id':'test-model','model_name':'test','icp_predictions':[{'element':e,'predicted_mean_w_wt_percent':value} for e in s['elements']],
             'predicted_total_wt_percent':value*len(s['elements']),'p_xrd_pass':p,'outside_training_support':False} for s in rows]


class DiscoveryContracts(unittest.TestCase):
    def setUp(self):self.rows=specimens(70)

    def test_hypothesis_feedback_compares_instants_across_timezones(self):
        recipes=contrasts(self.rows[0],'temperature',(1200,))
        card=register(recipes,forecast(recipes),'temperature','Temperature changes retention.')
        card['created']='2026-01-01T10:00:00+00:00'
        rows=[]
        for index,recipe in enumerate(recipes):
            row=copy.deepcopy(recipe)
            row['icp']=[{'element':e,'value':1.+index,'unit':'wt%','flag':'MEASURED'} for e in row['elements']]
            row['result_reported_at']='2026-01-01T06:00:00-05:00'
            rows.append(row)
        result=assess(card,rows)
        self.assertTrue(result);self.assertEqual(result[0]['observed_change_wt_percent'],1.)
        rows[1]['result_reported_at']='2026-01-01T11:00:00'
        self.assertIsNone(assess(card,rows)[0]['observed_change_wt_percent'])

    def test_query_removes_all_outcomes_and_canonicalizes_units(self):
        clean=query(self.rows[0]);self.assertNotIn('icp',clean);self.assertNotIn('xrd',clean)
        base=copy.deepcopy(self.rows[0]);base['pulse_limit_C']='900'
        self.assertEqual(recipe_id(base),recipe_id(self.rows[0]))
        bad=copy.deepcopy(base);bad['concentrations_mM']['Fe']=float('nan')
        with self.assertRaises(ValueError):query(bad)

    def test_structural_fail_and_missing_values_are_not_conflated(self):
        s=copy.deepcopy(self.rows[0]);s['icp']=[];s['xrd']={'state':'FAIL','mode':'MANUAL'}
        self.assertEqual(outcomes(s),({},0))
        s['sample_condition']='EQUIPMENT_ABORT';self.assertEqual(outcomes(s),({},None))

    def test_local_sources_and_private_selection_are_explicit(self):
        docs=public_documents()+history_documents(self.rows[:1])
        found=search(docs,'Fe precursor temperature mass balance',5)
        self.assertTrue(found);self.assertTrue(all('source' in r for r in found))
        self.assertFalse(any('SAC_0002' in r['passage'] for r in found))
        self.assertEqual(search([document('English','carbon nitrogen','manual')],'zzzzzzzzzzzz',3),[])

    def test_candidate_amount_conversion_and_cutoff_grid(self):
        clean=query(self.rows[0]);pool=candidates(clean,['Fe'],{'Fe':(1.,1.)},(995,1025),[1],16)
        self.assertTrue(pool)
        self.assertTrue(all(r['concentrations_mM']['Fe']==50 for r in pool))
        self.assertTrue(all(995<=r['pulse_limit_C']<=1025 and r['pulse_limit_C']%10==0 for r in pool))
        self.assertEqual(len({recipe_id(r) for r in pool}),len(pool))
        with self.assertRaises(ValueError):candidates(clean,['Fe'],{'Fe':(1,1)},(1001,1009),[1],16)

    def test_inverse_filters_missing_probabilities_and_extrapolation(self):
        pool=candidates(query(self.rows[0]),['Fe'],{'Fe':(.1,1)},(900,1100),[1,2],16)
        forecasts=forecast(pool)
        forecasts[0]['p_xrd_pass']=None;forecasts[1]['outside_training_support']=True
        result=rank(pool,forecasts,(.5,1.5),.5,4)
        self.assertEqual(len(result),4);self.assertNotIn(recipe_id(pool[0]),[r['recipe_key'] for r in result])
        self.assertNotIn(recipe_id(pool[1]),[r['recipe_key'] for r in result])

    def test_ledger_is_immutable_pending_import_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder,'REAL');store.import_bundle(self.rows[:20],[])
            current=store.all('sample');ledger=CampaignLedger(folder,'REAL')
            frozen=ledger.freeze(current,{'model_id':'test-model'})
            recipes=[query(self.rows[21])];predictions=forecast(recipes)
            proposal=ledger.propose(frozen['id'],recipes,predictions,[identity(s) for s in current])
            predictions[0]['p_xrd_pass']=.1
            self.assertEqual(ledger.records('proposal')[0]['predictions'][0]['p_xrd_pass'],.7)
            ids=pending_from_proposal(store,ledger,proposal['id']);self.assertEqual(len(ids),1)
            self.assertEqual(ids,pending_from_proposal(store,ledger,proposal['id']))
            pending=store.get('sample',ids[0]);self.assertEqual(outcomes(pending),({},None))
            pending.update(icp=self.rows[21]['icp'],xrd={'mode':'MANUAL','state':'PASS'},batch_id='BATCH_NEW')
            store.import_bundle([pending],[],overwrite=True)
            result=ledger.evaluate(store.all('sample'));self.assertEqual(result[0]['quantified_specimens'],1)
            pending['pulse_limit_C']+=10;store.import_bundle([pending],[],overwrite=True)
            self.assertEqual(ledger.evaluate(store.all('sample'))[0]['changed_recipes_excluded'],1)

    def test_cannot_freeze_measured_or_cross_namespace_recipe(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=CampaignLedger(folder,'REAL');frozen=ledger.freeze(self.rows[:2],{})
            with self.assertRaises(ValueError):ledger.propose(frozen['id'],self.rows[:1],forecast(self.rows[:1]))
            recipe=query(self.rows[0]);recipe['origin']='DEMO'
            with self.assertRaises(ValueError):ledger.propose(frozen['id'],[recipe],forecast([recipe]))

    def test_calibration_masks_and_small_group_gate(self):
        unique=[];seen=set()
        for s in self.rows:
            if recipe_id(s) not in seen:seen.add(recipe_id(s));unique.append(s)
        calibration=Calibration().fit(unique[:20],forecast(unique[:20]),unique[20:30],unique[30:40])
        self.assertIsNotNone(calibration.radius)
        with self.assertRaises(ValueError):Calibration().fit(unique[:10],forecast(unique[:10]),unique[:1])
        small=Calibration().fit(unique[:3],forecast(unique[:3]));self.assertIsNone(small.radius)
        self.assertIsNone(small.apply(forecast(unique[:1]))[0]['icp_predictions'][0]['calibrated_lower_wt_percent'])

    def test_weights_are_not_fit_to_protected_results(self):
        a=forecast(self.rows[:12],1.);b=forecast(self.rows[:12],2.)
        with self.assertRaises(ValueError):learn_weights(self.rows[:12],{'a':a,'b':b},self.rows[:1])
        weights,status=protected_weights(self.rows[:2],{'a':a[:2],'b':b[:2]})
        self.assertEqual(weights,{'a':.5,'b':.5});self.assertIn('NOT_FITTED',status)
        output=fuse({'a':a,'b':b},weights);self.assertAlmostEqual(output[0]['predicted_total_wt_percent'],1.5)
        self.assertGreater(disagreement(self.rows[:12],{'a':a,'b':b})[0]['ICP_model_SD_wt_percent'],0)

    def test_hypotheses_change_only_one_factor(self):
        recipes=contrasts(self.rows[0],'temperature',[1000,1200]);card=register(recipes,forecast(recipes),'temperature','Temperature affects retention')
        altered=copy.deepcopy(recipes);altered[1]['pulse_count']+=1
        with self.assertRaises(ValueError):register(altered,forecast(altered),'temperature','Invalid contrast')
        self.assertTrue(all(r['status']=='AWAITING_NEW_MEASUREMENTS' for r in assess(card,self.rows)))


class DiscoveryModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime=execution({'device':'cpu','threads':2,'cpu_jobs':2});cls.runtime.__enter__()
        cls.rows=specimens(70);cls.model=SetResidual(epochs=8,seed=3).fit(cls.rows[:50],'REAL')

    @classmethod
    def tearDownClass(cls):cls.runtime.__exit__(None,None,None)

    def test_set_encoder_is_permutation_invariant_and_mass_bounded(self):
        s=query(self.rows[0]);s['elements']=['Fe','Co','Ni'];s['concentrations_mM']={'Fe':20.,'Co':30.,'Ni':40.}
        a,orders=self.model.encode([s]);permutation=torch.tensor([2,0,1]);b=[v.clone() for v in a]
        for i in [0,1,2,4,6]:b[i]=b[i][:,permutation]
        self.model.model.eval()
        with torch.no_grad():wa,pa=self.model.model(*a);wb,pb=self.model.model(*b)
        torch.testing.assert_close(wa[:,permutation],wb);torch.testing.assert_close(pa,pb)
        from sacfactory import pulseicp_bridge as bridge
        from sacfactory._pulseicp.mass_decoder import prior
        p=self.model.predict([s])[0];content={r['element']:r['predicted_mean_w_wt_percent'] for r in p['icp_predictions']}
        comp=bridge.composition(s,self.model.physics);total=sum(content.values());self.assertLess(total,100)
        for e,w in content.items():
            physical=prior(comp.component(e),self.model.physics)
            implied=w*physical['matrix_mass_g']/(physical['feed_metal_g']*(100-total))
            self.assertTrue(0<=implied<=1.00001)

    def test_poisoning_query_labels_does_not_change_predictions(self):
        s=copy.deepcopy(self.rows[51]);a=self.model.predict([s])
        s['icp'][0]['value']='99';s['xrd']['state']='FAIL' if s['xrd']['state']=='PASS' else 'PASS'
        self.assertEqual(a,self.model.predict([s]))
        s['origin']='DEMO'
        with self.assertRaises(ValueError):self.model.predict([s])

    def test_checkpoint_roundtrip_and_stale_data_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            item=train_candidate(folder,self.rows[:50],'REAL',epochs=4)
            loaded=load_model(folder,'REAL','SAC-Gen-Set',self.rows)
            self.assertEqual(item['model_id'],loaded.model_id)
            changed=copy.deepcopy(self.rows);changed[0]['icp'][0]['value']='.1'
            with self.assertRaises(ValueError):load_model(folder,'REAL','SAC-Gen-Set',changed)

    def test_calibrated_bundle_persistence_and_disjoint_pools(self):
        model=DecisionEnsemble().fit(self.rows,'REAL',epochs=3)
        pools=[set(model.split[k]) for k in ['fit_keys','weight_keys','calibration_keys']]
        self.assertTrue(all(not a&b for i,a in enumerate(pools) for b in pools[i+1:]))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model';model.save(path);loaded=DecisionEnsemble.load(path)
            a=model.predict([query(self.rows[0])])[0];b=loaded.predict([query(self.rows[0])])[0]
            self.assertAlmostEqual(a['predicted_total_wt_percent'],b['predicted_total_wt_percent'],places=6)

    def test_outer_masks_and_replay_do_not_reveal_test_groups(self):
        result=validate(self.rows,'REAL',folds=2,epochs=3,include_sac=False)
        tests=[]
        for fold in result['folds']:
            keys=[set(fold[k]) for k in ['fit_recipe_keys','weight_recipe_keys','calibration_recipe_keys','test_recipe_keys']]
            self.assertTrue(all(not a&b for i,a in enumerate(keys) for b in keys[i+1:]))
            tests+=fold['test_keys']
        self.assertEqual(len(tests),len(set(tests)));self.assertEqual(set(tests),{identity(s) for s in self.rows})
        result=replay(self.rows,'REAL',budget=16,initial=12,step=4,epochs=3)
        held=set(result['held_out_keys'])
        self.assertTrue(all(not held&set(r['chosen_keys']) and not held&set(r['revealed_keys']) for r in result['queries']))
        self.assertEqual({r['observations'] for r in result['curves']},{12,16})


if __name__=='__main__':unittest.main()
