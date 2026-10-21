"""Mass-conserving migration, unit edits and protected fitting regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sacfactory import pulseicp_bridge as bridge
from sacfactory.domain import validate_sample
from sacfactory.planning import fingerprint
from sacfactory.preparation import to_mM, from_mM
from sacfactory.quantity_correction import reinterpret_recorded_values_as_mmol
from sacfactory.storage import Store
from sacfactory.ui_record_tables import condition_rows, record_element_rows, apply_record_tables


def specimens(count=60, origin='REAL'):
    from sacfactory._pulseicp.composition import decode_composition
    physics = bridge.configuration()['physics']
    rows = []
    for i in range(count):
        sample = {'sample_id':f'SAC_{i+1:04d}', 'origin':origin, 'batch_id':f'BATCH_{i//10+1:04d}',
            'elements':['Fe'], 'concentrations_mM':{'Fe':10.+(i%10)*5.}, 'pulse_count':1+i%3,
            'pulse_limit_C':900.+(i%4)*100., 'pan_mass_g':2., 'solvent_volume_mL':20.,
            'precursor_family':'METAL_SALTS_V1', 'support_batch':'PAN_DERIVED_CARBON', 'protocol_id':'PAN_DMF_PULSE_V1',
            'icp_review':{'state':'PASS' if i%3 else 'FAIL'}, 'xrd':{'state':'PASS' if i%3 else 'FAIL','mode':'MANUAL'},
            'stem':{'state':'NOT_TESTED'}, 'icp':[], 'notes':'Synthetic implementation test',
            'result_reported_at':f'2026-09-{i//10+1:02d}T12:00:00Z'}
        c = bridge.composition(sample, physics)
        w = float(decode_composition(c, {'Fe':-.3+.1*(i%4)}, physics)['Fe'])
        sample['icp'] = [{'element':'Fe', 'value':str(w), 'unit':'wt%', 'flag':'MEASURED', 'method_id':'', 'source':'Synthetic test'}]
        rows.append(validate_sample(sample))
    return rows


class PulseICPBridgeTests(unittest.TestCase):
    def test_legacy_parameter_aliases_share_the_migrated_physics_source(self):
        from sacfactory.thermal_loss import loss_settings
        from unittest.mock import patch
        config=bridge.configuration()['physics'];config['temperature_exponent']=3.1;config['metal_tau_pulses']=4.2
        try:
            loss_settings.cache_clear()
            with patch('sacfactory._pulseicp.configuration.physics_settings',return_value=config):
                settings=loss_settings()
                self.assertEqual(settings['temperature_exponent'],3.1)
                self.assertEqual(settings['reference_tau_pulses'],4.2)
                self.assertEqual(settings['thermal_weights'],config['thermal_weights'])
        finally:
            loss_settings.cache_clear()

    def test_roundtrip_quantities_and_mass_anchor_are_consistent(self):
        from sacfactory.physical_priors import mass_balance
        self.assertEqual(to_mM(1,'mmol',20),50)
        self.assertEqual(from_mM(50,'mmol',20),1)
        for volume in (5.,20.,40.):
            for amount in (.05,1.,2.4,50.):
                self.assertAlmostEqual(from_mM(to_mM(amount,'mmol',volume),'mmol',volume),amount)
        a=mass_balance({'Fe':50.},1000,1)
        b=mass_balance({'Fe':50.},1000,1,pan_mass_g=4.)
        self.assertAlmostEqual(a['elements']['Fe']['true_w_wt_percent'],2.5)
        self.assertAlmostEqual(a['elements']['Fe']['retention_fraction'],b['elements']['Fe']['retention_fraction'])
        self.assertLess(b['elements']['Fe']['true_w_wt_percent'],a['elements']['Fe']['true_w_wt_percent'])
        from sacfactory._pulseicp.composition import decode_composition
        sample=specimens(1)[0];sample.update(concentrations_mM={'Fe':50.},pulse_limit_C=1000.,pulse_count=1)
        self.assertAlmostEqual(float(decode_composition(bridge.composition(sample),{'Fe':0.},bridge.configuration()['physics'])['Fe']),2.5)

    def test_record_table_quantity_changes_are_bidirectional_and_conflicts_rejected(self):
        original=specimens(1)[0]
        baseline=record_element_rows([original]); conditions=condition_rows([original])
        cells=copy.deepcopy(baseline);cells[0]['precursor_mmol']=1.
        changed=apply_record_tables([original],conditions,cells,baseline)[0]
        self.assertEqual(changed['concentrations_mM']['Fe'],50.)
        cells=copy.deepcopy(baseline);cells[0]['precursor_mM']=25.
        changed=validate_sample(apply_record_tables([original],conditions,cells,baseline)[0])
        self.assertEqual(changed['precursor_amounts_mmol']['Fe'],.5)
        cells[0]['precursor_mmol']=5.
        with self.assertRaisesRegex(ValueError,'disagree'):
            apply_record_tables([original],conditions,cells,baseline)
        conditions[0]['solvent_volume_mL']=40.
        changed=validate_sample(apply_record_tables([original],conditions,baseline,baseline)[0])
        self.assertEqual(changed['concentrations_mM']['Fe'],original['concentrations_mM']['Fe'])
        self.assertAlmostEqual(changed['precursor_amounts_mmol']['Fe'],original['concentrations_mM']['Fe']*.04)

    def test_explicit_historical_correction_is_atomic_idempotent_and_namespace_safe(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(root,'REAL'); records=specimens(2)
            store.put_many('sample',records,'sample_id')
            records=store.all('sample'); before=copy.deepcopy(records)
            expected={s['sample_id']:fingerprint(s) for s in records}
            result=reinterpret_recorded_values_as_mmol(store,expected,'test-correction')
            self.assertEqual(result['corrected_samples'],2)
            for old,new in zip(before,store.all('sample')):
                self.assertEqual(old['icp'],new['icp'])
                self.assertEqual(new['precursor_amounts_mmol'],old['concentrations_mM'])
            self.assertTrue(reinterpret_recorded_values_as_mmol(store,expected,'test-correction')['already_applied'])
            with self.assertRaises(ValueError):
                reinterpret_recorded_values_as_mmol(Store(root,'DEMO'),expected,'test')
            with self.assertRaises(ValueError):
                reinterpret_recorded_values_as_mmol(store,expected,'stale-review')
            self.assertEqual(len(store.all('sample')),2)

    def test_bound_incompatible_targets_are_audited_without_mutation(self):
        rows=specimens(3);rows[0]['icp'][0]['value']='99'
        before=copy.deepcopy(rows);data=bridge.prepare(rows,'REAL')
        self.assertEqual(len(data.records),2)
        self.assertFalse(bool(data.audit.iloc[0].eligible))
        self.assertIn('feed',data.audit.iloc[0].exclusion_reason)
        self.assertEqual(rows,before)
        self.assertEqual(len(bridge.prepare(rows,'DEMO').records),0)

    def test_schema_uses_shared_fifty_component_inputs_and_unknown_mask(self):
        from sacfactory.composition import transfer_feature
        from sacfactory.chemical_descriptors import model_features
        row=specimens(1)[0]
        x=transfer_feature(row,'chemistry_pulseicp')
        self.assertTrue(np.isfinite(model_features([x],'chemistry_pulseicp')).all())
        self.assertEqual(bridge.recipe_features(row)[-1],1.)
        unknown={**row,'pan_mass_g':None}
        self.assertEqual(bridge.recipe_features(unknown),[0.]*51)
        after={**row,'icp':[{'element':'Fe','value':'99'}],'xrd':{'state':'FAIL'}}
        self.assertEqual(bridge.recipe_features(row),bridge.recipe_features(after))

    def test_two_ensembles_use_protected_batches_safe_checkpoints_and_fixed_scaler(self):
        from sacfactory._pulseicp.ensemble import ResidualEnsemble
        from sacfactory._pulseicp.features import TrainingScaler
        scaler=TrainingScaler().fit(np.array([[1.,1.],[2.,1.]],dtype=np.float32))
        self.assertEqual(float(scaler.transform(np.array([[1.,9.]],dtype=np.float32))[0,1]),0.)
        rows=specimens();config=bridge.configuration();settings=config['model']
        settings.update(members=1,epochs_baseline=2,epochs_diffusion=2,draws_per_member=4,diffusion_steps=10)
        before=copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as root:
            destination,points,comparison,info=bridge.validate(rows,'REAL',settings,config['physics'],Path(root)/'pulseicp/real')
            self.assertEqual(set(comparison.model),{'Unfitted physical prior',*bridge.FAMILIES})
            roles={r['role']:set(r['batch_ids']) for r in info['partitions']}
            self.assertFalse(roles['train']&roles['test'])
            self.assertFalse(roles['calibration']&roles['test'])
            self.assertEqual(info['origin'],'EXPERIMENTAL')
            model=ResidualEnsemble.load(destination/'gaussian')
            forecast=bridge.forecast([rows[0]],'REAL',root,current_samples=rows)
            self.assertEqual(len(forecast),1)
            altered=copy.deepcopy(rows)
            trained=set(model.training_ids)
            next(s for s in altered if s['sample_id'] in trained)['concentrations_mM']['Fe'] *= 2
            with self.assertRaisesRegex(ValueError,'training records changed'):
                bridge.forecast([rows[0]],'REAL',root,current_samples=altered)
            with self.assertRaisesRegex(ValueError,'context'):
                bridge.forecast([{**rows[0],'support_batch':'OTHER_SUPPORT'}],'REAL',root,current_samples=rows)
            self.assertEqual(model.calibration['status'],'INSUFFICIENT_CALIBRATION_BATCHES')
            test_ids=next(r['sample_ids'] for r in info['partitions'] if r['role']=='test')
            test=copy.deepcopy([s for s in rows if s['sample_id'] in test_ids])
            first=bridge.predict(model,test,'REAL')
            for s in test:s['icp'][0]['value']='98'
            second=bridge.predict(model,test,'REAL')
            np.testing.assert_array_equal(first.predicted_mean_w_wt_percent,second.predicted_mean_w_wt_percent)
            with self.assertRaises(ValueError):bridge.predict(model,test,'DEMO')
            self.assertTrue(np.isfinite(points.predicted_mean_w_wt_percent).all())
        # The held-out outcome perturbation was deliberate; original training input was untouched.
        self.assertEqual(rows[:30],before[:30])

    def test_mixture_decoder_closes_mass_and_forecast_never_claims_fitted_probability(self):
        from sacfactory._pulseicp.composition import decode_composition
        row=specimens(1)[0];row.update(elements=['Fe','Co'],concentrations_mM={'Fe':50.,'Co':40.})
        c=bridge.composition(row);physics=bridge.configuration()['physics']
        a=decode_composition(c,{'Fe':0.,'Co':0.},physics)
        b=decode_composition(c,{'Fe':0.,'Co':3.},physics)
        self.assertLess(float(b['Fe']),float(a['Fe']))
        self.assertLess(sum(float(v) for v in b.values()),100.)
        with tempfile.TemporaryDirectory() as root:
            frame=bridge.forecast([row],'REAL',root)
            self.assertEqual(set(frame.model),{'Unfitted physical prior'})
            self.assertNotIn('p_screen_pass',frame)

    def test_interaction_denominators_units_and_annotations_are_explicit(self):
        from sacfactory.analysis import experiment_frame
        from sacfactory.analysis_extended import interaction_rates
        from sacfactory.ui_charts import interaction_chart
        rows=specimens(4);rows[1]['solvent_volume_mL']=40.
        frame=experiment_frame(rows,{s['sample_id']:{'status':'PASS'} for s in rows})
        self.assertEqual(frame.iloc[0].Fe_mmol,rows[0]['concentrations_mM']['Fe']*.02)
        self.assertEqual(frame.iloc[1].Fe_mmol,rows[1]['concentrations_mM']['Fe']*.04)
        rates=interaction_rates(frame,'Fe_mM','pulse_limit_C')
        self.assertEqual(int(rates.n.sum()),4)
        self.assertTrue((rates.failed==0).all())
        self.assertTrue((rates.lower_95<1).all())
        chart=interaction_chart(rates,'Fe / mM','Cutoff / C').to_dict()
        self.assertEqual(len(chart['layer']),2)

    def test_weighted_validation_neural_head_stays_inside_each_training_fold(self):
        from sacfactory.weighted_validation import weighted_validation
        rows=specimens();settings=bridge.configuration()['model']
        settings.update(members=1,epochs_baseline=2,epochs_diffusion=2,draws_per_member=4,diffusion_steps=10)
        points,_,_,_,info=weighted_validation(rows,{},'REAL',['Fe'],protocol='Batch holdout',
            profile='chemistry_pulseicp',max_folds=2,trees=5,icp_family='Gaussian residual',icp_settings=settings)
        self.assertTrue(points.predicted_w_wt_percent.notna().any())
        for fold in info['folds']:
            self.assertFalse(set(fold['train_sample_ids']) & set(fold['test_sample_ids']))
            self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))
            self.assertTrue(fold['pulseicp'].get('model_id'))
        json.dumps(info,allow_nan=False)

    def test_capture_competition_is_explicitly_unvalidated_and_conserves_atom_budget(self):
        from sacfactory._pulseicp.screening_prior import predict_xrd_prior
        config=bridge.configuration();config['screening_prior']['draws']=32
        row=predict_xrd_prior([bridge.composition(specimens(1)[0])],config['physics'],config['screening_prior']).iloc[0]
        self.assertIn('UNVALIDATED',row.xrd_prior_status)
        self.assertLess(float(row.xrd_competition_maximum_atom_budget_error),1e-8)
        self.assertFalse(bool(row.xrd_fixed_temperature_penalty_applied))


if __name__=='__main__':
    unittest.main()
