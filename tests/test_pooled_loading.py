"""Shared regression really pools elements while holding out whole specimens/batches."""
import copy
import json
import unittest

import numpy as np

from sacfactory.pooled_loading import collect,pooled_loading_validation
from sacfactory.loading import MODELS
from test_element_campaign import singles


class PooledLoadingTests(unittest.TestCase):
    def test_all_benchmark_families_share_elements_and_keep_batch_masks(self):
        rows=singles();before=copy.deepcopy(rows)
        for family in MODELS:
            points,audit,info=pooled_loading_validation(rows,'DEMO',['Fe','Co'],family=family)
            self.assertEqual(len(points),48)
            self.assertEqual(set(points.element),{'Fe','Co'})
            self.assertTrue(points.predicted_w_wt_percent.between(0,100).all())
            self.assertEqual(info['eligible_specimens'],48)
            self.assertFalse(points.duplicated(['sample_id','element']).any())
            for fold in info['folds']:
                self.assertFalse(set(fold['train_batch_ids'])&set(fold['test_batch_ids']))
                self.assertEqual(set(fold['train_elements']),{'Fe','Co'})
            json.dumps(info,allow_nan=False)
        self.assertEqual(rows,before)

    def test_withheld_values_do_not_change_predictions_or_active_scaling(self):
        rows=singles()
        first=pooled_loading_validation(rows,'DEMO',['Fe','Co'],family='Random forest')
        fold=first[2]['folds'][0]
        changed=copy.deepcopy(rows)
        for s in changed:
            if s['sample_id'] in fold['test_sample_ids']:s['icp'][0]['value']='40'
        second=pooled_loading_validation(changed,'DEMO',['Fe','Co'],family='Random forest')
        a=first[0].loc[first[0].fold==fold['fold']]
        b=second[0].loc[second[0].fold==fold['fold']]
        np.testing.assert_array_equal(a.predicted_w_wt_percent,b.predicted_w_wt_percent)
        self.assertFalse(np.array_equal(a.observed_w_wt_percent,b.observed_w_wt_percent))

    def test_mixture_target_identity_is_explicit_and_members_share_test_fold(self):
        rows=singles()[:24]
        for s in rows:
            s['elements']=['Fe','Co'];s['concentrations_mM']['Co']=.05
            s['icp'].append({**s['icp'][0],'element':'Co','value':'.5'})
        records,_=collect(rows,'DEMO',['Fe','Co'],'chemistry_ion')
        fe=next(r for r in records if r['element']=='Fe')
        co=next(r for r in records if r['element']=='Co')
        self.assertFalse(np.array_equal(fe['x'],co['x']))
        points,_,info=pooled_loading_validation(rows,'DEMO',['Fe','Co'],family='Ridge baseline')
        self.assertEqual(len(points),48)
        self.assertTrue((points.groupby('sample_id').fold.nunique()==1).all())
        self.assertEqual(info['metrics']['physical_specimens'],24)

    def test_failures_with_numeric_icp_are_included_but_unknown_and_burnout_are_not_zero(self):
        rows=singles()
        rows[0]['icp'][0].update(value='ND',flag='ND')
        rows[1]['sample_condition']='BURNED_OUT'
        rows[2]['icp'][0]['value']=''
        points,audit,info=pooled_loading_validation(rows,'DEMO',['Fe','Co'],family='Random forest')
        self.assertEqual(len(points),45)
        self.assertEqual(int((~audit.eligible).sum()),3)
        self.assertGreater(sum(s['xrd']['state']=='FAIL' and s['sample_id'] in set(points.sample_id) for s in rows),0)

    def test_contexts_remain_separate_and_physical_models_use_identical_target_pairs(self):
        rows=singles()
        for s in rows[24:]:s['support_batch']='OTHER'
        points,_,info=pooled_loading_validation(rows,'DEMO',['Fe','Co'],family='Random forest')
        self.assertEqual(len(points),48)
        self.assertTrue(all(len(f['train_elements'])==1 for f in info['folds']))
        physics={'confirmed':True,'source':'Synthetic common preparation','pan_mass_g':2.,
            'solvent_volume_mL':20.,'carbonization_loss':.06}
        paired,_,info=pooled_loading_validation(rows,'DEMO',['Fe','Co'],family='Random forest',physics=physics)
        self.assertEqual(len(paired),48)
        self.assertEqual(len(info['paired_comparison']),3)
        self.assertTrue(paired.physical_prior_w_wt_percent.notna().all())

    def test_missing_targets_and_small_folds_return_auditable_empty_reports(self):
        points,audit,info=pooled_loading_validation(singles(),'DEMO',['Ni'])
        self.assertTrue(points.empty)
        self.assertTrue(audit.empty)
        self.assertEqual(info['per_element'][0]['evaluated_targets'],0)
        points,audit,info=pooled_loading_validation(singles()[:5],'DEMO',['Fe'])
        self.assertTrue(points.empty)
        self.assertEqual(info['eligible_targets'],5)
        self.assertFalse(audit.evaluated.any())
        json.dumps(info,allow_nan=False)

    def test_sum_above_one_hundred_is_excluded_without_modifying_assays(self):
        rows=singles()[:24]
        for s in rows:
            s['elements']=['Fe','Co'];s['concentrations_mM']['Co']=.05
            s['icp'][0]['value']='60'
            s['icp'].append({**s['icp'][0],'element':'Co'})
        before=copy.deepcopy(rows)
        points,audit,info=pooled_loading_validation(rows,'DEMO',['Fe','Co'])
        self.assertTrue(points.empty)
        self.assertEqual(info['eligible_targets'],0)
        self.assertEqual(len(audit),48)
        self.assertTrue(audit.exclusion_reason.str.contains('above 100').all())
        self.assertEqual(rows,before)
