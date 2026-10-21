"""Whole-batch masking, independent outcome heads and complete-score accounting."""
import copy
import json
import unittest

import numpy as np
import pandas as pd

from sacfactory.domain import screen
from sacfactory.outcome_validation import outcome_labels, normalized_weights, score_samples
from sacfactory.weighted_validation import ROTATING_CV, weighted_validation
from test_element_campaign import singles


def outcomes():
    rows=singles()
    for i,s in enumerate(rows):
        s['sample_condition']='BURNED_OUT' if i%4==3 else 'RECOVERED'
        s['icp_review']['state']='PASS' if i%4<2 else 'FAIL'
        s['stem']={'state':'CONFIRMED' if i%4<2 else 'NOT_CONFIRMED',
            'source':'Synthetic evidence', 'reviewer':'Demo reviewer'}
    return rows


class WeightedOutcomeTests(unittest.TestCase):
    def test_rotating_one_or_multiple_batches_cover_every_sample_once(self):
        rows=outcomes(); before=copy.deepcopy(rows)
        for group in (1,2,4):
            points,scores,_,_,info=weighted_validation(rows,{},'DEMO',['Fe','Co'],
                protocol=ROTATING_CV,batches_per_fold=group,trees=8,max_folds=2)
            self.assertEqual(len(points),48)
            self.assertEqual(len(scores),48)
            self.assertFalse(points.duplicated(['sample_id','element']).any())
            self.assertEqual(info['eligible_selected_samples'],info['predicted_selected_samples'])
            for fold in info['folds']:
                self.assertFalse(set(fold['train_batch_ids']) & set(fold['test_batch_ids']))
                self.assertLessEqual(len(fold['test_batch_ids']),group)
            self.assertEqual({m['evaluated_samples'] for m in info['outcome_metrics']},{48})
            self.assertEqual(len(info['folds']),(12+group-1)//group)
            json.dumps(info,allow_nan=False)
        self.assertEqual(rows,before)

    def test_heldout_results_cannot_change_any_head_or_baseline(self):
        rows=outcomes()
        first=weighted_validation(rows,{},'DEMO',['Fe','Co'],protocol=ROTATING_CV,
            batches_per_fold=2,trees=8)
        fold=first[4]['folds'][0]
        changed=copy.deepcopy(rows)
        for s in changed:
            if s['sample_id'] in fold['test_sample_ids']:
                s['icp_review']['state']='FAIL'
                s['icp'][0]['value']='33'
                s['xrd']['state']='FAIL'
                s['stem']['state']='NOT_CONFIRMED'
                s['sample_condition']='RECOVERED'
        second=weighted_validation(changed,{},'DEMO',['Fe','Co'],protocol=ROTATING_CV,
            batches_per_fold=2,trees=8)
        columns=['predicted_w_wt_percent','p_screen_pass']+[f'{k}_predicted' for k in
            ('icp_pass','xrd_pass','stem_confirmed','burnout')]
        a=first[0].loc[first[0].fold==fold['fold'],columns]
        b=second[0].loc[second[0].fold==fold['fold'],columns]
        np.testing.assert_allclose(a,b)
        self.assertEqual(first[4]['folds'][0]['outcome_heads'],second[4]['folds'][0]['outcome_heads'])

    def test_unknown_stem_and_recovery_do_not_become_negative_labels(self):
        s=singles()[0]
        s.pop('sample_condition',None)
        s['stem']={'state':'CONFIRMED'}
        label=outcome_labels(s,screen(s,{}))
        self.assertIsNone(label['stem_confirmed'])
        self.assertIsNone(label['burnout'])
        s['xrd']['state']='FAIL';s['icp_review']['state']='FAIL'
        self.assertIsNone(outcome_labels(s,screen(s,{}))['burnout'])

    def test_binary_only_score_works_and_missing_required_endpoint_blocks_score(self):
        rows=singles()
        points,scores,_,_,info=weighted_validation(rows,{},'DEMO',['Fe','Co'],protocol=ROTATING_CV,
            trees=8,endpoint_weights={'xrd_pass':1})
        self.assertEqual(info['n_scored_samples'],48)
        self.assertAlmostEqual(scores.validation_loss.mean(),points.xrd_pass_loss.mean())
        _,scores,_,_,info=weighted_validation(rows,{},'DEMO',['Fe','Co'],protocol=ROTATING_CV,
            trees=8,endpoint_weights={'xrd_pass':1,'stem_confirmed':1})
        self.assertEqual(info['n_scored_samples'],0)
        self.assertTrue(scores.missing_score_endpoints.str.contains('stem_confirmed').all())
        self.assertTrue(scores.validation_score.isna().all())

    def test_mixtures_require_all_selected_icp_values_and_binary_counts_once(self):
        data=[]
        for element,icp in [('Fe',1.),('Co',3.)]:
            row={'fold':'F','sample_id':'S','batch_id':'B','composition':'Fe+Co','element_count':2,
                'element':element,'icp_loss':icp,'icp_baseline_loss':4.}
            for k in ('screening','icp_pass','xrd_pass','stem_confirmed','burnout'):
                row.update({f'{k}_loss':.25,f'{k}_baseline_loss':.5})
            data.append(row)
        frame=pd.DataFrame(data)
        result=score_samples(frame,{'icp':.6,'xrd_pass':.4})
        self.assertAlmostEqual(result.iloc[0].validation_loss,1.3)
        frame.loc[1,'icp_loss']=np.nan
        self.assertTrue(pd.isna(score_samples(frame,{'icp':.6,'xrd_pass':.4}).iloc[0].validation_score))

    def test_invalid_weights_and_insufficient_context_coverage_are_explicit(self):
        for weights in ({},{'xrd_pass':-1},{'xrd_pass':float('nan')},{'invented':1}):
            with self.assertRaises(ValueError):
                normalized_weights(weights,.6)
        rows=singles()[:8]
        points,_,_,_,info=weighted_validation(rows,{},'DEMO',['Fe'],protocol=ROTATING_CV,trees=8)
        self.assertTrue(points.empty)
        self.assertEqual(len(info['uncovered_samples']),8)
