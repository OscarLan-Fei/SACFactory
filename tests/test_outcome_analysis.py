"""Chemical associations retain original labels, missing concentrations and exclusions."""
import copy
import unittest
from unittest.mock import patch

import pandas as pd

from sacfactory.outcome_analysis import enriched_frame, outcome_groups, variable_groups
from sacfactory.descriptor_analysis import descriptor_report
from test_chemistry import campaign


class OutcomeAnalysisTests(unittest.TestCase):
    def test_all_element_concentrations_keep_absence_missing_and_presence_explicit(self):
        rows = campaign()
        original = copy.deepcopy(rows)
        frame = enriched_frame(rows, {}, 'REAL', ['Fe', 'Co', 'Au'])
        self.assertTrue(frame.loc[frame.composition == 'Fe', 'Co_mM'].isna().all())
        self.assertTrue(frame['Au_mM'].isna().all())
        self.assertEqual(frame['Au_present'].sum(), 0)
        self.assertEqual(frame.loc[frame.composition == 'Fe', 'Co_present'].sum(), 0)
        self.assertTrue(frame.total_concentration_mM.notna().all())
        self.assertTrue(frame.en_pauling_mean.notna().all())
        self.assertIn('Co_mM', variable_groups(frame, ['Fe', 'Co'])['Concentrations'])
        self.assertEqual(rows, original)

    def test_outcome_comparisons_exclude_pending_conflicts_and_retests(self):
        frame = pd.DataFrame({'label':[1.,0.,None,1.,None], 'conflict':[False,False,False,True,False],
            'status':['PASS','BURNED_OUT','PENDING','PASS','RETEST'], 'batch_id':['B1','B2','UNKNOWN','B4','B5'],
            'metric':[2.,None,7.,100.,10.]})
        resolved, summary = outcome_groups(frame, 'metric')
        self.assertEqual(len(resolved), 2)
        self.assertEqual(summary.resolved_records.sum(), 2)
        fail = summary.loc[summary.outcome_group == 'Screen FAIL'].iloc[0]
        self.assertEqual(fail.metric_observed, 0)
        self.assertTrue(pd.isna(fail['median']))
        self.assertEqual(fail.metric_missing, 1)
        branches, _ = outcome_groups(frame, 'metric', True)
        self.assertEqual(branches.outcome_group.tolist(), ['PASS', 'BURNED_OUT'])

    def test_descriptor_report_conflicting_outcomes_do_not_enter_associations(self):
        row = campaign()[0]
        with patch('sacfactory.descriptor_analysis.screen', return_value={'status':'PASS','label':1,'review_conflicts':['Conflict']}):
            frame = descriptor_report([row], {}, 'REAL')['recipes']
        self.assertIsNone(frame.label.iloc[0])
        self.assertTrue(frame.conflict.iloc[0])

    def test_only_one_class_and_unknown_batches_are_described_without_fake_difference(self):
        frame = pd.DataFrame({'label':[1.,1.], 'conflict':[False,False], 'status':['PASS','PASS'],
            'batch_id':['UNKNOWN','UNASSIGNED'], 'metric':[1.,3.]})
        _, summary = outcome_groups(frame,'metric')
        self.assertEqual(len(summary),1)
        self.assertEqual(summary.iloc[0]['median'],2.)
        self.assertEqual(summary.iloc[0].named_batches_with_metric,0)


if __name__ == '__main__':
    unittest.main()
