"""Lazy analysis execution preserves scientific settings and all selected rows."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from sacfactory.analysis import correlations
from sacfactory.ui_analysis import cached_analysis_frame
from test_element_campaign import singles


class CorrelationRegressionTests(unittest.TestCase):
    def test_native_spearman_matches_pairwise_ranking_with_missing_values_and_ties(self):
        random = np.random.default_rng(123)
        frame = pd.DataFrame(random.integers(0, 7, size=(35, 9)), dtype=float)
        frame.columns = [f'variable_{i}' for i in range(9)]
        frame.iloc[::3, 0] = np.nan
        frame.iloc[::5, 1] = np.nan
        frame['constant'] = 2.
        frame['missing'] = np.nan
        original = frame.copy(deep=True)
        result = correlations(frame, list(frame.columns))
        for row in result.itertuples():
            paired = frame[[row.parameter_1, row.parameter_2]].dropna()
            self.assertEqual(row.paired_n, len(paired))
            if len(paired) >= 3 and all(paired[c].nunique() > 1 for c in paired):
                expected = paired.iloc[:, 0].rank().corr(paired.iloc[:, 1].rank())
                self.assertAlmostEqual(row.spearman_rho, expected, places=12)
            else:
                self.assertTrue(pd.isna(row.spearman_rho))
        pd.testing.assert_frame_equal(frame, original)

    def test_small_selections_and_pair_threshold(self):
        frame = pd.DataFrame({'a': [1., 2., 3.], 'b': [3., 2., 1.]})
        self.assertTrue(correlations(frame, []).empty)
        self.assertTrue(correlations(frame, ['a']).empty)
        self.assertTrue(correlations(frame, ['a', 'b'], minimum_pairs=4).spearman_rho.isna().all())

    def test_fast_matrix_serialization_retains_the_validated_spec_and_all_cells(self):
        from sacfactory.ui_charts import matrix_chart
        names=['Fe', 'Co', 'Au']
        cells=pd.DataFrame([{'x':left, 'y':right, 'value':None if left==right else .5}
                            for left in names for right in names])
        chart=matrix_chart(cells,'x','y','value',names,names,'Association')
        self.assertEqual(chart.to_dict(validate=True),chart.to_dict(validate=False))
        self.assertEqual(len(chart.to_dict(validate=False)['data']['values']),9)


class AnalysisNavigationTests(unittest.TestCase):
    def make_app(self):
        app = AppTest.from_string('''
import streamlit as st
from sacfactory.ui_analysis import render_analysis
render_analysis(st.session_state['rows'], {}, 'DEMO')
''', default_timeout=30)
        app.session_state['rows'] = singles()
        app.session_state['analysis_scope_DEMO'] = 'Cross-cohort exploratory associations'
        return app

    def test_hidden_analysis_tabs_never_execute(self):
        app = self.make_app()
        with patch('sacfactory.ui_loading.render_loading') as loading, \
             patch('sacfactory.ui_benchmarks.render_benchmarks') as benchmarks, \
             patch('sacfactory.ui_weighted_validation.render_weighted_validation') as weighted, \
             patch('sacfactory.ui_analysis_extended.repeatability') as repeats:
            app.run()
            self.assertEqual(len(app.exception), 0)
            loading.assert_not_called()
            benchmarks.assert_not_called()
            weighted.assert_not_called()
            repeats.assert_not_called()
            app.session_state['analysis_views_DEMO'] = 'ICP loading prediction'
            app.run()
            loading.assert_called_once()
            benchmarks.assert_not_called()
            weighted.assert_not_called()

    def test_selected_variables_survive_tab_navigation(self):
        app = self.make_app(); app.run()
        variable = next(w for w in app.multiselect if w.label == 'Correlation variables')
        key = variable.key
        variable.set_value(['Fe_mM', 'label']).run()
        app.session_state['analysis_views_DEMO'] = 'Batch overview'; app.run()
        app.session_state['analysis_views_DEMO'] = 'Correlations & coverage'; app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.multiselect(key=key).value, ['Fe_mM', 'label'])

    def test_cached_frame_invalidates_on_record_edit_and_keeps_namespaces_separate(self):
        rows = singles()
        before = copy.deepcopy(rows)
        cached_analysis_frame.clear()
        with patch('sacfactory.ui_analysis.enriched_frame', return_value=pd.DataFrame({'value': [1.]})) as build:
            cached_analysis_frame(rows, {}, 'DEMO')
            cached_analysis_frame(rows, {}, 'DEMO')
            self.assertEqual(build.call_count, 1)
            edited = copy.deepcopy(rows)
            element = edited[0]['elements'][0]
            edited[0]['concentrations_mM'][element] *= 2
            cached_analysis_frame(edited, {}, 'DEMO')
            cached_analysis_frame(rows, {}, 'REAL')
            self.assertEqual(build.call_count, 3)
        self.assertEqual(rows, before)
        cached_analysis_frame.clear()

    def test_repeatability_reuses_summary_but_invalidates_after_data_changes(self):
        from sacfactory.ui_analysis_extended import cached_repeatability
        cached_repeatability.clear()
        frame = pd.DataFrame({'sample_id': ['one']})
        with patch('sacfactory.ui_analysis_extended.repeatability', return_value=frame) as build:
            cached_repeatability(frame)
            cached_repeatability(frame)
            self.assertEqual(build.call_count, 1)
            cached_repeatability(pd.DataFrame({'sample_id': ['changed']}))
            self.assertEqual(build.call_count, 2)
        cached_repeatability.clear()

    def test_all_elements_and_all_records_remain_selectable(self):
        app = self.make_app(); before = copy.deepcopy(app.session_state['rows']); app.run()
        chooser = next(w for w in app.multiselect if w.label == 'Concentration elements to include')
        self.assertEqual(len(chooser.options), 118)
        chooser.set_value(['Fe', 'Co', 'Au']).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(next(w.value for w in app.metric if w.label == 'Selected records'), str(len(before)))
        self.assertEqual(app.session_state['rows'], before)


class OtherWorkspaceNavigationTests(unittest.TestCase):
    def test_sample_and_material_tabs_render_only_the_selected_view(self):
        from sacfactory.storage import Store
        from test_science import sample
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'SACFACTORY_DATA_DIR': temporary}):
            store = Store(temporary, 'REAL')
            store.import_bundle([sample()], [])
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'), default_timeout=30)
            with patch('sacfactory.ui_records.render_records') as records:
                app.run()
                app.radio(key='page').set_value('Samples & data').run()
                self.assertEqual(len(app.exception), 0)
                records.assert_not_called()
                self.assertFalse(any(w.label=='Pending sample to edit' for w in app.selectbox))
                app.session_state['sample_views_REAL']='Experiment records';app.run()
                self.assertEqual(len(app.exception), 0)
                records.assert_called_once()
            app.radio(key='page').set_value('Materials & consumables').run()
            for view in ['Inventory', 'Chemical references', 'Precursors & fixed system']:
                app.session_state['material_views_REAL']=view;app.run()
                self.assertEqual(len(app.exception), 0)

    def test_audit_download_labels_match_the_exported_snapshot(self):
        from sacfactory.ui_records import audit_export_bytes
        from sacfactory.domain import RULE_VERSION
        from test_science import sample
        row=sample()
        class ReadOnlyExport:
            def export(self):
                return {'samples':[row], 'methods':[], 'origin':'REAL', 'audit':[]}
        payload=json.loads(audit_export_bytes(ReadOnlyExport()))
        self.assertEqual(payload['rule_version'],RULE_VERSION)
        self.assertEqual(list(payload['current_labels']),[row['sample_id']])
        self.assertEqual(payload['samples'],[row])


if __name__ == '__main__':
    unittest.main()
