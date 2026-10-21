"""Periodic-table drafts, commits, allowed scope and field isolation."""
import unittest

from streamlit.testing.v1 import AppTest


SCRIPT = '''
import streamlit as st
from sacfactory.domain import ELEMENTS
from sacfactory.ui_elements import element_multiselect, element_selectbox, element_text_area
element_multiselect('Analysis elements', ELEMENTS, default=['Fe'], key='analysis_elements')
element_selectbox('Single target', ['Fe', 'Co', 'Ni'], key='single_target')
element_text_area('Recipe elements', 'Fe;Mg', key='recipe_elements')
'''


class PeriodicUITests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_string(SCRIPT).run()

    def assert_clean(self):
        self.assertEqual(len(self.app.exception), 0)

    def test_multiselect_draft_apply_and_cancel_preserve_other_fields(self):
        self.app.button(key='analysis_elements_periodic_open').click().run()
        self.assert_clean()
        self.assertEqual(len([b for b in self.app.button if b.key and b.key.startswith('periodic_draft_analysis_elements_')]), 125)
        self.app.button(key='periodic_draft_analysis_elements_clear').click().run()
        self.app.button(key='periodic_draft_analysis_elements_3_Co').click().run()
        self.app.button(key='periodic_draft_analysis_elements_3_Ni').click().run()
        self.assertEqual(self.app.multiselect(key='analysis_elements').value, ['Fe'])
        self.app.button(key='periodic_draft_analysis_elements_apply').click().run()
        self.assert_clean()
        self.assertEqual(self.app.multiselect(key='analysis_elements').value, ['Co', 'Ni'])
        self.assertEqual(self.app.selectbox(key='single_target').value, 'Fe')
        self.assertEqual(self.app.text_area(key='recipe_elements').value, 'Fe;Mg')
        self.app.button(key='analysis_elements_periodic_open').click().run()
        self.app.button(key='periodic_draft_analysis_elements_all').click().run()
        self.app.button(key='periodic_draft_analysis_elements_cancel').click().run()
        self.assertEqual(self.app.multiselect(key='analysis_elements').value, ['Co', 'Ni'])
        self.assert_clean()

    def test_all118_single_scope_and_text_input(self):
        self.app.button(key='analysis_elements_periodic_open').click().run()
        self.app.button(key='periodic_draft_analysis_elements_all').click().run()
        self.app.button(key='periodic_draft_analysis_elements_apply').click().run()
        self.assertEqual(len(self.app.multiselect(key='analysis_elements').value), 118)
        self.app.button(key='single_target_periodic_open').click().run()
        self.assertTrue(self.app.button(key='periodic_draft_single_target_6_Og').disabled)
        self.app.button(key='periodic_draft_single_target_3_Co').click().run()
        self.app.button(key='periodic_draft_single_target_3_Ni').click().run()
        self.app.button(key='periodic_draft_single_target_apply').click().run()
        self.assertEqual(self.app.selectbox(key='single_target').value, 'Ni')
        self.app.button(key='recipe_elements_periodic_open').click().run()
        self.app.button(key='periodic_draft_recipe_elements_clear').click().run()
        self.app.button(key='periodic_draft_recipe_elements_3_Cu').click().run()
        self.app.button(key='periodic_draft_recipe_elements_5_Au').click().run()
        self.app.button(key='periodic_draft_recipe_elements_apply').click().run()
        self.assertEqual(self.app.text_area(key='recipe_elements').value, 'Cu;Au')
        self.assert_clean()


if __name__ == '__main__':
    unittest.main()
