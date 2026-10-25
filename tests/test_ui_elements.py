"""Direct periodic-table selection and synchronization across independent fields."""
import unittest

from streamlit.testing.v1 import AppTest


SCRIPT = '''
import streamlit as st
from sacfactory.domain import ELEMENTS
from sacfactory.ui_elements import element_multiselect, element_selectbox, element_text_area, periodic_selector
element_multiselect('Analysis elements', ELEMENTS, default=['Fe'], key='analysis_elements')
scope = st.radio('Available targets', ['All targets', 'Ni only'], key='scope')
element_selectbox('Single target', ['Fe', 'Co', 'Ni'] if scope == 'All targets' else ['Ni'], key='single_target')
element_text_area('Recipe elements', 'Fe;Mg', key='recipe_elements')
periodic_selector('primary_elements', ['Fe'])
'''


class PeriodicUITests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_string(SCRIPT, default_timeout=15).run()
        self.assert_clean()

    def assert_clean(self):
        self.assertEqual(len(self.app.exception), 0)

    def test_direct_multi_toggle_updates_immediately_and_preserves_other_fields(self):
        self.app.button(key='analysis_elements_3_Co').click().run()
        self.assert_clean()
        self.assertEqual(self.app.multiselect(key='analysis_elements').value, ['Fe', 'Co'])
        self.app.button(key='analysis_elements_3_Fe').click().run()
        self.assertEqual(self.app.multiselect(key='analysis_elements').value, ['Co'])
        self.assertEqual(self.app.selectbox(key='single_target').value, 'Fe')
        self.assertEqual(self.app.text_area(key='recipe_elements').value, 'Fe;Mg')
        self.assertEqual(self.app.session_state['primary_elements'], ['Fe'])
        self.assertFalse(any(b.label in {'Apply element selection', 'Cancel selection'} for b in self.app.button))
        self.assertFalse(any(b.key and b.key.endswith('_periodic_open') for b in self.app.button))
        self.assertFalse(self.app.button(key='analysis_elements_3_Co').proto.help)

    def test_all118_clear_and_primary_selection(self):
        self.app.button(key='analysis_elements_all').click().run()
        self.assertEqual(len(self.app.multiselect(key='analysis_elements').value), 118)
        self.app.button(key='analysis_elements_clear').click().run()
        self.assertEqual(self.app.multiselect(key='analysis_elements').value, [])
        self.app.button(key='primary_elements_6_Og').click().run()
        self.assertEqual(self.app.session_state['primary_elements'], ['Fe', 'Og'])
        self.assertFalse(self.app.button(key='primary_elements_6_Og').proto.help)
        self.assert_clean()

    def test_single_replaces_selection_and_respects_updated_scope(self):
        self.assertTrue(self.app.button(key='single_target_6_Og').disabled)
        self.app.button(key='single_target_3_Co').click().run()
        self.assertEqual(self.app.selectbox(key='single_target').value, 'Co')
        self.app.button(key='single_target_3_Ni').click().run()
        self.assertEqual(self.app.selectbox(key='single_target').value, 'Ni')
        self.app.selectbox(key='single_target').select('Fe').run()
        self.app.radio(key='scope').set_value('Ni only').run()
        self.assertEqual(self.app.selectbox(key='single_target').value, 'Ni')
        self.assertTrue(self.app.button(key='single_target_3_Fe').disabled)
        self.assert_clean()

    def test_text_entry_and_table_stay_synchronized(self):
        self.app.button(key='recipe_elements_clear').click().run()
        self.app.button(key='recipe_elements_3_Cu').click().run()
        self.app.button(key='recipe_elements_5_Au').click().run()
        self.assertEqual(self.app.text_area(key='recipe_elements').value, 'Cu;Au')
        self.app.text_area(key='recipe_elements').set_value('Fe;Co').run()
        self.app.button(key='recipe_elements_3_Co').click().run()
        self.assertEqual(self.app.text_area(key='recipe_elements').value, 'Fe')
        self.assertEqual(self.app.multiselect(key='analysis_elements').value, ['Fe'])
        self.assert_clean()

    def test_disabled_and_empty_fields(self):
        app = AppTest.from_string('''
from sacfactory.ui_elements import element_multiselect, element_selectbox
element_multiselect('Read only', ['Fe','Co'], default=['Fe'], disabled=True, key='locked')
element_selectbox('Empty field', [], key='empty')
''', default_timeout=15).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.button(key='locked_3_Co').disabled)
        self.assertTrue(app.button(key='locked_all').disabled)
        self.assertTrue(app.button(key='locked_clear').disabled)
        self.assertTrue(app.button(key='empty_3_Fe').disabled)
        self.assertIsNone(app.selectbox(key='empty').value)


if __name__ == '__main__':
    unittest.main()
