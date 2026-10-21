"""Automatic single-context scope and selectable multiple process contexts."""
import ast
import copy
import json
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest
from test_pulseicp_bridge import specimens


class PublicContextTests(unittest.TestCase):
    def render(self, rows):
        source='import json, tempfile\nfrom sacfactory.ui_pulseicp import render_pulseicp\n'
        source+=f'rows=json.loads({json.dumps(rows)!r})\n'
        source+="render_pulseicp(rows,'REAL',tempfile.gettempdir())"
        return AppTest.from_string(source,default_timeout=30).run()

    def test_single_context_is_automatic_without_a_selector(self):
        ui=self.render(specimens())
        self.assertFalse(ui.exception)
        self.assertFalse(any(w.key=='picp_contexts_REAL' for w in ui.multiselect))
        self.assertFalse(any(w.key=='picp_contexts_REAL_all' for w in ui.button))
        self.assertTrue(any('60 eligible specimens' in w.value for w in ui.caption))
        self.assertEqual(len(ui.get('json')),0)

    def test_multiple_contexts_remain_selectable_and_filter_the_inputs(self):
        rows=specimens()
        for row in rows[30:]:row['protocol_id']='SECOND_PROTOCOL'
        before=copy.deepcopy(rows)
        ui=self.render(rows)
        self.assertFalse(ui.exception)
        self.assertEqual(ui.multiselect(key='picp_contexts_REAL').value,[0,1])
        ui.multiselect(key='picp_contexts_REAL').set_value([1]).run()
        self.assertFalse(ui.exception)
        self.assertTrue(any('30 eligible specimens' in w.value for w in ui.caption))
        self.assertEqual(rows,before)

    def test_public_views_do_not_render_code_or_raw_json(self):
        root=Path(__file__).resolve().parents[1]
        for path in [root/'app.py',*(root/'sacfactory').glob('ui*.py')]:
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertFalse(isinstance(node.func.value,ast.Name) and
                        node.func.value.id=='st' and node.func.attr in {'json','code'},str(path))


if __name__=='__main__':unittest.main()
