"""Automatic batch previews follow live deletions and respect manual overrides."""
import json
from pathlib import Path
import tempfile
import unittest

from streamlit.testing.v1 import AppTest
from sacfactory.domain import validate_sample
from sacfactory.storage import Store
from test_science import sample


class BatchNumberingUITests(unittest.TestCase):
    def test_blank_batch_field_saves_automatic_eight_then_advances_to_nine(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp,'REAL')
            records=[validate_sample({**sample(),'sample_id':f'SAC_{i:04d}',
                'batch_id':f'BATCH_{i:04d}'}) for i in range(1,8)]
            store.import_bundle(records,[])
            root=Path(__file__).resolve().parents[1]
            script="from pathlib import Path\nimport json\nfrom sacfactory.storage import Store\nfrom sacfactory.ui_batch import render_batch_panel\n"
            script+=f"store=Store({tmp!r},'REAL')\ncontext=json.loads({json.dumps(records[0])!r})\n"
            script+=f"render_batch_panel(store,context,{{}},Path({str(root)!r}))"
            ui=AppTest.from_string(script,default_timeout=45).run()
            ui.number_input(key='batch_n_REAL').set_value(10).run()
            self.assertEqual(ui.text_input(key='entry_batch_id_REAL').value,'BATCH_0008')
            ui.text_input(key='entry_batch_id_REAL').set_value('').run()
            self.assertEqual(ui.text_input(key='entry_batch_id_REAL').value,'BATCH_0008')
            ui.button(key='entry_preview_REAL').click().run()
            self.assertFalse(ui.exception)
            self.assertEqual({r['batch_id'] for r in ui.session_state['entry_preview_data_REAL']['rows']},{'BATCH_0008'})
            ui.button(key='entry_save_REAL').click().run()
            self.assertFalse(ui.exception)
            self.assertFalse(ui.error)
            self.assertEqual(sum(s['batch_id']=='BATCH_0008' for s in store.all('sample')),10)
            self.assertEqual(ui.text_input(key='entry_batch_id_REAL').value,'BATCH_0009')
            self.assertEqual(store.next_sample_ids()[0],'SAC_0018')
            ui.button(key='entry_preview_REAL').click().run()
            ui.button(key='entry_save_REAL').click().run()
            self.assertFalse(ui.exception)
            self.assertEqual(sum(s['batch_id']=='BATCH_0009' for s in store.all('sample')),10)
            self.assertEqual(ui.text_input(key='entry_batch_id_REAL').value,'BATCH_0010')

    def test_cached_automatic_batch_rewinds_after_deletion_and_keeps_manual_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp,'REAL')
            records=[validate_sample({**sample(),'sample_id':f'SAC_{i:04d}',
                'batch_id':f'BATCH_{i:04d}'}) for i in range(1,9)]
            store.import_bundle(records,[])
            root=Path(__file__).resolve().parents[1]
            script="from pathlib import Path\nimport json\nfrom sacfactory.storage import Store\nfrom sacfactory.ui_batch import render_batch_panel\n"
            script+=f"store=Store({tmp!r},'REAL')\ncontext=json.loads({json.dumps(records[0])!r})\n"
            script+=f"render_batch_panel(store,context,{{}},Path({str(root)!r}))"
            ui=AppTest.from_string(script,default_timeout=45).run()
            self.assertFalse(ui.exception)
            self.assertEqual(ui.text_input(key='entry_batch_id_REAL').value,'BATCH_0009')
            store.delete_samples(['SAC_0008'],realign=False)
            ui.run()
            self.assertFalse(ui.exception)
            self.assertEqual(ui.text_input(key='entry_batch_id_REAL').value,'BATCH_0008')
            ui.text_input(key='entry_batch_id_REAL').set_value('CUSTOM_RUN').run()
            store.delete_samples(['SAC_0007'],realign=False)
            ui.run()
            self.assertFalse(ui.exception)
            self.assertEqual(ui.text_input(key='entry_batch_id_REAL').value,'CUSTOM_RUN')
