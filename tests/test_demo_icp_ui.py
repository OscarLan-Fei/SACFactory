"""Exercise ICP validation using the actual bundled synthetic campaign."""
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DemoICPUITests(unittest.TestCase):
    def test_bundled_fe_demo_fits_and_displays_held_out_predictions(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            script = (
                'from sacfactory.bootstrap import load_demo_seed\n'
                'from sacfactory.ui_pulseicp import render_pulseicp\n'
                f'rows = load_demo_seed({str(root)!r})["samples"]\n'
                'rows = [sample for sample in rows if sample["elements"] == ["Fe"]]\n'
                f'render_pulseicp(rows, "DEMO", {directory!r})\n')
            ui = AppTest.from_string(script, default_timeout=90).run()
            self.assertFalse(ui.exception)
            self.assertFalse(ui.button(key='picp_run_DEMO').disabled)
            ui.number_input(key='picp_members_DEMO').set_value(1)
            ui.number_input(key='picp_gepoch_DEMO').set_value(2)
            ui.button(key='picp_run_DEMO').click().run()
            self.assertFalse(ui.exception)
            self.assertFalse(ui.error)
            _, points, _, info = ui.session_state['picp_result_DEMO'][1]
            self.assertEqual(info['coverage_fraction'], 1.0)
            self.assertEqual(points.sample_id.nunique(), 58)
            self.assertFalse(points.predicted_mean_w_wt_percent.isna().any())


if __name__ == '__main__':
    unittest.main()
