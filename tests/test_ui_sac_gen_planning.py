"""Full app SAC-Gen preview survives reruns and skips legacy fitting."""
import copy
from pathlib import Path
import unittest
import test_ui as fixture
from test_pulseicp_bridge import specimens
from test_sac_gen import budget
from sacfactory.storage import Store
from sacfactory.sac_gen import train_deployment
from sacfactory.compute_runtime import execution


class SACGenPlanningUITests(unittest.TestCase):
    setUp=fixture.UITests.setUp
    tearDown=fixture.UITests.tearDown

    def test_saved_sac_gen_drives_next_batch_without_preview_invalidation(self):
        store=Store(self.tmp.name,'REAL');store.clear_namespace();store.import_bundle(specimens(40),[])
        source=store.all('sample');context=copy.deepcopy(source[0]);context['pulse_limit_C']=1000.
        space={'verified':True,'concentration_bounds_mM':{'Fe':{'min_mM':70.,'max_mM':90.}},
               'pulses':[1],'max_total_mM':100.,'pulse_limit_min_C':1000.,'pulse_limit_max_C':1000.,
               'range_source':'Synthetic UI fixture range','temperature_step_C':10}
        store.put_many('meta',[{'id':'plan_context','origin':'REAL','context':context},
                               {'id':'space','origin':'REAL','space':space}],'id')
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            train_deployment(source,'REAL',Path(self.tmp.name),'Gaussian residual',budget())
        self.app.radio(key='page').set_value('Next batch').run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.checkbox(key='next_sac_gen_REAL').value)
        self.assertEqual(self.app.session_state['planning_state_REAL'][1].info['engine'],'sac_gen')
        next(b for b in self.app.button if b.label=='Generate batch preview').click().run(timeout=90)
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)
        rows=self.app.session_state['preview_REAL']
        self.assertTrue(rows)
        self.assertTrue(all(r.get('sac_gen_prediction') for r in rows))
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.session_state['preview_REAL']),len(rows))


if __name__=='__main__':unittest.main()
