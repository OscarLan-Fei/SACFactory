"""Information ablations remain independent of omitted scientific/assay inputs."""
import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
from test_pulseicp_bridge import specimens
from sacfactory.compute_runtime import execution
from sacfactory.weighted_validation import collect
from sacfactory.data_only import DataOnlyModel,features
from sacfactory.science_only import SciencePriorModel


class InformationControlTests(unittest.TestCase):
    def test_data_only_features_ignore_science_and_results(self):
        a=specimens(1)[0];b=copy.deepcopy(a)
        b.update(icp=[],xrd={'state':'FAIL'},precursors={'Fe':{'compound':'UNRELATED','oxidation_state':8}},
            carbonization_loss_fraction=.9,notes='Changed outside the permitted input contract')
        np.testing.assert_array_equal(features(a,'Fe'),features(b,'Fe'))

    def test_math_backends_mask_results_and_roundtrip(self):
        rows=specimens(40);records,_=collect(rows,{},'REAL','periodic',use_thermal=False)
        for backend in ('Random forest','Gradient boosting','Gaussian process'):
            with execution({'device':'cpu','threads':1,'cpu_jobs':1}),tempfile.TemporaryDirectory() as folder:
                model=DataOnlyModel(backend).fit_records(records,'REAL')
                before=model.predict(rows[:2]);changed=copy.deepcopy(rows[:2])
                for r in changed:r.update(icp=[],xrd={'state':'PENDING'})
                after=model.predict(changed)
                np.testing.assert_allclose([p['p_xrd_pass'] for p in before],[p['p_xrd_pass'] for p in after])
                path=Path(folder)/'model';model.save(path);loaded=DataOnlyModel.load(path)
                self.assertEqual(model.model_id,loaded.model_id)
                self.assertEqual(model.info['ICP_training_pairs'],40)

    def test_fixed_science_prior_has_no_fitted_training_ids(self):
        rows=specimens(2);model=SciencePriorModel('REAL')
        first=model.predict(rows);changed=copy.deepcopy(rows)
        for r in changed:r.update(icp=[],xrd={'state':'PENDING'})
        second=model.predict(changed)
        np.testing.assert_allclose([r['predicted_total_wt_percent'] for r in first],[r['predicted_total_wt_percent'] for r in second])
        np.testing.assert_allclose([r['p_xrd_pass'] for r in first],[r['p_xrd_pass'] for r in second])
        self.assertEqual(model.training_ids,[])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model';model.save(path);self.assertEqual(model.model_id,SciencePriorModel.load(path).model_id)

    def test_scientific_descriptor_option_is_explicitly_supervised(self):
        from sacfactory.scientific_descriptors_model import ScientificDescriptorModel
        rows=specimens(30);records,_=collect(rows,{},'REAL','periodic',use_thermal=False)
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}),tempfile.TemporaryDirectory() as folder:
            model=ScientificDescriptorModel('Gradient boosting').fit_records(records,'REAL')
            self.assertIn('supervised',model.info['label_disclosure'])
            path=Path(folder)/'model';model.save(path)
            loaded=ScientificDescriptorModel.load(path)
            self.assertEqual(model.model_id,loaded.model_id)


if __name__=='__main__':unittest.main()
