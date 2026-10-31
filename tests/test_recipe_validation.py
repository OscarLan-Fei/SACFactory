"""Random recipe masks preserve all specimen outcomes and nested separation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from test_pulseicp_bridge import specimens
from test_sac_gen import budget
from sacfactory.validation_groups import RANDOM_RECIPE_CV, recipe_folds, assert_mask
from sacfactory.weighted_validation import collect, split_records
from sacfactory.sac_gen import SACGen
from sacfactory.compute_runtime import execution


class RecipeValidationTests(unittest.TestCase):
    def test_reproducibility_coverage_and_duplicate_recipe_separation(self):
        rows=specimens(60)
        duplicate=copy.deepcopy(rows[0]);duplicate.update(sample_id='REPEAT',batch_id='OTHER_BATCH')
        rows.append(duplicate)
        first=recipe_folds(rows,5,42);second=recipe_folds(rows,5,42)
        self.assertEqual([b.tolist() for _,b in first],[b.tolist() for _,b in second])
        held=[]
        for train,test in first:
            assert_mask([rows[i] for i in train],[rows[i] for i in test],True)
            held.extend(test.tolist())
        self.assertEqual(sorted(held),list(range(len(rows))))
        self.assertEqual(next(n for n,(_,b) in enumerate(first) if 0 in b),
                         next(n for n,(_,b) in enumerate(first) if 60 in b))

    def test_outer_and_nested_loading_masks(self):
        rows=specimens(60)
        records,_=collect(rows,{},'REAL','chemistry_pulseicp')
        splits=split_records(records,rows,'REAL',['Fe'],RANDOM_RECIPE_CV,5)
        self.assertEqual(sum(len(test) for _,test,_,_ in splits),60)
        settings={**budget(),'crossfit_grouping':'recipe'}
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            model=SACGen(settings=settings).fit(rows,'REAL')
        by_id={s['sample_id']:s for s in rows}
        held=[]
        for fold in model.xrd_info['inner_folds']:
            self.assertNotIn('unavailable_reason',fold)
            assert_mask([by_id[i] for i in fold['train_ids']],[by_id[i] for i in fold['held_ids']],True)
            self.assertFalse(set(fold['loading_training_ids']) & set(fold['held_ids']))
            held.extend(fold['held_ids'])
        self.assertEqual(sorted(held),sorted(by_id))

    def test_matched_comparison_uses_the_same_recipe_masks(self):
        from sacfactory.model_benchmarks import compare
        rows=specimens(60)
        with execution({'device':'cpu','threads':1,'cpu_jobs':1}):
            points,summary,info=compare(rows,{},'REAL',['Fe'],models=['Mean / prevalence','SAC-Gen'],
                sac_family='Gaussian residual',sac_settings=budget(),protocol=RANDOM_RECIPE_CV,max_folds=2)
        self.assertEqual(set(points.model),{'Mean / prevalence','SAC-Gen'})
        self.assertTrue((summary.ICP_n==60).all())
        self.assertTrue((summary.XRD_n==60).all())
        self.assertEqual(len(info['partitions']),2)

    def test_saved_validation_opens_without_training(self):
        from streamlit.testing.v1 import AppTest
        from sacfactory.model_benchmarks import compare
        rows=specimens(60)
        points,summary,info=compare(rows,{},'REAL',['Fe'],models=['Mean / prevalence'],
            protocol=RANDOM_RECIPE_CV,max_folds=2)
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)/'sac_gen'/'real';directory=base/'test_run'/'validation'
            directory.mkdir(parents=True)
            (base/'latest.json').write_text(json.dumps({'run_id':'test_run','model_id':'test_model',
                'validation_protocol':RANDOM_RECIPE_CV}))
            points.to_csv(directory/'out_of_fold_predictions.csv',index=False)
            summary.to_csv(directory/'validation_metrics.csv',index=False)
            (directory/'validation_methods.json').write_text(json.dumps(info))
            script='import json\nfrom sacfactory.ui_validation import render_saved_deployment_validation\n'
            script+=f'rows=json.loads({json.dumps(rows)!r})\n'
            script+=f'render_saved_deployment_validation({folder!r},"REAL",rows)'
            ui=AppTest.from_string(script,default_timeout=60).run()
            self.assertFalse(ui.exception)
            ui.button(key='show_saved_deployment_validation_REAL_test_model').click().run()
            self.assertFalse(ui.exception)
            self.assertFalse(ui.error)
            self.assertTrue(ui.session_state['saved_deployment_validation_REAL_test_model']['points'].shape[0])


if __name__=='__main__':unittest.main()
