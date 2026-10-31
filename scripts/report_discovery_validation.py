"""Assemble an English HTML report from saved, actual discovery evaluations."""
import argparse
from html import escape
import json
from pathlib import Path
import sys
import pandas as pd

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    manifest=read('validation_manifest.json');checks=read('regression_tests.json')
    grouped=read('grouped_validation.json');replay=read('acquisition_replay.json')
    assert manifest['status']=='PASS' and checks['passed']
    frame=pd.read_csv(out/'model_summary.csv')
    table=frame[['model','ICP_MAE_wt_percent','ICP_RMSE_wt_percent','structural_Brier',
                 'FAIL_recall','PASS_recall','structural_balanced_accuracy']].copy()
    table.columns=['Model','ICP MAE / wt% points','ICP RMSE / wt% points','Brier','FAIL recall','PASS recall','Balanced accuracy']
    curves=pd.read_csv(out/'replay_learning_curves.csv')
    final=curves[curves.observations==curves.observations.max()]
    replay_table=final[['policy','observations','ICP_MAE_wt_percent','ICP_RMSE_wt_percent','FAIL_recall','observed_elements']].copy()
    replay_table.columns=['Policy','Observed recipes','ICP MAE / wt% points','ICP RMSE / wt% points','FAIL recall','Observed elements']
    ensemble=frame[frame.model=='Calibrated physical ensemble'].iloc[0]
    fold_table=pd.DataFrame([{'Fold':f['fold'],'Fit':len(f['fit_keys']),'Weight fit':len(f['weight_keys']),
                             'Calibration':len(f['calibration_keys']),'Test':len(f['test_keys'])} for f in grouped['folds']])
    tests=sum(len(f['test_keys']) for f in grouped['folds'])
    assert len({k for f in grouped['folds'] for k in f['test_keys']})==tests
    def html_table(data):return data.to_html(index=False,border=0,float_format=lambda value:f'{value:.4f}',na_rep='Unavailable')
    candidates=html_table(pd.DataFrame([{k:c[k] for k in ['name','model_id','activation']} for c in manifest['new_candidates']]))
    feature_rows=[
        ('1. Self-evolving campaigns','Frozen source, forecast and model versions; pending-to-feedback ledger','Immutable forecast, idempotent import and edited-recipe exclusion checks'),
        ('2. Source-linked memory','Local TF-IDF retrieval, PDF/text import and explicitly shared chat history','Source and sharing-boundary checks'),
        ('3. Disagreement and calibration','Nonnegative weight fit, conformal intervals and optional Platt calibration','Disjoint recipe pools, small-group gates and held-out coverage'),
        ('4. Set representation','SAC-Gen-Set element tokens, pooling and physical joint decoder','Permutation equivalence, physical bounds, checkpoint reload and five-fold validation'),
        ('5. Target-conditioned search','Bounded mmol feed search, process grid, constraints and batch diversity','Unit conversion, support checks and UI-to-Pending integration'),
        ('6. Hypothesis contrasts','One-factor registration, saved predictions and later feedback','Factor isolation and timezone-aware post-registration measurement checks')]
    features=html_table(pd.DataFrame(feature_rows,columns=['Feature','Delivered','Validation']))
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>SAC-Gen Discovery lab — validation</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#173047;background:#f4f7fa;margin:0}}main{{max-width:1180px;margin:36px auto;padding:36px;background:white;border-radius:12px}}h1{{line-height:1.2}}h2{{margin-top:36px;color:#075e67}}table{{width:100%;border-collapse:collapse;font-size:14px;margin:16px 0}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #d9e3ea;vertical-align:top}}th{{background:#edf5f7}}.scroll{{overflow:auto}}.note{{padding:16px;background:#eef6fa;border-left:4px solid #23788a}}code{{overflow-wrap:anywhere}}a{{color:#076a84}}@media print{{body{{background:white}}main{{margin:0;padding:8px;max-width:none}}h2,table{{break-inside:avoid}}}}</style></head><body><main>
<h1>SAC-Gen Discovery lab: implementation and validation</h1>
<p>Software 0.14.0 · 31 October 2026 · REAL data · development evaluation</p>
<p class="note">All six workflows are integrated. The focused regression suite passed
<strong>{checks['tests_run']} tests</strong>. Existing SAC-Gen-GB remains the strongest ICP predictor in this
comparison. The new Set model and discovery ensemble remain selectable candidates; no primary model was replaced.</p>
<h2>1. Delivered workflows</h2><div class="scroll">{features}</div>
<p>The integration test generates a batch in the interface, freezes its forecasts, imports six pending
rows into an isolated test database, returns a measured result and evaluates its original forecast.
No test result is written to the REAL experiment database. Tabs are lazy: opening them does not fit models.</p>
<h2>2. Source protection and evaluation design</h2>
<p>The source contained {manifest['source_samples']} records. {tests} eligible specimens each appeared in an outer test fold once.
Five recipe-group folds keep every component of a specimen together. Each outer training pool has separate
model-fit, ensemble-weight-fit and interval-calibration recipe groups. The Set model used 100 training epochs;
the random seed was 42. Fold-trained SAC-Gen-GB, Data GB and fixed prior were evaluated on the same tests.</p>
<div class="scroll">{html_table(fold_table)}</div>
<p>Real experiment items: <strong>unchanged</strong>. Primary deployment registry: <strong>unchanged</strong>.
Source snapshot fingerprint: <code>{escape(manifest['source_fingerprint'])}</code>.</p>
<h2>3. Held-out predictive results</h2><div class="scroll">{html_table(table)}</div>
<p>Entries are unweighted means of the five fold metrics. Within a fold, ICP errors give equal weight
to physical specimens and average the component errors of a multielement specimen. A wt% point is an absolute
mass-percentage difference. Structural metrics use recorded XRD/STEM assessments; they do not quantify atomic dispersion.</p>
<p>SAC-Gen-GB achieved the lowest ICP error. SAC-Gen-Set predicted PASS for every structural test case
at the 0.5 threshold, giving zero FAIL recall and balanced accuracy 0.5. Its low overall Brier score should
not obscure this failure under class imbalance. Data GB had the highest balanced accuracy in this run.
The ensemble weights were fitted to ICP error, not structural discrimination; the fused structural head
therefore needs further class-specific development. No claim of superior Set-model accuracy is supported.</p>
<h2>4. Calibration</h2>
<p>The validation ensemble combined fold-trained fixed prior, Set and SAC-Gen-GB using weights fitted
outside both the interval-calibration and test pools. At nominal 95% coverage, mean component coverage was
{ensemble.component_coverage:.2%}, simultaneous specimen coverage {ensemble.simultaneous_specimen_coverage:.2%},
and mean interval width {ensemble.mean_interval_width_wt_percent:.4f} wt% points. These are empirical
held-out results. They do not guarantee coverage for an unseen chemistry domain. The small full-source
deployment candidate currently combines Set and fixed prior; its composition differs from this validation ensemble.</p>
<h2>5. Equal-budget acquisition replay</h2>
<p>All policies started from the same 16 recipes, added eight per round, and ended at 48 revealed recipes.
A separate 20% recipe-group test pool was untouched throughout. The same Set learner (40 epochs) evaluated
every policy. GP-BO used Gaussian-process UCB on revealed structural labels; it was not an ICP regression model.</p>
<div class="scroll">{html_table(replay_table)}</div>
<p>GP-BO had the lowest final ICP error in this single-seed replay; disagreement acquisition did not beat
random selection. All replay learners had zero FAIL recall at the final budget. Learning curves were not
monotonically improving. This is a retrospective finite-pool result, not evidence of prospective discovery gain.
Repeated seeds and future controlled rounds are required before claiming a better acquisition policy.</p>
<h2>6. Saved candidates and operation</h2><div class="scroll">{candidates}</div>
<p>Open <strong>Discovery lab</strong> in the sidebar. Use Inverse design to generate proposals, inspect
both mmol and mM, and freeze them into Pending. Return results in Samples &amp; data; inspect Evolution.
Candidate families are available under Set model &amp; updates and can be selected within the discovery tools.</p>
<p>Research memory uses lexical retrieval, not dense semantic embeddings. Inverse design is bounded
target-conditioned search with a forward model, not a separate learned recipe generator. The workflow supports
self-evolving experiment cycles; measured improvements remain an empirical question. Equipment is not operated automatically.</p>
<h2>7. Evidence files</h2><ul>
<li><a href="model_summary.csv">Model summary CSV</a> and <a href="fold_metrics.csv">per-fold metrics</a></li>
<li><a href="grouped_validation.json">Full grouped evaluation, masks and weights</a></li>
<li><a href="calibrated_predictions.csv">Observed/predicted component values and intervals</a></li>
<li><a href="replay_learning_curves.csv">Replay learning curves</a> and <a href="acquisition_replay.json">query traces</a></li>
<li><a href="regression_tests.json">Regression result</a> and <a href="regression_tests.log">individual checks</a></li>
<li><a href="validation_manifest.json">Source/deployment protection manifest</a></li>
<li><a href="../SACFactory/DISCOVERY_LAB.md">Operating guide and method boundaries</a></li></ul>
</main></body></html>'''
    (out/'VALIDATION_REPORT.html').write_text(page,encoding='utf-8')
    print(json.dumps({'report':str(out/'VALIDATION_REPORT.html'),'tests':checks['tests_run'],'held_out_specimens':tests}))


if __name__=='__main__':main()
