# Compare models and follow returned results

## Before testing: compare the same recipes

1. Open **Next batch**, select a trained model and generate a batch preview.
2. Expand **Compare saved models on these recipes**. Choose two or more installed
   models and click **Compare recipe forecasts**. No training is started.
3. Inspect each model's total ICP W / wt%, structural PASS probability and support
   flag on the identical proposed recipes. The range columns show the difference
   between model predictions; they are not calibrated uncertainty intervals.
4. Click **Add batch to pending**. Both the selected acquisition forecast and the
   comparison forecasts are kept with their creation time, recipe and model ID.

An already saved group retains its original comparison forecasts. For changed
parameters, save the Pending edits and request a new forecast before entering
any results. The original proposal and manual forecast history are preserved.

## After testing: Forecast follow-up

Open **Data analysis → Forecast follow-up**. The shared experiment, cohort and
Batch filters apply. Select any elements in the inline periodic table and the
saved model versions to display. Use a cross-cohort scope to examine several
single-element campaigns together.

The module uses stored predictions, without refitting on the returned outcomes:

- **ICP regression:** each point is one quantified sample-element/model pair;
  predicted W / wt% is horizontal and measured W / wt% is vertical. The dashed
  line is y=x. Separate fitted agreement lines show slope and intercept.
- **Structural calibration:** points aggregate saved probabilities into five
  fixed bins. The vertical coordinate is the observed PASS fraction. Individual
  structural observations remain binary; failures are evaluated explicitly.
- **Eligibility audit:** inspect missing results, ND/BLQ, stale or changed
  recipes, post-result forecasts, superseded revisions and training overlaps.
- **Exports:** download endpoint pairs, model metrics and the full audit as CSV.

For each endpoint, only the latest matching forecast saved before that endpoint
was first recorded enters the metrics, once per physical specimen and model
version. A later, invalid post-result revision does not hide an earlier valid
forecast. ICP and structural results have separate first-recorded timestamps.
Historical rows without verified timing remain in the audit; use protected
Weighted validation for retrospective model evaluation instead of inventing
earlier forecast times. Record-entry time does not establish the physical time
at which a measurement was performed.

ND, BLQ and missing ICP are not replaced with zeros. Partial measured member
sets can contribute quantified pairs without claiming a complete mixture total.
Interrupted runs and contradictory mass fractions are excluded. Complete burnout
does not create a quantitative ICP measurement or an unperformed structural test.

## Metrics and interpretation

Let y_ie and p_ie be the measured and predicted ICP mass fraction for quantified
element e in specimen i. Each specimen receives total weight one, divided equally
among its selected quantified elements. MAE, RMSE, predictive R-squared and the
fitted agreement line use those weights. Predictive R-squared is:

```text
R2 = 1 - sum(w_ie * (y_ie - p_ie)^2)
         / sum(w_ie * (y_ie - weighted_mean(y))^2)
```

It is unavailable when the measured values are constant. The fitted line is a
diagnostic and does not replace predictive R-squared or alter saved predictions.

For a structural outcome z_i (PASS=1, FAIL=0), the Brier score is
mean((p_i-z_i)^2). A FAIL outcome predicted with PASS probability 0.1 contributes
0.01, while a FAIL predicted with PASS probability 0.9 contributes 0.81. Lower is
better. FAIL recall and balanced accuracy use each model's saved decision
threshold. Balanced accuracy requires both observed classes.

Public pretrained training membership is withheld. Its follow-up results describe
agreement with saved forecasts; they do not establish an independent test of the
original training set. **Require verified local training separation** restricts
metrics to a traced local snapshot, stable specimen identities and an actual
synthesis Batch outside that snapshot. This does not by itself establish
separation of repeated recipes, external laboratories or experimental protocols.

Original experimental records, returned evidence and local forecast histories
remain in the user's data directory. The public source package contains code,
templates, numeric inference assets and explicitly synthetic demonstrations.
