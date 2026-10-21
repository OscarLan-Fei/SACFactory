# Methods

## Inputs and outcomes

Pre-synthesis inputs are element composition, precursor amount/concentration,
PAN mass, DMF volume, pulse count and controlled temperature setpoint. Elemental
and declared precursor descriptors supplement these controls. Measured results
are excluded from future-recipe features. Concentrations are stored in mM and
ICP W in wt%. At 20 mL, 1 mmol corresponds to 50 mM.

XRD PASS/FAIL is the recorded experimental judgment. It is not inferred by
automatically reading an XRD pattern. ICP is a continuous loading target, not a
pass-probability task. STEM confirmation is optional and recorded separately.
ICP plus XRD screening does not alone establish atomic dispersion.

Chemical references retain source information and missing-value masks.
Precursor formal charge does not establish the final supported-metal oxidation
state. State-specific ionic radii require matching state/coordination/spin.
The declared Fe(II) square-planar high-spin radius of 64 pm is a modeling
assumption, not an experimental spin assignment.

## SAC-Gen

The loading learner predicts a conditional residual distribution relative to
a physical retention reference. Gaussian, conditional diffusion, Transformer
and diffusion-transformer architectures are available. The inherited joint
mass decoder shares one recovered-matrix denominator across mixture members
and bounds metal retention by feed mass.

The XRD head is L2 logistic regression with C=0.1. Its inputs are recipe and
chemical descriptors, the logit of a mechanistic XRD prior with a missing mask,
and generated ICP total-mean, marginal-spread and molar-summary features.
The sum of marginal standard deviations is a feature, not a joint confidence
interval. The mechanistic prior is an assumption requiring validation.

Training XRD loading features are generated with up to three inner whole-batch
ICP masks. An outer-source model generates outer-test loading features.
Observed test ICP never enters the XRD head. All scalers and learned weights
are fitted on source observations. The XRD head requires twelve judgments and
both outcome classes; unavailable fits remain missing.

## Evaluation

Rotating batch validation holds out disjoint batch groups until each eligible
sample has been evaluated. Unseen-element, new-recipe, chronological and
single-to-mixture protocols address different generalization questions.
Repeated recipes across batches are allowed in batch validation.

Report ICP MAE, RMSE and R², and XRD Brier score, ROC-AUC and calibration.
Individual agreement scores are:

    S_ICP = 100 / (1 + mean_e[((W_pred - W_obs) / s_W)^2])
    S_XRD = 100 * (1 - (p_PASS - y_PASS)^2)
    S = w_ICP * S_ICP + (1 - w_ICP) * S_XRD

Weights and error scale must be reported with a score. Agreement is not percent
classification accuracy or material quality. Correct FAIL forecasts receive
the same reward as equally accurate PASS forecasts. Missing endpoints do not
receive invented measurements or automatically renormalized weights.

The regression figure plots predicted against measured ICP W, including XRD
FAIL samples with measured W. The identity line and descriptive OLS line are
distinct; OLS does not recalibrate the predictions. XRD uses calibration plots
and unmodified binary observations.

## Deployment

Deployment refits the selected architecture on available source records and
saves neural weights, XRD coefficients, configuration and training identity.
These fits do not create new test-performance estimates. Updated or deleted
training records invalidate the checkpoint fingerprint. REAL and DEMO are
isolated. CPU/GPU selection affects numerical execution; hardware and budgets
should be reported for timing comparisons.

Next-batch acquisition supports boundary learning, high-loading XRD failure
mapping and pass-region validation. These proposal objectives do not change
how prediction correctness is scored. Pending edits require a refreshed
forecast and preserve the prediction made before outcomes were recorded.

## Interpretation

Cross-validation used to choose models or weights is development evidence,
not an additional independent final test. Physical anchors and loss-rate
assumptions are not independently validated mechanisms. Demonstrating method
advantages requires fair baselines, ablations and appropriate independent or
prospective evaluation. See BENCHMARKS.md for the comparison protocol.


## Predicted and measured composite outcome scores

This result index is separate from the post-result prediction-agreement score.
For whole-recipe ICP mass percent W and XRD PASS state y, define
`g(W) = W / (W + k) * (100 + k) / 100` and
`S_obs = 100 * [a * g(W_obs) + (1-a) * y]`.
The corresponding plug-in forecast is
`S_pred = 100 * [a * g(W_pred) + (1-a) * p_XRD_PASS]`.
Defaults are a = 0.8 and k = 2.5 wt%. k is a reference scale, not a loading
target or an aggregation threshold. The coordinate is continuous on physical
mass percent 0 to 100 and has no hard target-attainment plateau.

Observed scores use measurements only. Forecasts use whole-batch held-out ICP
means and XRD probabilities only. Priors are model inputs and do not enter the
observed score. This plug-in forecast does not integrate the nonlinear mapping
over predictive draws and is not described as E[S]. Both PASS and FAIL records
are retained: matching a low observed outcome score is an accurate prediction.
A high outcome score is not itself prediction accuracy, synthesis quality or
proof of atomic dispersion. Changing the weights changes the prediction target;
it does not demonstrate an improvement in the underlying model.

Whole-recipe totals require every recipe member, including members outside a
plot selection when explicit complete totals are saved. Missing endpoints and
out-of-range mass percentages are excluded without zero filling, clipping or
weight renormalization. Multi-model score comparisons use the intersection of
complete physical specimens, with coverage and exclusions exported.

## Detailed model-comparison diagnostics

All compared models use common paired support. ICP plots show original held-out
predictions, an identity reference, and a descriptive weighted OLS line. Predictive
R2 is computed from the original predictions; fit R2 is computed after the
plot-only OLS fit and is not a replacement for predictive R2. MAE, RMSE, bias,
slopes, intercepts and per-element results are exported. Metric weights give each
physical specimen equal mass and share that mass equally across selected members.
Residual and box plots display individual paired element errors without jitter.

XRD diagnostics count each physical specimen once and include Brier, ROC-AUC,
average precision, ROC, precision-recall and ten-bin calibration. Wilson intervals
are approximate binomial reference intervals and do not model dependence within
synthesis batches. Single-class selections do not receive ROC-AUC or average
precision values. Display filters and outcome-score settings do not retrain models.


## Multi-model weighted validation

The weighted-validation model selector uses the matched predictive benchmark
engine for conventional models and mixed selections. All selected estimators
share the exact outer partitions, descriptor profile, quantitative training
eligibility and scoring parameters. Supported protocols are rotating batches,
batch holdout, purged new-recipe holdout, chronological batches, unseen elements
and single-to-multiple-element transfer. Source-insufficient folds are recorded
as unavailable. Chronological training batches precede the held-out batch;
unseen-element masks remove target elements and batches from source training.

The conventional continuous/probability pairs are mean/prevalence,
Ridge/logistic, random forest, histogram gradient boosting and fixed-kernel
Gaussian process. SAC-Gen uses the selected neural loading architecture and
inner-cross-fitted loading summaries in its XRD head. Equal outer masks compare
complete workflows; they do not isolate architecture from all auxiliary features.
All model-score plots use identical complete specimens. Conventional out-of-range
loading forecasts remain in raw endpoint diagnostics and are excluded from the
physical outcome index without clipping. Model selection is a development step,
not an independent confirmation of superiority.
