# User guide

## Record experiments

Select elements from the periodic table. Enter precursor amount in mmol or
concentration in mM; changing either updates the other using DMF volume.
Defaults are 2 g PAN and 20 mL DMF. Record pulse count and temperature setpoint.
Measured peak temperature, voltage and current are separate recorded fields.

Enter ICP W as mass percent: 2.5 means 2.5 wt%. Record XRD PASS/FAIL directly.
Unperformed tests may remain pending; STEM is optional. Complete material loss
is recorded as BURNED_OUT, rather than inferred from two failed judgments.

Sample and Batch IDs advance automatically. Experiment records support edits
and deletion. Pending samples support revised controls and later result entry.

## Train and validate

Choose the shared experiment selection under Data analysis, then open Weighted
validation. Select elements, architecture and validation protocol. Training and
scoring settings are collapsed by default. Run masked validation for held-out
predictions; Train deployment model fits a model for future recipes.

The default device is Auto. A CUDA-capable PyTorch installation enables NVIDIA
GPU execution. CPU and worker budgets can be set explicitly. Small models may
run faster on CPU. The packaged software does not include private checkpoints.

## Plan the next batch

Enable the trained SAC-Gen checkpoint in Next batch. Set concentration, pulse
and temperature ranges, choose an acquisition objective and generate a preview.
Review predicted ICP loading and XRD probabilities, then add samples to pending.
Edited pending controls require a refreshed prediction. Recorded outcomes never
overwrite the forecast saved before testing.

## Compare methods

Model comparison contains separate predictive-model and acquisition-policy
experiments. Select models or policies, set a common budget and run the
comparison. Export tables and methods together. See BENCHMARKS.md for the
meaning and limits of each comparison.

## Data and language model

REAL and DEMO records and models are separate. DEMO is explicitly synthetic.
The optional language-model connection requires a provider key and explicit
data-sharing settings. Experiment management and numerical modeling work
locally without a language-model API.


## Regression and composite-score plots

Weighted validation displays the predicted-versus-measured composite outcome
score first. ICP regression and post-result agreement are separate expandable
sections. Score weights and loading reference affect only the score display;
existing held-out predictions are reused. Use the same fixed settings for every
model being compared. Download paired scores and regression coefficients.

Model comparison provides Regression fits, Residuals and element errors, XRD
probabilities, and Composite score fits. All element choices support the periodic
table. Regression tables distinguish predictive R2 from descriptive fit R2.


## Selecting validation models

In Weighted validation, select one or more entries under Validation models, or
use Select all validation models. Options are SAC-Gen, Mean / prevalence,
Ridge / logistic, Random forest, Gradient boosting and Gaussian process.
The neural architecture selector applies to SAC-Gen. Conventional estimators
use CPU workers. Their selection does not change the deployed Next batch model.

Click Run masked validation once. Selected models use the same protocol,
whole-batch or element masks and outcome formula. The primary display compares
predicted and measured composite scores by model. ICP and XRD model diagnostics
contains endpoint metrics, regression fits, residuals and probability curves.
All prediction pairs, metrics and fold definitions can be downloaded.

Selecting a different model set or protocol requires new validation. Changing
only the displayed outcome-score weights reuses the stored held-out predictions.
SAC-Gen alone retains the same individual validation workflow. Empty model
selection disables the run button. Source-insufficient folds remain unavailable;
no test observation is moved into training to fill a missing prediction.
