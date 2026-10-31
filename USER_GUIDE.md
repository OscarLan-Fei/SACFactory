# SACFactory user guide

The workflow connects recorded recipes, chemical representation, model
evaluation, deployment training and new experiments. See
[Local deployment with Codex](INSTALL_WITH_CODEX.md) for setup and
[Methods](METHODS.md) for equations and fitting details.

## Choose a data space

**Real experiments · REAL** holds your local experimental records.
**Simulated demo · DEMO** holds the synthetic campaign. Records and models remain
separate. A DEMO model is a demonstration model, not a model validated on your
experiments. Algorithms are included; private trained checkpoints are not.

## Workspace reference

### Experiment chat

Use local shortcuts for supported experiment tasks. For a research conversation,
configure **Language model**, enable requests and choose records to share. The
conversation helps interpret data; numerical engines perform fitting and
evaluation. A chat response is not an experimental observation.

### Samples & data

| Tab | Purpose |
| --- | --- |
| **Fill batch table** | Enter several samples with shared defaults and per-element details |
| **Experiment records** | Inspect, edit or delete saved samples and measured outcomes |
| **Pending experiments** | Edit proposed conditions before synthesis and refresh forecasts |
| **Result feedback** | Return results for a saved pending sample |
| **Bulk import / report** | Import structured records or export records and reports |
| **Register sample** | Register one sample through the detailed workflow |

Select elements from the periodic table. Precursor amount and concentration
synchronize using the solution volume:

```text
concentration_mM = 1000 * amount_mmol / DMF_volume_mL
amount_mmol = concentration_mM * DMF_volume_mL / 1000
```

**1 mmol in 20 mL = 50 mM**; 0.1 mmol = 5 mM. This conversion describes solution
concentration, not final ICP loading. Preparation defaults are **2 g PAN**,
**20 mL DMF**, **32 V** and **80 A**; edit them to match the actual experiment.

Record pulse count and controlled cutoff setpoint. Measured peak temperature
is separate; pulse duration is not assumed to be a fixed controlled input.
Do not treat a default as a measurement without checking it.

ICP is **W / wt%**: enter `2.5` for 2.5 wt%, not 0.025. Preserve ND/BLQ flags.
Direct ICP and XRD PASS/FAIL judgments can be saved without completing every
optional evidence field. Numeric loading regression still requires eligible
quantified measurements. STEM is optional and separate. Record complete material
loss as **BURNED_OUT**; two failed assay judgments do not establish burnout.

Automatic Sample IDs use `SAC_xxxx`; manual IDs are supported. Batch IDs advance
with saved batches. Inspect the preview when deleting or renumbering records,
since identifiers connect samples to saved forecasts.

### Materials & consumables

Maintain precursor compounds, hydrates, types and declared chemical states,
alongside support and preparation materials. Chloride defaults are editable
catalogue references; confirm the actual reagent. A catalogue default does not
prove a historical reagent identity or the final supported-metal oxidation state.

Chemical properties retain sources and missing-value information. Ionic radii
require matching oxidation state, coordination and spin. The Fe(II) square-planar
HS radius of 64 pm is an explicit modeling assumption. Unknown state-dependent
values remain missing rather than being guessed.

### Next batch

1. Choose target elements and process context.
2. Set allowed concentrations, pulses, cutoff temperatures and preparation basis.
   Equal lower/upper bounds fix a variable.
3. If a current SAC-Gen deployment checkpoint exists, enable it for prediction;
   otherwise inspect the available planner and any design-only status.
4. Set the acquisition objective and requested size; generate and review a preview.
5. Use **Add batch to pending** to preserve proposals before testing.

Requested size is subject to candidate ranges, deduplication and reservations.
One eligible recipe can produce one proposal even with a larger requested size.
A design-only planner can propose controls without a fitted probability.

Validation and deployment are separate fits. Edited source records can invalidate
a checkpoint fingerprint; retrain before using updated data. Edited pending
controls require a refreshed prediction. Preserve the forecast made before
outcomes were recorded so it can be compared with the later experiment.

### Data analysis

Start with shared experiment selection. Individual panels also have element,
process or batch controls and eligibility audits. Inspect the count for each
calculation rather than assuming every panel uses every stored record.

**One comparable cohort** restricts descriptive process analysis to common
composition/process context. **All cohorts: overview only** combines descriptive
totals. **Cross-cohort exploratory associations** supports broader associations;
composition and batch differences can confound them. A constant descriptor in
an Fe-only cohort has no defined correlation. A broader set can supply variation
but does not remove confounding. The batch filter is **empty = all**. Fitting
panels have their own scopes; Weighted validation and Model comparison expose
their own calculation controls.

| Analysis tab | Question | Interpretation |
| --- | --- | --- |
| **Batch overview** | What was tested and how do batches differ? | Counts and descriptive screening rates |
| **Parameter relationships** | How do outcomes vary with concentration, pulses or temperature? | Observed trends and interaction maps, not causal effects |
| **Correlations & coverage** | Which variables co-vary and which elements/pairs were explored? | Associations with availability checks and experiment coverage |
| **Chemical descriptors** | Which reference properties are available and used? | Values, provenance, missingness and model-use information |
| **Success & failure** | How do conditions and descriptors differ by outcome? | Exploratory comparisons retaining failures |
| **ICP loading prediction** | How accurately is metal content predicted? | Held-out regression, residuals, eligibility and interval diagnostics |
| **Element campaign** | What supports moving between elements and into mixtures? | Campaign coverage and transfer context |
| **Weighted validation** | Do predicted ICP/XRD outcomes and scores match observations? | Shared-mask comparisons and endpoint diagnostics |
| **Model comparison** | Which model or acquisition policy performs best under a common protocol? | Matched comparisons, not automatic superiority claims |

Element selectors support multiple elements and a periodic-table interface.
Regression requires quantified compatible ICP; mixture totals require the
necessary member measurements. A sample can remain visible in the audit while
being ineligible for a particular metric. Missing measurements are not zero.

### Language model

Configure the optional provider base URL, API format, model ID and key. Applying
settings and enabling requests are separate actions. Choose shared records
before a research conversation. Local data management and numerical training
work without this API. Keep credentials out of the repository and screenshots.

### About

Read the workbench overview and method context here. Detailed equations and
benchmark definitions are in [Methods](METHODS.md) and [Benchmarks](BENCHMARKS.md).

## Validate first, then train for use

In **Data analysis → Weighted validation**:

1. Select elements and one or more **Validation models**. Public options include
   SAC-Gen-Diffusion, SAC-Gen-GB, data-only Gaussian process/random forest/gradient
   boosting, a supervised descriptor control and a fixed science-prior control.
2. Select the **Deployment variant** when training for use: SAC-Gen-Diffusion or
   SAC-Gen-GB. The variants share the physical framework and structural-assessment
   head, with different ICP learners. Selection alone does not train a model.
3. Choose **Validation protocol** and **Test batches per fold**. Rotating batches
   can evaluate every eligible sample while keeping each outer test batch out
   of its fold's training. Unseen-element, new-recipe, chronological and
   single-to-mixture protocols address different questions.
   Random recipe-grouped validation keeps identical complete recipes together;
   it can leave batch identifiers in both source and held-out pools.
4. Open **Training and scoring settings** if needed. Start with a modest budget.
   Auto uses the available execution device; conventional models use CPU.
5. Click **Run masked validation**. Inspect coverage and unavailable-fold reasons
   alongside metrics. Export predictions and fold definitions.
6. Use **Train deployment model** to fit the model used by **Next batch**.
   This refit creates a usable checkpoint, not new test evidence.

SAC-Gen learns an ICP residual distribution around a physical retention reference
and uses a joint mass-conserving decoder. Its XRD head uses recipe/chemical
features and cross-fitted generated loading summaries; observed outer-test ICP
does not enter that head. Architectures are compact tabular models rather than
pretrained image generators or language models.

The **ICP loading prediction** tab separately offers statistical regression and
SAC-Gen loading validation. Inspect its audit, protected split and fitting budget.
This is an ICP endpoint workflow, distinct from combined ICP/XRD score validation.

## Understanding the prediction displays

| Display | Coordinates / definition | Meaning |
| --- | --- | --- |
| **ICP regression** | Predicted W on x; measured W on y, in wt% | Continuous metal-content prediction |
| **Composite outcome regression** | Predicted score on x; observed score on y | Agreement of the same fixed ICP/XRD result index |
| **Individual prediction agreement** | Sample index versus error-based score from 0 to 100 | Post-result agreement, not percent accuracy |

The composite outcome index uses whole-recipe ICP and XRD:

```text
g(W) = W / (W + k) * (100 + k) / 100
S_observed = 100 * [a * g(W_observed) + (1-a) * y_XRD_PASS]
S_predicted = 100 * [a * g(W_predicted) + (1-a) * p_XRD_PASS]
```

Defaults are `a = 0.8` and `k = 2.5 wt%`. The smooth mapping avoids a hard
target-attainment plateau. This result index does not prove atomic dispersion.
Priors inform predictions; they do not add points to observed results. Matching
a low observed score is an accurate forecast.

The separate agreement diagnostic rewards accurate PASS and FAIL forecasts
equally. Its ICP error scale and component weights control how errors map to
points; these differ from the outcome-index parameters above.

For one model, expand **ICP regression details** and **Additional agreement and
XRD calibration diagnostics**. For multiple models, expand **ICP and XRD model
diagnostics**. Model comparison includes regression fits, residuals, per-element
errors, XRD probabilities and composite fits.

The dashed line is `y = x`. Descriptive OLS summarizes the plotted relationship
without recalibrating saved predictions. Report predictive R-squared, MAE and
RMSE alongside slope/intercept. Evaluate XRD probabilities with calibration and
Brier score; observed binary labels remain binary.

Keep score definitions fixed between models. Changing a score weight can change
apparent fit without improving ICP/XRD prediction. Results used to tune weights
or select a model are development evidence, not an independent final test.

## Compare models and acquisition policies

**Model comparison** separates predictive evaluation from acquisition replay.
Use common data, outer masks and budgets. Bayesian optimization is an experiment
selection policy rather than a direct substitute for an ICP regressor; compare
policies under a stated objective and common experiment budget.

Useful evidence includes endpoint errors, calibration, unseen-element and
single-to-mixture transfer, plus ablations of chemical and physical inputs.
Report coverage and missing predictions. Retrospective replay is not prospective
laboratory validation. Synthetic DEMO tests the software on a constructed
campaign; it cannot establish experimental generalization or model superiority.
