# SAC-Gen design

SAC-Gen combines recipe controls, chemical references and explicit physical
assumptions to predict continuous ICP metal content and a recorded structural
assessment. The application keeps the observed outcomes, model forecasts,
validation results and deployment checkpoints as distinct objects.

## Model variants and information controls

| Model | Inputs and inductive assumptions | Learner |
| --- | --- | --- |
| **SAC-Gen-Diffusion** | Recipe controls, chemical descriptors, physical retention reference and shared mass-conservation decoder | Diffusion Transformer loading residual; regularized structural-assessment head |
| **SAC-Gen-GB** | The same physical framework and structural-assessment interface | Histogram gradient-boosting loading residual |
| **Data-only Gaussian process / Random forest / Gradient boosting** | Recorded controls and nominal element identity; no elicited thermal-retention reference | Supervised numerical baselines |
| **Scientific descriptors Gradient boosting** | Recorded controls and sourced chemical/thermal descriptors; no elicited heating or retention functions | Supervised descriptor control |
| **Science prior (fixed)** | Declared mass, thermal, mixing and kinetic assumptions | Fixed calculation without fitting current experimental labels |

These controls test different information contracts. The fixed scientific prior
includes empirical anchors and is not a first-principles prediction. A descriptor
model still uses measured labels for supervised fitting. Equal outer masks compare
workflows; they do not make different feature sets identical.

## Prediction pathway

1. **Encode the planned recipe.** Element-specific concentration, PAN mass,
   solvent volume, pulse count and controlled cutoff are available before testing.
   Reference properties and declared precursor chemistry augment these inputs.
   Missing chemical states are masked rather than inferred from measured outcomes.
2. **Predict loading residuals.** The learned correction modifies a physical
   metal-retention reference. A shared recovered-matrix denominator converts all
   members to W / wt%, with retained metal bounded by feed mass.
3. **Generate loading summaries for the structural head.** Totals, marginal
   spread and molar summaries come from predicted ICP. Source-training summaries
   are generated under inner batch or recipe masks; outer-test ICP is excluded.
4. **Predict structural-assessment probability.** A regularized logistic head
   combines recipe/chemical features, a masked mechanistic prior and the generated
   summaries. It returns PASS and FAIL probabilities plus the declared threshold.
5. **Review and reserve future experiments.** Next batch uses a selected local
   deployment checkpoint, allowed ranges and an acquisition objective. Manual
   edits require a refreshed forecast before the result is returned.

The public user interface exposes the two SAC-Gen variants above. The underlying
loading API also retains Gaussian, conditional-diffusion and Transformer families
for compatible research workflows and existing checkpoints. SAC-Gen-GB returns
deterministic point estimates: repeated point-mass support in the common decoder
API is not a predictive uncertainty distribution.

## Structural labels and missing outcomes

The structural-assessment resolver uses a resolved primary XRD/STEM judgment
first. If the primary field is unresolved, an explicitly recorded legacy STEM
conclusion can supply PASS or FAIL. Unperformed or inconclusive tests stay pending.
The probability head targets this resolved recorded assessment, independently of
the ICP pass judgment. Keep assay identity and evidence provenance when reporting
results; pooled assessments are not equivalent to a single standardized assay.

Neither predicted ICP nor an ICP/XRD screening label independently proves atomic
dispersion. Continuous loading, structural assessment and optional confirmation
remain separately inspectable in the data model.

## Protected training and refinement

Random recipe-grouped cross-validation holds identical complete recipes together,
including repeated specimens from different batches. Batch validation instead
keeps whole synthesis batches together. Select the mask that matches the intended
generalization claim; recipe grouping can leave batch identifiers in both pools.

The optional validated token-training variant uses a source-only recipe holdout
for a bounded learning-rate search and early stopping, then refits on the full
source pool using the selected budget. Inner scalers are fitted only on inner
training observations. Optional structural-head refinement selects regularization,
compares monotone sigmoid calibration with identity, and selects a decision
threshold using source-pool cross-fitted scores. The complete pipeline is assessed
by the outer masks. These inner diagnostics are not additional independent tests.

The gradient-boosting residual variant likewise selects its declared tree
configuration using a source-only recipe split. All compared methods must retain
the same outer masks, outcome definitions and reported prediction coverage.

## Evaluation and use

Report ICP MAE, RMSE, predictive R-squared and per-element errors, together with
structural probability metrics and calibration. Composite outcome regression and
post-result agreement are different views, defined in [Methods](METHODS.md).
Keep score definitions fixed across model comparisons.

Deployment training creates a checkpoint for future proposals; it does not create
new test evidence. Saved model identity includes the training and configuration
information needed to detect stale inputs. REAL and DEMO remain isolated.

The repository contains algorithms and synthetic demonstration data. Private
experimental records, validation exports and trained checkpoints are excluded.
Train the relevant models locally before using their deployment checkpoints.

## Implementation map

| Responsibility | Source |
| --- | --- |
| SAC-Gen orchestration, nested loading features, checkpoint registry | [sac_gen.py](sacfactory/sac_gen.py) |
| Loading bridge and physical decoder interface | [pulseicp_bridge.py](sacfactory/pulseicp_bridge.py) |
| Token training with source-only selection | [tuned_token_residual.py](sacfactory/tuned_token_residual.py) |
| Gradient-boosting residual | [gb_residual.py](sacfactory/gb_residual.py) |
| Structural-head refinement | [xrd_refinement.py](sacfactory/xrd_refinement.py) |
| Recipe masks and separation checks | [validation_groups.py](sacfactory/validation_groups.py) |
| Information controls | [data_only.py](sacfactory/data_only.py), [scientific_descriptors_model.py](sacfactory/scientific_descriptors_model.py), [science_only.py](sacfactory/science_only.py) |
| Matched comparisons | [model_benchmarks.py](sacfactory/model_benchmarks.py) |

The research scripts under `scripts/` use locally supplied records. Several
training scripts explicitly select CUDA; the parallel refinement launcher uses
Windows process flags. Use the application's device controls for portable CPU/GPU
selection. Frozen campaign reports and scripts tied to private run identities are
not included in the public source package.
