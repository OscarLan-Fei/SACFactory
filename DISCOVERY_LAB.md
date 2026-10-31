# Discovery lab

Discovery lab links recipe prediction, literature context, model updates,
experimental decisions and returned evidence. Start with **Real experiments →
Discovery lab → Predict** to use the bundled trained SAC-Gen models immediately.
See the [model card](pretrained/MODEL_CARD.md) for inputs and inference limits.

| Tab | Purpose | Requires local experimental data? |
| --- | --- | --- |
| **Predict** | Enter a proposed recipe and forecast ICP content and structural-assessment probability with the included trained models | No |
| **Evolution** | Freeze forecast rounds, link pending experiments and track returned feedback without overwriting the original prediction | Only for evaluating returned results |
| **Knowledge** | Search cited method summaries, user-indexed documents and explicitly selected local experiment records | No for bundled literature summaries |
| **Model disagreement** | Compare the forecasts of selected forward models on existing recipes | Yes for historical-recipe comparisons |
| **Set model & updates** | Train and verify a versioned candidate using the local records | Yes |
| **Inverse design** | Search bounded precursor amounts, temperatures and pulse counts for a target loading interval or learning objective | No when using a bundled pretrained model |
| **Hypotheses** | Register controlled one-factor contrasts with predictions before testing | A local baseline recipe is needed |
| **Characterization** | Triage uploaded XRD patterns or SEM/STEM images and record reviewed evidence | Uploads remain local; result linkage requires a local sample |
| **Validation** | Run protected grouped validation and equal-budget acquisition replay | Yes |

## First prediction

1. Open **Discovery lab → Predict** in REAL.
2. Select **SAC-Gen-GB** for a fast point forecast, or **SAC-Gen-Diffusion** for
   neural loading sampling.
3. Select elements directly in the periodic table. Enter each precursor amount
   in mmol. Defaults are 2 g PAN, 20 mL DMF, 1000 C and one pulse.
4. Click **Predict recipe**. Inspect total and per-element ICP W / wt%, the
   structural PASS probability and the support disclosure. Download the forecast
   if needed. No experimental outcome is invented or saved.

## Proposed experiments and feedback

Inverse design is a bounded target-conditioned search using a forward model;
it does not claim to train a separate generative distribution over recipes.
Choose elements, amount bounds, controlled-cutoff bounds, allowed pulse counts,
target ICP interval and probability requirements. Review all proposals before
using **Freeze forecasts and add to Pending**. For public pretrained models,
exact training-recipe membership is withheld; explicitly permit the conservative
support flag if you want to search these conditions.

Returned results are entered through **Samples & data**. The campaign ledger
retains the forecast made before the experiment and its model identity. A model
trained on updated local records is a new candidate; it does not replace frozen
forecasts. Local candidates and public pretrained releases are separate choices.

## Characterization and research memory

Characterization tools extract candidate peaks, morphology or site-like features.
They support review and preserve source information; they do not certify atomic
dispersion or supply calibrated chemical-class probabilities. Reviewed assessments
can be linked to a saved sample. SEM morphology is not a single-atom conclusion.

Knowledge retrieval uses lexical search over source-linked summaries and selected
local material. Uploaded documents remain local and are not automatically sent to
an LLM API. Retrieved passages are evidence, never executable instructions.

## Evaluation boundary

Validation keeps recipe groups separated and distinguishes fit, ensemble-weight,
calibration and outer-test roles. Acquisition replay is retrospective. Changes
in round error can reflect different experiment difficulty and do not alone prove
causal improvement. A bundled model's prediction on records that helped train it
is not a held-out result; refit within each outer mask for comparisons.

The repository contains Discovery lab source, cited public summaries, synthetic
tests and inference-only SAC-Gen assets. It excludes private campaign ledgers,
uploaded characterization files, documents, experimental databases and original
training checkpoints.
