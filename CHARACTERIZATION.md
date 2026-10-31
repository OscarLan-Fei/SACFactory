# Characterization recognition and feedback

Discovery lab > **Characterization** is the reviewed-evidence workflow.
It associates SEM, XRD and STEM measurements with a physical sample, extracts
candidate features, records reviewed observations and commits a reviewed
screening assessment to the experiment loop.

## Current capability

| Measurement | Input | Local output | Interpretation |
| --- | --- | --- | --- |
| SEM | Single-frame TIFF, PNG or JPEG; optional calibrated nm/pixel and analysis ROI | Bright-feature components, equivalent diameters and positions | Morphology context; candidates may be support features or artifacts. SEM alone does not resolve an isolated-atom judgment. |
| XRD | Exported two-column CSV/TXT/XY/DAT: increasing 2theta in degrees and intensity | Baseline, prominent candidate peaks, widths and optional sourced reference proximity | Candidate crystalline evidence. Reference matching requires explicitly matching measurement/reference wavelengths. Peak proximity is not a validated phase identification. |
| STEM | Single-frame exported image; optional calibrated nm/pixel and ROI | LoG bright-spot candidates, coordinates and nearest-neighbor distances | Positions for review, not automatic element identities or a measured single-atom fraction. |

All automatic analyses initially return **PENDING**, with no calibrated class
probability. These are classical signal/image-processing baselines. AtomAI,
EMcopilot, microSAM and XRD-AutoAnalyzer weights are **not** installed or presented
as validated SAC recognizers. Their papers and implementation links are indexed
in Knowledge for the next development stage.

## Operation

1. Select the sample and measurement type. Choose a useful field of view and
   export one image frame or a two-column diffraction pattern.
2. Enter verified image calibration when available. Keep units unknown when it
   is absent. Crop scale bars, instrument text and labels out of the analysis ROI.
3. For XRD reference comparisons, provide phase name, reference peak position,
   source and wavelength. Verify the instrument wavelength explicitly. The
   software does not assume Cu radiation for an unknown instrument.
4. Click **Extract candidate features**. Inspect the source and candidates; tune
   the extraction settings as needed. This does not change any experiment label.
5. Click **Save source and analysis for review**. Original bytes are retained
   under a SHA-256 name together with algorithm version, parameters and sample
   recipe identity.
6. Review each saved field or scan and record an observation, reviewer and notes.
7. Inspect the combined evidence. Enter the final screening state and criteria,
   then click **Commit reviewed assessment to Loop**.
8. Review Evolution. The result is now available to existing masked validation,
   structural-probability training and future candidate updates. Train a new
   candidate deliberately; the commit button does not start training or replace
   a deployment model.

## Evidence fusion

Reviewed target crystalline peaks or STEM clusters support a screening FAIL.
Reviewed isolated-site candidates, or absence of resolved target crystalline
peaks within the measured XRD scan, can support a screening PASS. This retains
the existing application's screening definition: XRD non-detection is **not**
proof of atomic dispersion. Record scan range, detection sensitivity and
complementary characterization when interpreting a material.

SEM supplies morphology context. Reported aggregates trigger a warning and a
request for structural interpretation; SEM alone cannot produce the structural
training label. Coexisting isolated-site and crystalline/cluster observations
are flagged as conflicting evidence. An explicit reviewer interpretation is
required before resolving that conflict. Inconclusive evidence remains pending.
Existing resolved judgments require an explicit override before being replaced.

The original per-file observations remain separate from the unified screening
assessment. Conflicting files are not silently discarded. Correctly predicting
a reviewed FAIL remains useful predictive performance and does not become a
successful synthesis.

## Loop and data protection

The committed assessment uses the existing reviewed XRD/STEM outcome interface.
Consequently, campaign Brier scores/class recalls and subsequent model fits see
the new label without a second competing target definition. Original forecasts
remain immutable. Raw images, peak positions and post-synthesis classifications
do not enter recipe-only prediction inputs; this prevents outcome leakage.

Sample revision and assessment event are committed in one attached-database
transaction, with rollback on failure. The source checksum is checked during
review and final assessment. Evidence for a changed recipe is excluded. Sample
data and campaign events remain separate for REAL and DEMO. Source upload and
feature extraction never overwrite existing manual judgments automatically.

Original ICP values and their existing reporting time are retained. Committing
only structural evidence does not invent an ICP result. Measurements and review
records can be exported through the discovery ledger; files remain local and are
not sent to a language-model API.

## Literature and implementation roadmap

| Reference | Useful design | Required SAC-specific validation |
| --- | --- | --- |
| [AtomAI, Nature Machine Intelligence (2022)](https://www.nature.com/articles/s42256-022-00555-8) | Trainable atomic/mesoscopic segmentation and positions | Annotated HAADF-STEM fields with separate specimens and instruments held out |
| [AtomAI reusability report, Nature Machine Intelligence (2025)](https://www.nature.com/articles/s42256-024-00958-9) | Independent reproduction and domain-transfer checks | Reproduce baseline behavior before claiming a new-material advantage |
| [XCA, Nature Computational Science (2021)](https://www.nature.com/articles/s43588-021-00059-2) | Probabilistic diffraction classification from physical reference data | Relevant target/support structures, mixed phases, realistic noise and detection limits |
| [Adaptive XRD, npj Computational Materials (2023)](https://www.nature.com/articles/s41524-023-00984-y) | Request measurements in informative scan regions | Compare fixed and targeted scans at equal acquisition budgets |
| [EMcopilot, npj Computational Materials (2025)](https://www.nature.com/articles/s41524-025-01800-5) | Segmentation under morphology and contrast variability | Real held-out microscopy fields; generated training data must not derive from test images |
| [Segment Anything for Microscopy, Nature Methods (2025)](https://www.nature.com/articles/s41592-024-02580-4) | Reviewer-assisted annotation and segmentation | Catalyst-specific annotation agreement and particle segmentation; biological demonstrations are not SAC validation |
| [CRESt, Nature (2025)](https://www.nature.com/articles/s41586-025-09640-5) | SEM statistics returned to a multimodal experimental loop | Field-of-view dependence and measured prospective selection gains |

Bibliographic metadata were checked against Crossref where reachable and the
publisher's accessible records. Some direct article requests redirected to
institutional login; accessible abstracts, figures/captions and author code
repositories informed this implementation. This is not a claim to have read all
subscription full texts or reproduced their results.

For learned recognition, group splits by parent specimen and source image before
cropping. Keep all fields, crops and augmentations from one specimen in the same
fold. Report localization precision/recall and count/size error for microscopy;
phase-wise precision/recall and unknown-phase rejection for diffraction; and
class-wise recall, Brier score, abstention and conflict rates for the unified
assessment. Synthetic detector tests establish software behavior, not real-image
accuracy. Real annotated measurements are required before reporting recognition
performance or enabling unreviewed automatic labels.
