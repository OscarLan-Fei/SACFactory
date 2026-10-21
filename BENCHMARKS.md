# Benchmarks

## Predictive models

Data analysis / Model comparison / Predictive models includes:

| Model | ICP regressor | XRD classifier |
| --- | --- | --- |
| Mean / prevalence | Source mean | Source prevalence |
| Ridge / logistic | Scaled ridge, alpha=10 | Scaled logistic, C=0.1 |
| Random forest | 120 trees, minimum leaf=3 | Same budget |
| Gradient boosting | 100 iterations, 15 leaves, L2=1 | Same budget |
| Gaussian process | Scaled Matérn 2.5 + white noise | Scaled Matérn 2.5 |
| SAC-Gen | Selected conditional residual network | Trained nested-loading XRD head |

All selected models share outer batch folds, source observations, quantitative
eligibility and the selected pre-synthesis descriptor profile. SAC-Gen adds
learned ICP summaries and its mechanistic XRD prior to its XRD head: this is
a comparison of complete workflows. Controlled feature/physics ablations are
needed to attribute gains to individual components.

Tables use the common eligible endpoint intersection across selected models;
coverage before intersection is also reported. Pooled ICP errors give equal
weight to specimens and equal weight to selected members within a specimen.
Statistical regressor predictions are not clipped to improve metrics.
Prevalence predictions remain defined for a single source class; trained
classifiers require both classes. Unavailable models are recorded explicitly.

Baseline parameters are fixed. GP kernel optimizers are disabled; length scale
is the square root of feature dimension and regression white-noise level is
0.05. These are reproducible initial baselines, not hyperparameter-optimized
upper bounds. A final superiority study should tune competing models within
inner training folds using comparable search budgets and evaluate the chosen
models on independent data. No automatic superiority claim is generated.

## Acquisition policies

Data analysis / Model comparison / Acquisition strategies includes random
sampling, greedy max-min space filling, GP expected improvement, GP upper
confidence bound and SAC-Gen predictive-entropy sampling.

The replay uses a finite pool of recorded recipes. Recipe duplicates are
collapsed by sample-ID order, independent of results. Whole batches are held
out for evaluation before acquisition. All policies share initial observations,
the candidate pool, query count and evaluation checkpoints. A fixed random
forest trained only on revealed observations evaluates learning on the untouched
test batches. Its ICP MAE and XRD Brier describe acquisition effects without
changing the evaluation learner between policies.

GP-BO uses the recorded XRD PASS indicator as its revealed objective. EI uses
xi=0.01; UCB uses beta=2. Batch scores use a 0.05 diversity term. SAC-Gen is
refitted only on revealed experiments; full-data deployment weights are never
loaded. Unsupported trained probabilities fall back to space filling and are
listed in the methods export. No unobserved recipe receives a fabricated
experimental outcome.

BO targets PASS discovery; entropy sampling targets boundary learning. Held-out
predictive error, PASS/FAIL discoveries and element coverage are separate
outcomes. Discovering a FAIL is not automatically a model failure. Replay
cannot establish prospective closed-loop experimental efficiency and should
not be presented as a physical experiment campaign.

## Exports

Both comparisons export tables, individual test predictions, data identity,
fold membership and settings. Acquisition also exports query order, initial
observations and held-out batches. Run multiple prespecified seeds and retain
batch-aware uncertainty before making publication-level comparative claims.
