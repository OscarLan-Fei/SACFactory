# Bundled SAC-Gen inference models

This directory contains trained **SAC-Gen-Diffusion** and **SAC-Gen-GB** forward
models. Downloading the repository includes these assets. No experimental database,
API key or initial model training is needed for inference.

## Contents and privacy boundary

- JSON files contain the learned structural head, aggregate feature normalizers,
  physical inference configuration and architecture/schema information.
- The diffusion NPZ contains numeric neural-network state tensors. Loading uses
  `allow_pickle=False`; no executable estimator pickle is distributed.
- The GB JSON stores numeric decision-tree thresholds, branches and leaf values,
  plus its intercept. Training objects and per-node training counts are excluded.
- The registry checks file hashes. Reference identities are checked against the
  bundled software. Keep code, references and model assets from the same revision.

The export does **not** include training recipes or measured ICP/XRD records,
sample/Batch identifiers, record identities, source databases, split membership,
bootstrap lists, training curves, per-sample validation or private run reports.
These are inference parameters learned from experiments, not a dataset release.

## Use

Start SACFactory, select **Real experiments**, then **Discovery lab → Predict**.
Choose a pretrained variant, elements, precursor amounts in mmol, PAN mass, DMF
volume, controlled cutoff and pulse count. Click **Predict recipe**. Predictions
are not saved as measured experiments. GB provides deterministic point forecasts;
the diffusion model also provides sampling-based loading summaries.

For inverse search, select a **Pretrained** model under **Inverse design**.
Exact private training-recipe membership is not distributed, so these releases
conservatively flag support as unverified. Review the ranges and explicitly enable
flagged extrapolation before accepting inverse-search proposals.

Command-line example, from the project folder:

```text
python scripts/predict_pretrained.py --recipe examples/pretrained_query.json --model SAC-Gen-GB --output forecast.json
```

Python API:

```python
from sacfactory.pretrained import load

model = load("SAC-Gen-GB", device="cpu")
forecast = model.predict([{
    "elements": ["Fe"],
    "concentrations_mM": {"Fe": 50.0},
    "pulse_limit_C": 1000,
    "pulse_count": 1,
    "pan_mass_g": 2.0,
    "solvent_volume_mL": 20.0,
    "origin": "REAL",
}])
```

Here 50 mM in 20 mL is 1 mmol of precursor. Outputs use ICP W / wt% and a
structural-assessment PASS probability. The declared precursor/support/protocol
context is supplied by the model and must not be changed to an incompatible
process. Actual precursor chemistry can affect predictions; unknown declarations
retain the software's missing-information masks.

## Interpretation

Model parameters are fixed for this release. User data are not used to retrain
the bundled files. New local candidates remain separate, versioned artifacts.
In-sample forecasts from these models are not independent validation. For model
comparison, fit each workflow within the chosen held-out protocol.

Exact training-composition membership is private, so support flags are conservative.
Loading draws and learned probabilities do not guarantee coverage or calibration
on another laboratory's samples. Quantitative metal content and a recorded
structural judgment do not independently establish atomic dispersion. The public
release does not expose frozen private forecasts for validation or reconstruct
the author's experimental history.
