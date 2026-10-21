# SACFactory

SACFactory is a local research workbench for single-atom catalyst experiments.
Its SAC-Gen model predicts continuous ICP loading and XRD PASS/FAIL probability.

## Install and run

Use Python 3.12 and a virtual environment:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m streamlit run app.py --server.port 8517
```

Open http://127.0.0.1:8517 in a browser. On Windows, Start-SACFactory.cmd starts the local service after the .venv
environment is installed. NVIDIA GPU execution requires a
compatible CUDA PyTorch installation; Auto selects an available device.

## Workflow

1. Record compositions, precursor quantities, preparation conditions and outcomes.
2. Validate SAC-Gen with held-out batches, then train a deployment checkpoint.
3. Generate the next batch and return results through pending experiments.
4. Compare predictive models and acquisition strategies on matched data.

ICP is recorded as W / wt%. Amounts and concentrations are synchronized using
DMF volume. REAL and synthetic DEMO data are separate. A language-model API is
optional and requires explicit sharing settings.

## Documentation

- [User guide](USER_GUIDE.md)
- [Model and evaluation methods](METHODS.md)
- [Predictive and acquisition benchmarks](BENCHMARKS.md)
- [Source acknowledgments](CREDITS.md)

The source package contains code, templates, synthetic examples and tests.
Private experimental databases, credentials and trained checkpoints are excluded.
Comparison results are evidence to inspect, not an automatic claim of superiority.

## Synthetic demonstrations and private records

Select **Simulated demo** to use the included synthetic campaign. The packaged
demo contains 1,476 simulated preparations and is not experimental evidence.
Real experiments start in a separate empty local database. This repository
does not contain the author's experimental records, raw assays, API credentials
or models trained on private experimental results. Runtime databases and model
checkpoints are excluded by .gitignore.

This repository stores the application source. Uploading it to GitHub does not
automatically deploy a hosted application. Use Python 3.12 when configuring a
separate deployment; the Streamlit entry point is app.py.

The synthetic fixture is stored as `examples/demo_seed.json.gz` and loaded
automatically using Python gzip support. Regenerated local JSON is also supported.
