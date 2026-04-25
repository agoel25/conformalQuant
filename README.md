# Conformal Prediction vs Volatility Baselines on US Equities

A systematic empirical study of whether
online conformal-prediction methods produce practically useful daily
return prediction intervals for US equities, and whether they survive
the high-volatility crisis periods where reliable intervals matter
most.

## Question

Do CP methods produce better prediction intervals than properly
calibrated rolling GARCH and FHS on US equity returns through the
2008, 2020, and 2022 crisis windows?  And does base-model complexity
(LSTM / Transformer) help once the uncertainty-quantification layer is
fixed?

## Method categories

| Category        | Models                                          | Wrappers                       |
|-----------------|-------------------------------------------------|--------------------------------|
| `Baselines`     | FHS, rolling GARCH(1,1) Student-t               | none (parametric intervals)    |
| `Simple_Native` | histogram-binning, Quantile Regression Forest   | none (native intervals)        |
| `Simple_CP`     | linear AR (`num_lags = 5`)                       | split, ACI, AgACI              |
| `Complex_CP`    | LSTM(50), Transformer encoder(d=32, h=4)         | split, ACI, AgACI              |

A separate `TCP_Paper` notebook documents *why* TCP / TCP-RM were
dropped from the main grid.  See `src/analysis/TCP_Paper/README.md`.

## Quick start

```bash
# Set up the environment (Python 3.12+, conda or venv)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the full grid (~10 min on CPU; faster with GPU for deep models)
python scripts/run_grid.py

# Render figures + summary tables
jupyter notebook src/analysis/results.ipynb
```

`scripts/run_grid.py --only {baselines, simple_native, simple_cp, complex_cp, tcp_paper, all}`
runs subsets.  Each per-category Jupyter notebook under
`src/analysis/<Category>/` re-runs only that slice and renders one
timeline figure per `(model, wrapper, ticker, regime)` cell into
`report/figures/timelines/<Category>/`.

## Repository layout

```
src/
  conformal/      Volatility baselines (garch.py, fhs.py), CP wrappers (split,
                  aci, agaci, tcp_paper), and native-interval methods
                  (hist_binning_native.py, tree_native.py).
  models/         Point predictors used as bases for CP wrappers
                  (linear_ar, lstm, transformer_encoder).
  metrics/        Metric implementations.  Spec lives in metrics/Metrics.md.
  data/           yfinance loader and lag-window transform.
  analysis/       One folder per method category, plus results.ipynb
                  aggregator and TCP_Paper standalone sanity check.
  plotting.py     Shared timeline-figure helper used by every notebook.
scripts/
  run_grid.py         End-to-end grid runner with --only flag.
  build_artifacts.py  Aggregate metrics + paper figures (CLI form of results.ipynb).
papers/           Reference PDFs (TCP, FHS, Fantazzini).
report/           LaTeX + figures + compiled PDF.
results/          Per-(model, wrapper, ticker, regime) interval CSVs (gitignored).
datasets/         Cached yfinance data (gitignored).
```

## Where to read more

* `implementation.md` — full technical reference: workflow, time
  estimates, where every output lands, decision log.
* `src/metrics/Metrics.md` — evaluation spec; load-bearing for
  cross-model comparability.
* `src/analysis/<Category>/README.md` — brief description of every
  model in that category with paper references.
