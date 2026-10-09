# Implementation Notes

Technical reference: what's built, why, where the gotchas are, how
the workflow runs end-to-end, and how long each step takes.  See
`README.md` for the overview and headline results.

---

## 1. Project context

* **Question:** do CP methods produce better daily-return prediction
  intervals than properly-calibrated rolling GARCH and FHS on US
  equities through 2008, 2020, and 2022 crisis windows? Does
  base-model complexity (LSTM / Transformer) help once the
  uncertainty-quantification layer is fixed?
* **Method categories** evaluated against `Baselines` (FHS, rolling
  GARCH):  `Simple_Native` (histogram-binning, QRF), `Simple_CP`
  (LinearAR + {split, ACI, AgACI}), `Complex_CP` (LSTM, Transformer
  + {split, ACI, AgACI}).
* **Excluded** from the main grid: TCP / TCP-RM (separate
  `TCP_Paper` sanity shows they lose to a properly-fitted rolling
  GARCH), EnbPI (compute cost vs marginal benefit).
* **Crises:** 2008 Lehman {AIG, GNW, MGM}, 2020 COVID {MCD, MGM, AZO},
  2022 rate-hike {SNAP, NFLX, META} — 9 (ticker, regime) pairs total
  (MGM appears twice).
* **Confidence level:** 1−α = 0.90 (α = 0.10) throughout.

---

## 2. Repository layout

```
.
├── README.md                          # public-facing overview
├── implementation.md                  # this file
├── requirements.txt
├── environment.yml                    # conda spec
├── datasets/                          # gitignored; cached yfinance panel
├── results/                           # per-cell interval CSVs +
│                                      # metrics_long.csv aggregate
├── report/                            # write-up: LaTeX source + compiled PDF
│   ├── report.tex / refs.bib / paper.sty / report.pdf
│   └── figures/                       # paper figures
│       └── timelines/<Category>/      # per-cell timeline PNG/PDF (one per cell)
├── scripts/
│   ├── run_grid.py                    # end-to-end runner with --only flag
│   └── build_artifacts.py             # aggregate metrics + paper figures (CLI form of results.ipynb)
└── src/
    ├── plotting.py                    # timeline figure + metric helpers (delegates to src.metrics)
    ├── conformal/
    │   ├── base.py                    # IntervalPredictor + BaseConformal
    │   ├── garch.py                   # GarchHardcoded + RollingGarch
    │   ├── fhs.py                     # Filtered Historical Simulation
    │   ├── hist_binning_native.py     # histogram-binning native intervals
    │   ├── tree_native.py             # Quantile Regression Forest (Meinshausen 2006)
    │   ├── split_conformal.py
    │   ├── adaptive_conformal.py      # ACI (Gibbs & Candès 2021)
    │   ├── agaci_conformal.py         # AgACI (Zaffran et al. 2022)
    │   └── tcp_paper.py               # paper-faithful TCP (sanity-only)
    ├── models/
    │   ├── base_model.py
    │   ├── linear_ar.py
    │   ├── lstm.py
    │   └── transformer_encoder.py
    ├── metrics/
    │   ├── Metrics.md                 # the metrics spec
    │   ├── prediction_cols.py
    │   ├── coverage_metrics.py        # marginal + high/low-vol-stratified coverage
    │   ├── winkler_metrics.py
    │   ├── interval_metrics.py        # widths
    │   ├── adaptation_metrics.py      # T_recovery, max_drawdown, recovered
    │   └── metrics.py                 # facade: one Metrics object → 4 sub-metrics
    ├── data/
    │   ├── data.py                    # yfinance loader, Cols/Ticker enums
    │   └── transform.py               # build_deep_data → (N, num_lags, 1) tensors
    └── analysis/
        ├── _shared.py                 # ExperimentSplit + iterate_experiments +
        │                              # walk_forward + make_wrapper + I/O
        ├── results.ipynb              # cross-category aggregator (Tables 1-3 + scatter)
        ├── regime_overview.ipynb      # crisis-window sanity-check plots
        ├── Baselines/                 # README + fhs.ipynb + garch_roll.ipynb
        ├── Simple_Native/             # README + hist_binning.ipynb + tree_based.ipynb
        ├── Simple_CP/                 # README + linear_ar.ipynb
        ├── Complex_CP/                # README + lstm.ipynb + transformer.ipynb
        └── TCP_Paper/                 # README + tcp_paper.ipynb (sanity-only)
```

---

## 3. The grid

### 3.1 Tickers (9 ticker × regime cells)

| Regime | Tickers | Onset | End | Event |
|--------|---------|-------|-----|-------|
| 2008 | AIG, GNW, MGM   | 2008-09-15 | 2009-03-31 | Lehman bankruptcy |
| 2020 | MCD, MGM, AZO   | 2020-02-20 | 2020-04-30 | S&P pre-crash peak |
| 2022 | SNAP, NFLX, META | 2022-01-03 | 2022-12-31 | S&P pre-bear peak |

### 3.2 Split geometry (per ticker × regime)

Defined in `_shared.iterate_experiments(train_years=4, calib_months=6, test_pad_months=6)`:

```
train     : [onset - 4y - 12mo,  onset - 12mo)
calibrate : [onset -      12mo,  onset -  6mo)
test      : [onset -       6mo,  end   +  6mo]
```

Per-crisis refits (no global model) so we measure "does CP stay
calibrated *into* each crisis", which needs a fresh pre-crisis fit.

### 3.3 Method × wrapper grid

| Category        | Models                | Wrappers           | Cells per category |
|-----------------|----------------------|--------------------|--------------------|
| Baselines       | fhs, garch_roll      | native             | 2 × 9 = 18         |
| Simple_Native   | hist_binning, tree_based | native         | 2 × 9 = 18         |
| Simple_CP       | linear_ar            | split, aci, agaci  | 1 × 3 × 9 = 27     |
| Complex_CP      | lstm, transformer    | split, aci, agaci  | 2 × 3 × 9 = 54     |
| **Main grid**   |                      |                    | **117 cells**      |
| TCP_Paper sanity | garch_hardcoded, garch_roll, fhs, tcp_paper | n/a | 4 × 3 = 12 (COVID 2020 only) |

### 3.4 Refit cadence

Single `REFIT_EVERY = 21` constant in `scripts/run_grid.py` shared by
*every* adaptive method (`RollingGarch`, `TCPPaperConformal`).  This is
the apples-to-apples comparison constraint.

---

## 4. Metric pipeline

* **One source of truth:** `src/metrics/Metrics`.  The `Metrics` class
  computes the high/low volatility mask once per asset (75th-percentile
  rolling-20-day variance over full sample) and exposes
  `get_coverage_metrics()`, `get_winkler_metrics()`,
  `get_interval_metrics()`, `get_adaptation_metrics()`.
* **`src/plotting.py :: compute_cell_metrics`** delegates to `Metrics`
  so plot titles match the aggregate tables exactly.
* **`compute_cell_metrics_full`** returns the full Metrics.md schema:
  marginal cov, high/low-vol-stratified cov + width + Winkler,
  in-crisis cov/width/Winkler, T_recovery, max_drawdown_cov.
* **`results/metrics_long.csv`** is built by
  `scripts/build_artifacts.py` (and equivalently by
  `src/analysis/results.ipynb`).  One row per (model, wrapper, ticker,
  regime), all metrics columns.

The metrics spec (`src/metrics/Metrics.md`) is the document to read
before changing anything in `src/metrics/`.

---

## 5. End-to-end workflow

### 5.1 Environment

```bash
# venv (works out of the box)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# OR conda
conda env create -f environment.yml && conda activate conformalquant
```

Python 3.12+ required.  Direct deps: `numpy`, `pandas`,
`scikit-learn`, `torch`, `arch`, `scipy`, `yfinance`, `matplotlib`.

### 5.2 Reproduction commands

```bash
# Full grid (~10 min on CPU)
python scripts/run_grid.py

# Or run subsets
python scripts/run_grid.py --only baselines       # ~1 min
python scripts/run_grid.py --only simple_native   # ~1-2 min  (QRF is the slow one)
python scripts/run_grid.py --only simple_cp       # ~30 s
python scripts/run_grid.py --only complex_cp      # ~3-5 min on CPU; <1 min on GPU
python scripts/run_grid.py --only tcp_paper       # standalone TCP sanity (1 min)

# Aggregate metrics + paper figures
python scripts/build_artifacts.py
# OR run src/analysis/results.ipynb interactively
```

### 5.3 Per-model notebooks

Each model has its own notebook (e.g. `lstm.ipynb`, `fhs.ipynb`).
Every per-model notebook is self-contained and idempotent:

1. Bootstraps `sys.path` so `from src...` works regardless of CWD.
2. Runs the model on every `(ticker, regime)` cell across its
   applicable wrappers, *skipping* cells whose CSV already exists in
   `results/` (set `FORCE_RERUN = True` to override).
3. Loads every CSV for that model and renders one timeline figure per
   cell into `report/figures/timelines/<Category>/`.
4. Prints a per-cell metrics table and a summary aggregated by
   wrapper (or just by ticker/regime for native-interval models).

### 5.4 Time estimates

Wall-clock measurements from a CPU-only laptop (i7, 16 GB RAM):

| Step                          | Time (CPU)  | Time (GPU)  | Notes |
|-------------------------------|-------------|-------------|-------|
| `--only baselines`            | ~1 min      | ~1 min      | 18 cells |
| `--only simple_native`        | ~10 min     | ~10 min     | QRF predict is leaf-lookup-bound (35 s/cell) |
| `--only simple_cp`            | ~30 s       | ~30 s       | LinearAR is trivial |
| `--only complex_cp`           | ~3-5 min    | ~30-60 s    | LSTM + Transformer × 3 wrappers × 9 cells |
| `--only tcp_paper`            | ~1 min      | ~1 min      | only run once for sanity |
| `scripts/build_artifacts.py`  | ~5 s        | ~5 s        | reads CSVs, writes metrics + figures |

If you re-run with everything cached the runner skips already-present
CSVs (delete the relevant files in `results/` to force a rerun).

### 5.5 Where outputs land

```
results/
├── <model>_<wrapper>_<ticker>_<regime>.csv     # one CSV per cell
└── metrics_long.csv                            # aggregate metrics

report/figures/
├── timelines/
│   ├── Baselines/<model>_<wrapper>_<ticker>_<regime>.{png,pdf}
│   ├── Simple_Native/...
│   ├── Simple_CP/...
│   └── Complex_CP/...
└── aggregate/
    ├── scatter_cov_winkler.{png,pdf}           # cross-method scatter
    └── rolling_coverage.{png,pdf}              # 3-panel rolling cov per crisis
```

The interval CSV schema is in `_shared.REQUIRED_COLUMNS`:
`{date, point, l, u, r, model, wrapper, ticker, regime}`.

---

## 6. Architecture details

### 6.1 Base predictors (`src/models/`)

Subclass `TimeSeriesModel(label_col_name, num_lags)`.  Implement
`fit(df)` and `predict(df) -> float`.  `predict` returns the next
value after the last `num_lags` rows of the dataframe — single-step
only.

| Class                  | File                       | Algorithm |
|------------------------|----------------------------|-----------|
| `LinearARModel`        | `linear_ar.py`             | sklearn `LinearRegression` on lagged features |
| `LSTMModel`            | `lstm.py`                  | single-layer LSTM(50) + linear head, mini-batch Adam, early stopping |
| `TransformerModel`     | `transformer_encoder.py`   | 1-layer Transformer encoder (d=32, h=4, dropout=0.1) + linear head |

`HistogramBinningModel` and `TreeBasedModel` (point-predictor versions)
are kept for back-compat but are no longer in the main grid — the
native-interval versions in `src/conformal/{hist_binning_native,tree_native}.py`
replace them.

### 6.2 Conformal wrappers + native interval methods (`src/conformal/`)

| Class                       | File                       | One-liner |
|-----------------------------|----------------------------|-----------|
| `RollingGarch`              | `garch.py`                 | rolling-fit GARCH(1,1) Student-t with parametric quantile (no conformal layer) |
| `GarchHardcoded`            | `garch.py`                 | TCP paper's exact GARCH (ω=1e-6, α=0.05, β=0.9, normal); sanity-only |
| `FHS`                       | `fhs.py`                   | Filtered Historical Simulation (Barone-Adesi et al. 1999) |
| `HistogramBinningNative`    | `hist_binning_native.py`   | quantile-bin lookup of training next-step returns |
| `QuantileTreeNative`        | `tree_native.py`           | Quantile Regression Forest (Meinshausen 2006) |
| `SplitConformal`            | `split_conformal.py`       | fixed half-width = (1-α)-quantile of |residuals| |
| `AdaptiveConformal`         | `adaptive_conformal.py`    | Online α update (Gibbs & Candès 2021) |
| `AgACIConformal`            | `agaci_conformal.py`       | Aggregated ACI (Zaffran et al. 2022) |
| `TCPPaperConformal`         | `tcp_paper.py`             | Paper-faithful TCP: quantile GBM + rolling split-conformal (sanity-only) |

All `IntervalPredictor` and `BaseConformal` subclasses share the same
two-method interface: `fit_and_calibrate(train, calib)` and
`predict_with_interval(data) -> (point, lower, upper)`.  Wrappers that
expose `.adapt(real_value)` are duck-typed by `walk_forward` and
called automatically.

### 6.3 Analysis utilities (`src/analysis/_shared.py`)

| Symbol                   | Purpose |
|--------------------------|---------|
| `iterate_experiments()`  | Yields one `ExperimentSplit` per (ticker, regime) cell |
| `walk_forward(...)`      | Steps a wrapper one trading day at a time across the test window; duck-types `adapt` |
| `make_wrapper(...)`      | Factory for the CP wrappers; defaults `confidence_level=0.90` |
| `save_intervals(...)`    | Writes the standardised CSV schema |
| `MODEL_CATEGORIES`       | Maps every model name to one of `{baseline, simple_native, simple_cp, complex_cp}` |
| `CRISIS_WINDOWS`         | The exogenous crisis onset/end dates and ticker baskets |
| `RESULTS_DIR`            | Project-root `/results` |

### 6.4 Plotting (`src/plotting.py`)

| Symbol                             | Purpose |
|------------------------------------|---------|
| `plot_interval_timeline(...)`      | Per-cell figure (filled-band interval + predicted point + realised returns + crisis shading) |
| `compute_cell_metrics(...)`        | Coverage / Winkler / width — delegates to `src.metrics.Metrics` |
| `compute_cell_metrics_full(...)`   | Full Metrics.md schema (incl. high/low-vol stratification, in-crisis subsets, T_recovery) |
| `render_cell_grid(...)`            | Convenience: render one figure per cell from a raw multi-cell DataFrame |

---

## 7. Decision log

Chronological non-obvious choices, for context if any get revisited.

1. **Per-crisis refits, not one global model.** The core question is
   "does CP stay calibrated *into* each crisis", which needs a fresh
   pre-crisis fit, not a global 20-year model.
2. **Crash-only ticker baskets.**  Stable baskets exist in `data.py`
   but aren't iterated; they dilute the headline comparison.  Re-add
   as controls if a reviewer asks.
3. **Exogenous crisis onsets.**  Deriving onset from returns would
   introduce circularity into the adaptation-speed metric.
4. **2022 onset = 2022-01-03 (S&P peak), not 2022-03-16 (first Fed
   hike)** — captures the pre-hike sell-off.
5. **Confidence level pinned at 0.90.**  `BaseConformal` defaults to
   0.95 but every entry point overrides it; results assume 0.90 throughout.
6. **`REFIT_EVERY = 21` shared across every adaptive method.**  TCP
   refitting every step (paper default) vs GARCH never refitting (TCP
   paper baseline) is an unfair comparison; we fix to monthly for
   everyone.
7. **Use the `arch` library, not hand-rolled GARCH.**  The earlier
   manual recursion had a double-counting bug in `predict_with_interval`
   (last calib observation applied twice).  Using `arch_model.fix(params)`
   with explicit percent-unit scaling kills both bug classes.
8. **Drop the conformal q_hat layer from `RollingGarch`.**  Adding an
   empirical quantile correction on top of a properly-fitted Student-t
   parametric quantile doubly-inflates intervals; pure parametric is
   the right baseline.
9. **TCP / TCP-RM excluded from the main grid.**  The
   `TCP_Paper` sanity shows they lose to rolling GARCH and FHS once
   GARCH is fairly fitted; including them would muddy the headline
   without changing it.
10. **Quantile Regression Forests for `tree_based`** (Meinshausen 2006)
    instead of three pinball-loss GBMs.  More principled: one fit,
    every quantile readable.
11. **EnbPI kept implemented but excluded from main grid.**  Adds
    compute cost without changing the qualitative finding; one-line
    re-enable in `run_grid.py` if needed.

---

## 8. Sanity check

COVID 2020 (MCD / MGM / AZO) ranking, used as a quick end-to-end
check after pipeline changes:

```
fhs              cov 0.908   W 0.128   width 0.083
garch_roll       cov 0.934   W 0.134   width 0.102
linear_ar × aci  cov 0.841   W 0.155   width 0.062
lstm × aci       cov 0.848   W 0.156   width 0.062
transformer × aci cov 0.839  W 0.155   width 0.061
hist_binning     cov 0.766   W 0.185   width 0.046
tree_based       cov 0.776   W 0.186   width 0.046  (QRF; ~2x slower)
linear_ar × split cov 0.731  W 0.202   width 0.042
```
