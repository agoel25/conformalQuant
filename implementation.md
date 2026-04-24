# Implementation Notes

Working document for continuing this project. Captures everything decided/built in the session ending 2026-04-23, with enough context that another collaborator can pick up cleanly.

---

## 1. Project context

- Goal: compare prediction-interval quality of **Split Conformal Prediction** vs. **Adaptive Conformal Inference (ACI)** wrappers, across several point-prediction models, on stock-return data spanning calm periods and three crisis regimes (2008 Lehman, 2020 COVID, 2022 rate-hike cycle)
- Final deliverable is a NeurIPS-style 6-page PDF paper with code linked as a GitHub repo; notebooks are the exploration workspace, not submitted directly

---

## 2. Current state (as of last commit `07a6400`)

Branch `master`, synced with `origin/master`.

**What works end-to-end:**
- Data download + caching via `ensure_data()` → 15 tickers, 2003–present, cached to `datasets/all_tickers.csv`
- `_shared.py` utilities for splits, walk-forward, interval I/O
- Three conformal wrappers: `SplitConformal`, `AdaptiveConformal`, `EnbPIConformal` (last one added by teammate)
- Five point-prediction models: `LinearARModel`, `LSTMModel`, `TransformerModel`, `HistogramBinningModel`, `TreeBasedModel`
- Six analysis notebooks:
  - `src/analysis/regime_overview.ipynb` — pre-analysis visualization (price + vol with crisis windows marked)
  - `src/analysis/Simple/linear_ar.ipynb` — sanity baseline
  - `src/analysis/Simple/hist_binning.ipynb`
  - `src/analysis/Simple/tree_based.ipynb`
  - `src/analysis/Complex/lstm.ipynb`
  - `src/analysis/Complex/tft.ipynb` (uses `TransformerModel` — see §10 TFT caveat)
- Metric modules on master (added by teammate): `src/metrics/{coverage_metrics,winkler_metrics,interval_metrics,adaptation_metrics,metrics,prediction_cols}.py`

**What does NOT yet exist:**
- `src/analysis/results.ipynb` — the aggregator that loads all saved intervals and produces the paper's tables + figures. **This is the next major piece of work.**
- FHS and GARCH baseline model implementations
- Proper TFT (the proposal names TFT but the code is a basic transformer encoder)
- Full-grid runs for LSTM and TFT (the full-grid cells exist in both notebooks but are gated by `RUN_FULL_GRID = False` by default — flip and run when ready)

**Files untracked locally** (intentionally kept out of git by owner):
- `environment.yml` — conda env spec

---

## 3. Experimental grid

### 3.1 Tickers (9 pairs)

| Regime | Tickers | Rationale |
|---|---|---|
| 2008 | AIG, GNW, MGM | financial collapse + casino hit hardest by Lehman |
| 2020 | MCD, MGM, AZO | in-person consumer / hospitality crushed by COVID |
| 2022 | SNAP, NFLX, META | ad revenue + streaming under rate-hike pressure |

Note: MGM appears in both 2008 and 2020 baskets. Each (ticker, regime) pair gets its own model fit.

`TICKERS_*_STABLE` / `TICKERS_2008_FINANCE_RESILIENT` baskets exist in `src/data/data.py` but are **not iterated** in `_shared.iterate_experiments()`. They can be re-added as controls if a reviewer asks "is ACI's advantage specifically about regime adaptation vs general quality?".

### 3.2 Crisis windows (exogenous dates)

| Regime | Onset | End | Event |
|---|---|---|---|
| 2008 | 2008-09-15 | 2009-03-31 | Lehman Brothers bankruptcy filing |
| 2020 | 2020-02-20 | 2020-04-30 | S&P 500 pre-crash peak (alt: WHO pandemic 2020-03-11) |
| 2022 | 2022-01-03 | 2022-12-31 | S&P 500 pre-bear peak (tech selloff started before 2022-03-16 first Fed hike) |

**Critical: these dates are exogenous — derived from external events, NOT from stock returns.** Using data-derived onset dates would introduce circularity into the adaptation speed metric. See `src/metrics/Metrics.md` §Metric 5.

### 3.3 Split geometry (per ticker × regime)

Defined in `_shared.iterate_experiments()` with defaults `train_years=4, calib_months=6, test_pad_months=6`:

```
train:     [onset - 4y - 12mo, onset - 12mo)          # 4 years
calibrate: [onset - 12mo,       onset - 6mo)          # 6 months of calm
test:      [onset - 6mo,        end + 6mo]            # crisis + 6mo calm on each side
```

- Per-crisis refits (NOT one global model across 2005–2025). Proposal's question is "does CP stay calibrated *into* each crisis", which means a fresh fit on pre-crisis data.
- Split CP and ACI on the same (ticker, regime) share **one underlying model fit** (they differ only in how the interval is computed).
- Fixed 6-month pad on each side — not scaled to crisis length. Tried equal crisis:non-crisis ratio (pad = crisis length); reverted because 2020's ~50-day post-crisis tail was too short for reliable adaptation-speed measurement. See §10 Decision Log.
- Crisis fractions of the test window therefore differ across regimes (35% / 16% / 50%) — this is fine because every metric in `Metrics.md` is reported per-regime.

### 3.4 Nominal coverage

**α = 0.10 (confidence_level = 0.90).** `BaseConformal` defaults to 0.95 — you MUST pass `confidence_level=0.90` explicitly or results won't match the metrics spec. `_shared.make_wrapper()` does this for you.

---

## 4. Architecture

Two parallel class hierarchies, glued by string column names and a shared `num_lags` convention.

### 4.1 Models (`src/models/`)

All subclass `TimeSeriesModel`, implement `fit(df)` and `predict(df) -> float`. `predict` returns the **next value** after the last `num_lags` rows of the supplied dataframe — single-step only, not sequence forecast.

| Class | File | Algorithm |
|---|---|---|
| `LinearARModel` | `linear_ar.py` | sklearn `LinearRegression` on lagged features |
| `LSTMModel` | `lstm.py` | single-layer LSTM + linear head (PyTorch) |
| `TransformerModel` | `transformer_encoder.py` | 1-layer Transformer encoder + linear head (PyTorch) |
| `HistogramBinningModel` | `histogram_binning.py` | lag-window mean → quantile bin → memorised bin mean of next step |
| `TreeBasedModel` | `tree_based.py` | sklearn `GradientBoostingRegressor` over lag windows |

All constructors accept `label_col_name` and `num_lags` as the required two arguments (BaseConformal passes only these when instantiating the model class).

### 4.2 Conformal wrappers (`src/conformal/`)

All subclass `BaseConformal` (`base.py`). Each wrapper owns a model instance constructed from a `model_class` passed in. Interface:
- `fit_and_calibrate(train_data, calib_data)` — fits the underlying model on train, walks calib computing residuals, stores calibration quantile
- `predict_with_interval(data) -> (point, lower, upper)`

| Class | File | Notes |
|---|---|---|
| `SplitConformal` | `split_conformal.py` | Fixed half-width = `conformal_quantile(abs_residuals, α)` |
| `AdaptiveConformal` | `adaptive_conformal.py` | Extends `SplitConformal`. Caller must invoke `adapt(realized)` between predictions to nudge α based on coverage. |
| `EnbPIConformal` | `enbpi_conformal.py` | Added by teammate; uses `mapie` |

**The conformal quantile** (`BaseConformal.conformal_quantile`) returns `+inf` when the calibration set is too small to finite-sample-guarantee the requested coverage (rank > n). See §6 for the ACI floor fix that prevents inf widths in practice.

### 4.3 Analysis utilities (`src/analysis/_shared.py`)

Central module imported by every notebook:

| Symbol | Purpose |
|---|---|
| `ALL_TICKERS` | Union of every basket in `data.py` — what `ensure_data()` downloads |
| `CRISIS_WINDOWS` | dict `{regime: {"onset", "end", "tickers"}}` — source of truth for the 9 experiments. Currently only `*_CRASH` baskets. |
| `ExperimentSplit` | Frozen dataclass with train/calib/test timestamps + `.slice_train/calib/full(data)` helpers |
| `iterate_experiments(train_years, calib_months, test_pad_months)` | Generator yielding `ExperimentSplit` per (ticker, regime) pair |
| `test_window(onset, end, pad_months=6)` | Helper returning `(test_start, test_end)` — used by notebooks for visualization |
| `ensure_data(file_name)` | Downloads ALL_TICKERS if not cached, with auto-refresh guard when ALL_TICKERS changes |
| `walk_forward(wrapper, data, test_start, test_end, is_adaptive)` | Steps the wrapper one day at a time, records `(date, point, l, u, r)`, calls `adapt(r)` if adaptive |
| `make_wrapper(model_class, label_col, num_lags, confidence_level=0.90, step_size=None)` | Factory — Split if `step_size=None`, ACI otherwise. Defaults to 0.90 confidence (matches Metrics.md). |
| `save_intervals(df, model, wrapper, ticker, regime, results_dir=RESULTS_DIR)` | Writes tagged CSV to `results/<model>_<wrapper>_<ticker>_<regime>.csv` |
| `load_all_intervals(results_dir)` | Concatenates every interval CSV with schema validation |
| `REQUIRED_COLUMNS` | Schema gate for interval CSVs |
| `RESULTS_DIR` | Project-root `/results` directory (gitignored) |

### 4.4 Data (`src/data/`)

- `data.py` — yfinance download wrapper. Columns named `Returns_<TICKER>` and `Log_Returns_<TICKER>`. `Cols` and `Ticker` are `StrEnum`. All the `TICKERS_*` basket constants live here. `load_data` parses `Date` as a `DatetimeIndex`.
- `transform.py` — `build_deep_X` / `build_deep_data` for reshaping into `(N, num_lags, 1)` tensors for LSTM/Transformer.

### 4.5 Metrics (`src/metrics/`, added by teammate)

| File | Contents (inspect directly — not written by this session) |
|---|---|
| `Metrics.md` | Evaluation spec — source of truth for nominal coverage, crisis dates, vol threshold convention |
| `prediction_cols.py` | Column name constants for saved predictions |
| `coverage_metrics.py` | Marginal + conditional coverage |
| `winkler_metrics.py` | Winkler score |
| `interval_metrics.py` | Width stratification |
| `adaptation_metrics.py` | `T_recovery`, `max_drawdown_cov`, recovered flag |
| `metrics.py` | Probably an aggregator / facade |

These will be consumed by the not-yet-written `results.ipynb`.

---

## 5. Saved interval schema (`results/*.csv`)

Each interval CSV, one row per out-of-sample trading day:

| column | type | description |
|---|---|---|
| date | `datetime64[ns]` | trading day from the DatetimeIndex |
| point | float | model's point prediction |
| l | float | lower interval bound |
| u | float | upper interval bound |
| r | float | realized log return |
| model | string | e.g. `"lstm"`, `"linear_ar"`, `"hist_binning"`, `"tree_based"`, `"tft"` |
| wrapper | string | `"split"` or `"aci"` |
| ticker | string | e.g. `"SNAP"` |
| regime | string | `"2008"`, `"2020"`, or `"2022"` |

Filename pattern: `<model>_<wrapper>_<ticker>_<regime>.csv`. `save_intervals()` and `load_all_intervals()` both live in `_shared.py` and enforce this schema.

---

## 6. Key implementation fixes

### 6.1 ACI alpha floor

Prior `adaptive_conformal.py` had `alpha = max(min(alpha, 0.5), 0.001)`. With n = 121 calibration residuals and target α = 0.10, during sustained crisis losses α was driven toward 0.001, but `conformal_quantile` returns `+inf` for any α < 1/(n+1) ≈ 0.0082. Result: 27/304 `inf` widths in the 2020 × MSFT smoke test.

**Fix** (on master in `src/conformal/adaptive_conformal.py:68`):
```python
self.alpha = max(min(self.alpha, 0.5), 1.0 / (len(self.residuals) + 1) + 1e-6)
```

The dynamic floor is always strictly above the finite-sample support boundary, so the quantile is always finite regardless of calibration size. Smoke test after the fix: zero inf days, clean crisis adaptation visible.

### 6.2 `ensure_data()` auto-refresh

When `ALL_TICKERS` or `TICKERS_*_CRASH` change, the cached `datasets/all_tickers.csv` from an earlier run is missing columns for the new tickers. `ensure_data()` validates the cache's columns against the expected set and deletes + re-downloads if any are missing. Prevents the "I edited the basket but my notebook kept reading the old CSV" footgun.

### 6.3 Notebook path bootstrap

Each notebook starts with:
```python
import sys
from pathlib import Path

_here = Path.cwd().resolve()
for _p in (_here, *_here.parents):
    if (_p / "requirements.txt").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
```

Walks up from the notebook's CWD looking for `requirements.txt` (marker for project root), so `from src.analysis._shared import ...` works whether Jupyter is launched from the project root or a subdirectory.

---

## 7. How to run

### 7.1 Environment setup

Two paths depending on whether you want conda:

```bash
# Option A: conda (reproducible, recommended for teammates)
conda env create -f environment.yml           # environment.yml is NOT committed — see §2
conda activate conformalquant
python -m ipykernel install --user --name conformalquant --display-name "Python (conformalquant)"

# Option B: system Python 3.12 with pip
pip install -r requirements.txt
pip install ipykernel jupyterlab              # not in requirements.txt
python -m ipykernel install --user --name python3.12 --display-name "Python 3.12"
```

Requirements as of this writing: `yfinance`, `numpy`, `pandas` (indirect), `scikit-learn`, `torch`, `matplotlib` (indirect), `mapie` (ENBPI), `arch` (GARCH — added by teammate, used by future GARCH baseline).

Python 3.12+ required: uses `StrEnum` (3.11+), `typing.override` (3.12+), PEP 604 unions `X | Y` (3.10+).

### 7.2 Producing interval CSVs

Each per-model notebook has the same structure:
1. Smoke test on one split (2020 × MCD or MSFT) — should print calm coverage ≈ 0.90, crisis coverage much lower, post-crisis recovery for ACI
2. Full grid gated by `RUN_FULL_GRID` flag

**Default `RUN_FULL_GRID`:**
- `linear_ar.ipynb`, `hist_binning.ipynb`, `tree_based.ipynb`: `True` (all three fit in seconds)
- `lstm.ipynb`, `tft.ipynb`: `False` (flip and run when ready — ~9 splits × 2 wrappers × ~minutes of training each)

Output lands in `results/` (gitignored). Overwrites on re-run (CSVs are keyed by model × wrapper × ticker × regime).

### 7.3 Next step: results.ipynb (not yet written)

Proposed structure:
1. `load_all_intervals()` → one long DataFrame
2. For each (model, wrapper) combination, compute the five Metrics.md metrics using the teammate's functions in `src/metrics/`
3. Produce:
   - **Main results table** per asset + one averaged: Marginal Cov, Cov_high, Cov_low, Gap, W_overall, W_high, W_low, Width_high, Width_low
   - **Crisis-specific tables** per crisis: Coverage, Winkler, Width, T_recovery, Max_drawdown_cov, Recovered
   - **Scatter figure** `Cov_low` vs `Cov_high` per model with diagonal reference line — one per asset + one averaged (per Metrics.md, this is the headline figure)
   - **Adaptation-speed robustness paragraph** — re-run with `threshold=0.80` and `persistence=1`, report whether model ranking changes

Tolerate partial interval coverage: the aggregator should run and produce partial tables even if some models haven't been run yet.

### 7.4 Regenerating regime overview

`src/analysis/regime_overview.ipynb` is standalone — load data, plot cumulative returns and rolling vol per (regime, ticker) with crisis windows shaded, plus a numeric summary table. Run this first to visually confirm the crisis windows correspond to real stock moves. **Restart the Jupyter kernel after any edit to `_shared.py` or `data.py`** — Jupyter caches imports and will show stale `CRISIS_WINDOWS` otherwise.

---

## 8. Numeric baselines (smoke tests from session)

All ran on 2020 × MCD split with α = 0.10, `num_lags=5`, default ACI step 0.01, fixed 6-month pad:

| Model | Wrapper | Calm cov | Crisis cov | Post cov | Mean width |
|---|---|---|---|---|---|
| Linear AR | Split | 96.0% | 42.0% | 79.7% | 0.045 |
| Linear AR | ACI | 92.9% | 52.0% | 91.4% | 0.055 |
| Hist binning | Split | 89.7% | 36.0% | 82.0% | 0.031 |
| Hist binning | ACI | 90.5% | 54.0% | 91.4% | 0.039 |
| Tree-based | Split | 86.5% | 34.0% | 73.4% | 0.028 |
| Tree-based | ACI | 88.9% | 48.0% | 91.4% | 0.038 |

Pattern holds across models: Split CP collapses during COVID, ACI partially recovers during crisis and fully recovers post-crisis. **This is the story the proposal is designed to document.** LSTM and TFT not yet benchmarked — run the full grids.

Regime-overview numeric summary (fixed 6-month pad):

| Regime | Ticker | Crisis days | Crisis frac | Max drawdown | Vol ratio (crisis/outside) |
|---|---|---|---|---|---|
| 2008 | AIG | 137 | 35% | -93% | 1.99 |
| 2008 | GNW | 137 | 35% | -94% | 3.26 |
| 2008 | MGM | 137 | 35% | -95% | 1.24 |
| 2020 | MCD | 50 | 16% | -36% | 4.42 |
| 2020 | MGM | 50 | 16% | -78% | 3.29 |
| 2020 | AZO | 50 | 16% | -37% | 4.06 |
| 2022 | SNAP | 251 | 50% | -83% | 1.87 |
| 2022 | NFLX | 251 | 50% | -72% | 2.37 |
| 2022 | META | 251 | 50% | -74% | 1.75 |

All 9 pairs: large drawdowns (≥ 36%) and vol ratios ≥ 1.24 confirm the crisis windows capture real stress.

---

## 9. Gotchas / known issues

- **`datasets/` and `results/` are gitignored** — never commit downloaded CSVs or interval outputs.
- **`BaseConformal` defaults to `confidence_level=0.95`** but Metrics.md uses 0.90. Use `_shared.make_wrapper()` or pass 0.90 explicitly. If you forget, results won't match the paper tables.
- **`AdaptiveConformal.adapt(r)` must be called by the caller** between predictions — `walk_forward()` does this when `is_adaptive=True`. Skipping it silently degrades ACI to plain split CP.
- **Jupyter kernel caches imports.** After editing `_shared.py` / `data.py`, restart the kernel (VS Code: top bar; Jupyter Lab: Kernel menu). Don't just re-run cells.
- **Infinite intervals are a valid signal, not always a bug.** `BaseConformal.conformal_quantile` returns `+inf` when the calibration set is too small to finite-sample-guarantee the requested coverage. See §6.1 for the ACI-specific fix.
- **`download_data` drops the leading NaN via pandas index alignment** (`returns_data = returns_data[1:]` before assigning columns). Works correctly but is non-obvious — don't "clean up" without thinking.
- **MGM appears in both 2008 and 2020 baskets.** The per-(ticker, regime) fit means MGM gets two independent models (different training windows, different calibration, different test).

---

## 10. Open decisions (need human call)

### 10.1 TFT vs transformer encoder

The folder file is `Complex/tft.ipynb` and the proposal names "TFT" (Temporal Fusion Transformer), but `src/models/transformer_encoder.py` is a basic 1-layer transformer encoder + linear head, not a proper TFT. Two ways forward:
- **Rename paper section to "Transformer encoder"** (cheap — just a naming change in the writeup)
- **Implement a proper TFT** using `pytorch-forecasting` or similar (more work; need to decide if it materially changes results)

Decision affects how you describe methods in the paper.

### 10.2 FHS and GARCH baselines

Proposal commits to FHS (Filtered Historical Simulation) and GARCH(1,1) with Student-t as the "baselines" (distinct from CP-wrapped methods). Neither is implemented. These don't fit the Point-Predictor + CP wrapper pattern — they produce intervals directly. Two options:
- Implement them as standalone interval producers; have dedicated notebooks write to `results/` with model name `"fhs"` / `"garch"` and `wrapper` = `"native"` or similar
- Drop them from the paper; note the simpler baselines already cover the "can CP beat classical methods" angle

The `arch` package is already in `requirements.txt` for future GARCH work.

### 10.3 Coverage-overlay figure

`src/analysis/regime_overview.ipynb` plots cumulative return and rolling vol per (ticker, regime). The plan was to later overlay rolling 20-day coverage per model onto those same subplots in `results.ipynb` — lets the reader visually see *coverage collapsing* at the same x-axis location where vol spikes. Not built yet; fold into `results.ipynb` when writing it.

### 10.4 Robustness checks not yet run

`Metrics.md` §Metric 5 specifies a required robustness check: re-run adaptation speed with `threshold=0.80` and `persistence=1`, verify ranking is stable. The `adaptation_metrics` module in `src/metrics/` presumably supports these parameters. Do this once model runs are complete.



---

## 11. What's left (prioritized)

1. **Write `src/analysis/results.ipynb`** — the aggregator. Consumes `load_all_intervals()` + teammate's `src/metrics/*` modules → paper tables + Cov_low-vs-Cov_high scatter + adaptation-speed robustness paragraph.
2. **Run LSTM full grid** — flip `RUN_FULL_GRID=True` in `Complex/lstm.ipynb`, run end-to-end (~15 minutes estimated).
3. **Decide TFT question** (§10.1) then run TFT full grid.
4. **Simple models full grids** — already default `True`, but re-run after any changes to `_shared.py` / models.
5. **FHS + GARCH** — decide per §10.2, implement if keeping.
6. **Write the paper** — once tables/figures are in hand.

---

## 12. Session decision log (2026-04-23)

Chronological record of non-obvious choices made during the session, for context if any get revisited.


2. **Notebook layout: per-model, not per-crisis.** Each model trains once per (ticker, regime) and saves intervals. `results.ipynb` aggregates. Avoids retraining across crises and keeps "model work" separable from "evaluation work".

3. **Split geometry: per-crisis refits, not one global model.** Testing "does CP stay calibrated *into* each crisis" requires fresh pre-crisis fits. One global 20-year model would test long-horizon distributional drift instead.

4. **Added ticker baskets to `CRISIS_WINDOWS` — initially included stable, then dropped to crash-only.** User decision after realizing stable baskets don't advance the proposal's central question and dilute the headline comparison. Stable baskets remain in `data.py` as recoverable controls.

5. **Dropped `TICKERS_2008_FINANCE_RESILIENT`** too for consistency (was a middle ground — financial sector during financial crisis, but didn't fully crash). Clean strict-crash grid is easier to defend.

6. **Adaptation speed onset dates: exogenous, not data-derived.** Extensive discussion of circularity: if onset comes from the stock's own returns, you're scoring models on a yardstick they helped build. Stuck with Lehman / S&P peak / Fed hike.

7. **2008-02-20 onset choice:** picked S&P pre-crash peak over WHO pandemic declaration (2020-03-11) because the market-detected peak gave the cleanest pre/post signal. Flagged as the least-exogenous of the three.

8. **2022 basket swap MSFT/AAPL/ZM → SNAP/NFLX/META** after first empirical check showed MSFT (-32%) / AAPL (-29%) / ZM (-48%) drawdowns with weak vol ratios (1.26–1.51). Ran yfinance comparison over 10 candidates; picked the three with best balance of drawdown AND vol-ratio shift that also had ≥ 5 years pre-2022 data for training.

9. **2022 onset moved from 2022-03-16 (first Fed hike) to 2022-01-03 (S&P peak).** Captures the pre-hike selloff in anticipation of tightening. Tech stocks started falling well before the first actual hike.

10. **Equal-ratio experiment, then reverted.** Tried `pad_multiplier = 1.0` so crisis was always 1/3 of the test window regardless of crisis length. Reverted because 2020's resulting ~50-day post-crisis tail was too short for adaptation-speed recovery detection (need ~25-day minimum to register `persistence=5` of rolling 20-day coverage). Per-regime metrics are reported separately in the paper so cross-regime weighting isn't actually required. Kept the `test_window(onset, end, pad_months)` helper in `_shared.py` for future flexibility.

11. **ACI `alpha` floor fix** (§6.1). `0.001` hardcoded floor was below the finite-sample support `1/(n+1) ≈ 0.0082` for our 121-row calibration sets, causing `inf` intervals during sustained crisis losses. Replaced with `1/(n+1) + 1e-6`.

12. **Added `HistogramBinningModel` and `TreeBasedModel`** to fill out the "simple" bucket in the proposal. Kept deliberately simple — quantile-bin lookup and gradient-boosted trees over lag windows. Both interface-compatible with `BaseConformal`'s `model_class` contract.

13. **Skipped a proper TFT for now.** See §10.1. The existing transformer encoder is adequate as a stand-in; decision deferred to writeup time.

---

## 13. Contact / handoff

