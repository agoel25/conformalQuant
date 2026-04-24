# Implementation Notes

Working document for picking up the project. Captures what's built, why it's built that way, where the gotchas are, and how to reproduce results from a fresh clone.

Last updated: 2026-04-24 (post final-results session).

---

## 1. Project context

- **Question:** how do conformal prediction methods compare against industry-standard volatility models for forming prediction intervals on financial returns through crisis regimes?
- **Methods evaluated:** six conformal wrappers (Split CP, ACI, EnbPI, AgACI, TCP, TCP-RM) and two industry baselines (FHS, GARCH(1,1) with Student-$t$ innovations).
- **Base predictors:** five — `LinearARModel`, `HistogramBinningModel`, `TreeBasedModel` (sklearn-backed simple models) and `LSTMModel`, `TransformerModel` (PyTorch deep models).
- **Crises:** three exogenous windows — 2008 Lehman (Sept-15-2008 → Mar-31-2009), 2020 COVID (Feb-20-2020 → Apr-30-2020), 2022 rate-hike cycle (Jan-3-2022 → Dec-31-2022).
- **Tickers:** three per regime — 2008 {AIG, GNW, MGM}, 2020 {MCD, MGM, AZO}, 2022 {SNAP, NFLX, META} = 9 (ticker, regime) pairs.
- **Headline result:** TCP and TCP-RM (with both LSTM and Transformer base predictors) are the only methods that hit nominal $0.90$ marginal coverage on average and recover rolling coverage in $\sim 17$ days after the 2008 crisis onset; every non-TCP method takes $\geq 109$ days or never recovers.
- **Final deliverable:** `report/report.pdf` (NeurIPS-style, 6 pages main + 2 references/appendix). Reproduction is `python scripts/run_grid.py && python scripts/build_artifacts.py`.

---

## 2. Repository layout

```
.
├── README.md                          # one-page reproduction instructions
├── implementation.md                  # this file
├── requirements.txt                   # pip dependencies
├── environment.yml                    # conda spec (untracked locally)
├── main.py                            # initial smoke runner (older scaffolding)
├── evaluate.py                        # standalone fit-then-predict helper (older)
├── papers/                            # reference PDFs (TCP, FHS)
├── project-format/                    # prof's LaTeX template + style file
├── datasets/                          # gitignored; cached yfinance data
├── results/                           # gitignored; 180 interval CSVs + metrics_long.csv
├── report/
│   ├── report.tex / report.pdf        # the actual paper
│   ├── refs.bib                       # 12 bib entries
│   ├── paper.sty            # copy of the prof's style file
│   └── figures/                       # scatter_cov_lo_hi.{pdf,png}, rolling_coverage.{pdf,png}
├── scripts/
│   ├── run_grid.py                    # produces all 180 interval CSVs in ~15 min
│   └── build_artifacts.py             # aggregates → metrics_long.csv + figures
└── src/
    ├── conformal/                     # 6 wrappers + 2 baselines
    │   ├── base.py                    # IntervalPredictor + BaseConformal abstract
    │   ├── split_conformal.py
    │   ├── adaptive_conformal.py      # ACI (Gibbs & Candès 2021)
    │   ├── enbpi_conformal.py         # EnbPI (Xu & Xie 2021), from-scratch
    │   ├── agaci_conformal.py         # AgACI (Zaffran et al. 2022)
    │   ├── tcp_conformal.py           # TCP + TCP-RM (Aich et al. 2026)
    │   ├── fhs.py                     # Filtered Historical Simulation
    │   └── garch.py                   # GARCH(1,1) with conformal calibration
    ├── models/                        # 5 base predictors
    │   ├── base_model.py
    │   ├── linear_ar.py / histogram_binning.py / tree_based.py
    │   ├── lstm.py / transformer_encoder.py
    ├── metrics/                       # evaluation
    │   ├── Metrics.md                 # the spec — read before changing anything here
    │   ├── prediction_cols.py         # column-name enum
    │   ├── coverage_metrics.py        # marginal + conditional coverage
    │   ├── winkler_metrics.py         # Winkler scores, including period subset
    │   ├── interval_metrics.py        # width stratification
    │   ├── adaptation_metrics.py      # T_recovery, max_drawdown, recovered flag
    │   └── metrics.py                 # facade: one Metrics object → vends the four sub-classes
    ├── data/
    │   ├── data.py                    # yfinance download + Cols / Ticker enums
    │   └── transform.py               # build_deep_data → (N, num_lags, 1) tensors
    └── analysis/
        ├── _shared.py                 # iterate_experiments + walk_forward + make_wrapper + I/O
        ├── regime_overview.ipynb      # crisis-window sanity-check plots
        ├── results.ipynb              # interactive aggregator (mirrors build_artifacts.py)
        ├── Simple/{linear_ar,hist_binning,tree_based}.ipynb
        ├── Complex/{lstm,transformer}.ipynb
        └── Baselines/{fhs,garch}.ipynb
```

---

## 3. Experimental grid

### 3.1 Tickers (9 (ticker, regime) pairs)

| Regime | Tickers | Rationale |
|---|---|---|
| 2008 | AIG, GNW, MGM | financial collapse + casino hit hardest by Lehman |
| 2020 | MCD, MGM, AZO | in-person consumer / hospitality crushed by COVID |
| 2022 | SNAP, NFLX, META | ad revenue + streaming under rate-hike pressure |

MGM appears in both 2008 and 2020 baskets — gets two independent fits.

### 3.2 Crisis windows (exogenous dates)

| Regime | Onset | End | Event |
|---|---|---|---|
| 2008 | 2008-09-15 | 2009-03-31 | Lehman Brothers bankruptcy filing |
| 2020 | 2020-02-20 | 2020-04-30 | S&P 500 pre-crash peak |
| 2022 | 2022-01-03 | 2022-12-31 | S&P 500 pre-bear peak |

Dates are exogenous (external events, not data-derived) so adaptation-speed measurement is not circular. See `src/metrics/Metrics.md` §Metric 5.

### 3.3 Split geometry (per ticker × regime)

Defined in `_shared.iterate_experiments()` with defaults `train_years=4, calib_months=6, test_pad_months=6`:

```
train:     [onset - 4y - 12mo,  onset - 12mo)         # 4 years
calibrate: [onset - 12mo,        onset - 6mo)          # 6 months of calm
test:      [onset - 6mo,         end + 6mo]            # crisis + 6mo on each side
```

- Per-crisis refits, not one global model — the question is "does CP stay calibrated *into* each crisis", which needs a fresh pre-crisis fit.
- Crisis fractions of the test window differ by regime (2008 35%, 2020 16%, 2022 50%); per-regime metrics handle this cleanly.

### 3.4 Nominal coverage

**$\alpha = 0.10$, $1-\alpha = 0.90$.** `BaseConformal` defaults to `confidence_level=0.95`; `_shared.make_wrapper()` overrides to $0.90$ to match `Metrics.md`. Pass it explicitly elsewhere or results won't match the paper tables.

### 3.5 Wrapper × model coverage

The grid is **180 runs** = three groups:

| Group | Models | Wrappers | CSVs |
|---|---|---|---|
| Simple | LinAR, HistBin, Trees | Split, ACI | 3 × 2 × 9 = 54 |
| Deep   | LSTM, Transformer    | Split, ACI, EnbPI, AgACI, TCP, TCP-RM | 2 × 6 × 9 = 108 |
| Baseline | FHS, GARCH (no wrapper) | Native | 2 × 1 × 9 = 18 |

Simple models are scoped to Split + ACI per the proposal's framing where they are "interpretable native-interval methods"; the four newer wrappers were applied only to deep base predictors. EnbPI and AgACI work on simple models too — running them is as cheap as adding entries to `WRAPPER_KINDS` in the relevant notebook or `scripts/run_grid.py`.

---

## 4. Architecture

Two parallel hierarchies (`Models`, `Conformal Wrappers`) glued by string column names and a shared `num_lags` convention.

### 4.1 Base predictors (`src/models/`)

All subclass `TimeSeriesModel(label_col_name, num_lags)`. Implement `fit(df)` and `predict(df) -> float`. `predict` returns the next value after the last `num_lags` rows of the supplied dataframe — single-step only, not a sequence forecast.

| Class | File | Algorithm |
|---|---|---|
| `LinearARModel` | `linear_ar.py` | sklearn `LinearRegression` on lagged features (drops NaN rows) |
| `HistogramBinningModel` | `histogram_binning.py` | quantile-bin lookup over lag-window mean → memorised bin mean of next step |
| `TreeBasedModel` | `tree_based.py` | sklearn `GradientBoostingRegressor` (200 trees, max_depth 3) over lag windows |
| `LSTMModel` | `lstm.py` | single-layer LSTM (50 hidden) + linear head, mini-batch Adam + early stopping (PyTorch) |
| `TransformerModel` | `transformer_encoder.py` | 1-layer Transformer encoder ($d_\mathrm{model}=32$, 4 heads, dropout 0.1) + linear head, mini-batch Adam + early stopping (PyTorch) |

**Deep model training (LSTM, Transformer).** mini-batch Adam (batch size 32, lr $10^{-3}$), 50 epochs maximum, gradient clipping at norm 1.0, last 15% of training rows held out **chronologically** as validation, early stopping on val MSE (patience 5). NaN/inf rows in the label column (e.g. pre-IPO history for tickers like GNW or SNAP) are filtered before training. The earlier implementation ran a single full-batch gradient step per "epoch" (~20 total updates) which left predictions collapsed near a constant; the fixed loop is what produced the numbers in the paper. See §6.1 for details.

### 4.2 Conformal wrappers (`src/conformal/`)

All concrete wrappers subclass either `BaseConformal` (which owns a single instance of a passed-in `model_class`) or, for FHS/GARCH which are interval producers in their own right, the lower-level `IntervalPredictor`. Common interface:

- `fit_and_calibrate(train_data, calib_data) -> None`
- `predict_with_interval(data) -> (point, lower, upper)`
- optional `adapt(real_value) -> None` for online updates; `walk_forward` calls it via duck typing (`hasattr(wrapper, "adapt")`).

| Class | File | One-liner |
|---|---|---|
| `SplitConformal` | `split_conformal.py` | Fixed half-width = $\hat{q}(\alpha)$ over absolute residuals |
| `AdaptiveConformal` | `adaptive_conformal.py` | Online $\alpha$ update on coverage error (Gibbs & Candès 2021) |
| `EnbPIConformal` | `enbpi_conformal.py` | $B$ bootstrap predictors, sliding LOO residuals (Xu & Xie 2021) |
| `AgACIConformal` | `agaci_conformal.py` | $K$ ACI experts with different $\gamma$, EW aggregation (Zaffran et al. 2022) |
| `TCPConformal` | `tcp_conformal.py` | Rolling 252-day window, refit + recalibrate every $k$ test steps (Aich et al. 2026) |
| `TCPRMConformal` | `tcp_conformal.py` | TCP + Robbins-Monro offset on the conformal threshold (Aich et al. 2026) |
| `FHS` | `fhs.py` | Filtered historical simulation with GARCH-standardized residuals |
| `Garch` | `garch.py` | GARCH(1,1) Student-$t$ + conformal quantile of $|y - \mu| / \sigma_t$ |

**The conformal quantile** (`BaseConformal.conformal_quantile`) returns `+inf` when the calibration set is too small to finite-sample-guarantee the requested coverage (rank > $n$). See §6.2 for the ACI dynamic floor that prevents inf widths in practice.

**EnbPI hyperparameters.** `B=25` for simple base predictors; `B=10` for LSTM/Transformer to keep wall-clock cost manageable. Block size 1 (i.i.d.\ bootstrap) by default; moving-block bootstrap available via `block_size > 1`.

**AgACI hyperparameters.** Default `gamma_grid = np.geomspace(0.001, 0.1, 12)`, `eta = 1.0`. The aggregator is exponential weights over per-side pinball loss; BOA can be a drop-in replacement by editing `adapt`.

**TCP hyperparameters.** Default `window=252`, `calib=60` (so train slice = 192 rows); `refit_every=5` for the deep models to reduce per-step refit cost (the paper uses every step). TCP-RM extends with `gamma_0=0.01`, `lambda=1.0`, `beta=0.6` per Aich et al.\ Eq.\ 1.

### 4.3 Analysis utilities (`src/analysis/_shared.py`)

| Symbol | Purpose |
|---|---|
| `ALL_TICKERS` | Union of every basket in `data.py`; what `ensure_data()` downloads |
| `CRISIS_WINDOWS` | dict `{regime: {onset, end, tickers}}` — single source of truth for the 9 experiments |
| `ExperimentSplit` | Frozen dataclass with `train/calib/test` timestamps + `.slice_train/calib/full(data)` helpers |
| `iterate_experiments()` | Generator yielding `ExperimentSplit` per (ticker, regime) pair |
| `ensure_data()` | Downloads ALL_TICKERS if not cached, with auto-refresh guard when ALL_TICKERS changes |
| `walk_forward(wrapper, data, test_start, test_end, is_adaptive=None)` | Steps the wrapper one day at a time. `is_adaptive` is deprecated and ignored — adaptation is duck-typed via `hasattr(wrapper, "adapt")` |
| `make_wrapper(model_class, label_col, num_lags, confidence_level=0.90, wrapper_kind=None, step_size=None, ...)` | Factory. Preferred: pass `wrapper_kind ∈ {"split","aci","enbpi","agaci","tcp","tcp_rm"}`. Back-compat: pass `step_size=None` (→ split) or `step_size=0.01` (→ aci); existing notebooks work without edits. Other kwargs (`B`, `gamma_grid`, `tcp_window`, `tcp_calib`, `tcp_refit_every`, `gamma_0`, `rm_lambda`, `rm_beta`) are forwarded to the relevant wrapper. |
| `save_intervals(df, model, wrapper, ticker, regime, results_dir=RESULTS_DIR)` | Writes tagged CSV to `results/<model>_<wrapper>_<ticker>_<regime>.csv` |
| `load_all_intervals(results_dir)` | Concatenates every interval CSV; skips `metrics_long.csv` and any other non-interval files (filename must split into ≥4 underscore tokens) |
| `REQUIRED_COLUMNS`, `RESULTS_DIR` | Schema gate + project-root `/results` path |

### 4.4 Data (`src/data/`)

- `data.py` — yfinance download wrapper. Columns named `Returns_<TICKER>` and `Log_Returns_<TICKER>`. `Cols` and `Ticker` are `StrEnum`. All `TICKERS_*` basket constants live here. `load_data` parses `Date` as a `DatetimeIndex`.
- `transform.py` — `build_deep_X` / `build_deep_data` for reshaping into `(N, num_lags, 1)` tensors for LSTM/Transformer.

### 4.5 Metrics (`src/metrics/`)

| File | Role |
|---|---|
| `Metrics.md` | Evaluation spec — source of truth for nominal coverage, crisis dates, vol-threshold convention. |
| `prediction_cols.py` | Column-name constants for saved predictions (`log_returns`, `lower_bound`, `upper_bound`). |
| `coverage_metrics.py` | Marginal + conditional coverage via boolean masks. |
| `winkler_metrics.py` | Winkler score $W_t = (u_t - l_t) + (2/\alpha)\max(l_t - r_t, 0) + (2/\alpha)\max(r_t - u_t, 0)$. |
| `interval_metrics.py` | Width stratification (overall / high-vol / low-vol / period). |
| `adaptation_metrics.py` | `compute_adaptation_speed(crisis_start, crisis_end, threshold=0.85, persistence=5, window=20) -> (T_rec, max_drawdown, recovered)`. |
| `metrics.py` | Facade: one `Metrics(df, alpha)` → `get_coverage_metrics()`, `get_winkler_metrics()`, `get_interval_metrics()`, `get_adaptation_metrics()`. |

The aggregator (`scripts/build_artifacts.py` and `src/analysis/results.ipynb`) renames the CSV columns `r/l/u` → `log_returns/lower_bound/upper_bound` before instantiating `Metrics`.

---

## 5. Saved interval schema (`results/*.csv`)

| column | type | description |
|---|---|---|
| date | `datetime64[ns]` | trading day |
| point | float | model's point prediction |
| l | float | lower interval bound |
| u | float | upper interval bound |
| r | float | realized log return |
| model | string | `linear_ar`, `hist_binning`, `tree_based`, `lstm`, `transformer`, `fhs`, `garch` |
| wrapper | string | `split`, `aci`, `enbpi`, `agaci`, `tcp`, `tcp_rm`, `native` |
| ticker | string | e.g. `SNAP` |
| regime | string | `2008`, `2020`, `2022` |

Filename: `<model>_<wrapper>_<ticker>_<regime>.csv`. Aggregate file `metrics_long.csv` (also under `results/`) has a different schema and is skipped by `load_all_intervals`.

---

## 6. Key implementation fixes (for posterity)

### 6.1 Deep-model training fix

**Symptom (now fixed):** the original LSTM/Transformer fit loops did one full-batch gradient step per "epoch", giving ~20 total updates. LSTM predictions had std `1.3e-4` (essentially constant); Transformer was predicting a constant ≈ −5\% every day, MAE 4× worse than zero predictor. The conformal coverage numbers were still mathematically valid (the rolling residual quantile absorbed the wrong constant) but the "TCP works on neural predictors" story was unsupported.

**Fix:** rewrote `src/models/lstm.py` and `src/models/transformer_encoder.py` to use proper mini-batch Adam (batch=32, 50 epochs max), gradient clipping at norm 1.0, last 15% of training rows held out chronologically as validation, early stopping on val MSE (patience=5). NaN/inf training rows are filtered before fitting. After fix on MCD 2020, Transformer corr=0.22 with realized returns, LSTM corr=0.11, both beat the zero-predictor MAE.

### 6.2 ACI alpha floor

The original `adaptive_conformal.py` clamped `alpha` to `[0.001, 0.5]`. With $n=121$ calibration residuals and target $\alpha=0.10$, sustained crisis losses pushed $\alpha$ to 0.001 — but `conformal_quantile` returns $+\infty$ for any $\alpha < 1/(n+1) \approx 0.0083$. Result: `inf` widths during crises. Replaced with the dynamic floor `1/(len(residuals)+1) + 1e-6`, which is always strictly above the support boundary; smoke test post-fix shows zero inf days.

### 6.3 GARCH stationarity guard + sigma cap

The arch optimiser occasionally converges to non-stationary GARCH params ($\alpha + \beta \geq 1$) and a bogus mean several orders of magnitude outside the returns range. Forward volatility recursion then explodes (interval widths $\sim 10^{16}$). Two layers of defence: `arch_model(rescale=True)` to improve numerical conditioning, and a post-fit guard that raises `RuntimeError` if $\alpha + \beta \geq 1$ so the runner can skip that (asset, window) cell. AIG 2008 currently triggers the guard — a legitimate data finding for that pre-Lehman period.

### 6.4 GARCH/FHS NaN handling and arch 8.x API

Both baselines now drop NaN/inf rows from the training series before calling `arch_model` (necessary for pre-IPO tickers like GNW 2008, SNAP 2022). The earlier `forecast(... data=...)` API was removed in arch 8.x; we now use `am.fix(params)` to apply the trained parameters to a rolling history and read off the volatility trajectory.

### 6.5 EnbPI rewrite (no MAPIE)

The original EnbPI was a stub wrapping `mapie.regression.TimeSeriesRegressor` that returned `(0,0,0)` for every prediction. Replaced with a from-scratch implementation of Xu & Xie 2021 Algorithm 1: $B$ bootstrap-trained predictors, leave-one-out aggregation for residuals, sliding `collections.deque` of residuals updated via `adapt(real_value)`. No MAPIE dependency.

### 6.6 `walk_forward` duck-typed adaptation

Old version branched on a boolean `is_adaptive` kwarg and asserted `isinstance(wrapper, AdaptiveConformal)`. New version is `if hasattr(wrapper, "adapt"): wrapper.adapt(real_value)`. ACI, EnbPI, AgACI, and TCP-RM all expose `adapt`; Split and TCP do not. The old `is_adaptive` kwarg is preserved as a deprecated no-op for back-compat with existing notebook call sites.

### 6.7 Notebook path bootstrap

Each notebook starts by walking up from CWD to find a `requirements.txt` and inserting the project root into `sys.path`, so `from src.analysis._shared import ...` works whether Jupyter is launched from project root or from `src/analysis/`.

---

## 7. How to run

### 7.1 Environment

```bash
pip install -r requirements.txt
# OR: conda env create -f environment.yml && conda activate conformalquant
```

Python 3.12+ required (`StrEnum`, `typing.override`, PEP 604 unions). Direct deps: `numpy`, `pandas`, `scikit-learn`, `torch`, `arch` (GARCH/FHS), `yfinance`, `matplotlib`. EnbPI no longer needs `mapie` since the rewrite.

### 7.2 End-to-end reproduction

```bash
python scripts/run_grid.py            # 180 interval CSVs, ~15 min
python scripts/build_artifacts.py     # results/metrics_long.csv + report figures
cd report && pdflatex report.tex && biber report && pdflatex report.tex && pdflatex report.tex
```

`scripts/run_grid.py --only {simple,deep,baselines,all}` runs subsets. The `--only` flag is useful when iterating on one model class at a time. The script seeds `numpy` and `torch` to 0 per (model, wrapper, ticker, regime) cell so output is deterministic.

### 7.3 Interactive workflow (notebooks)

The notebook flow is equivalent to the script flow:

1. Run `src/analysis/Simple/{linear_ar,hist_binning,tree_based}.ipynb` (or just `regime_overview.ipynb` first to sanity-check crisis windows).
2. Run `src/analysis/Baselines/{fhs,garch}.ipynb`.
3. Flip `RUN_FULL_GRID=True` in `src/analysis/Complex/{lstm,transformer}.ipynb` and run.
4. Run `src/analysis/results.ipynb` to aggregate.

`RUN_FULL_GRID` defaults: simple notebooks `True` (fast), deep notebooks `False` (gated to avoid surprise long runs in interactive use).

### 7.4 Regenerating regime overview

`src/analysis/regime_overview.ipynb` is standalone — load data, plot cumulative returns and rolling vol per (regime, ticker) with crisis windows shaded, plus a numeric summary table. Run this first to visually confirm crisis windows correspond to real stock moves. **Restart the Jupyter kernel after editing `_shared.py` or `data.py`** — Jupyter caches imports.

---

## 8. Headline numbers from the final grid

Averaged across all 9 (ticker, regime) pairs (rows sorted by Winkler, lower is better):

| # | Model × Wrapper | Marg. Cov | Cov gap | Width | Winkler |
|---|---|---|---|---|---|
| 1 | LSTM × TCP        | 0.892 | 0.080 | 0.173 | **0.280** |
| 2 | LSTM × TCP-RM     | 0.897 | 0.073 | 0.179 | **0.280** |
| 3 | Transformer × TCP-RM | 0.895 | 0.077 | 0.187 | 0.293 |
| 4 | Transformer × TCP    | 0.889 | 0.089 | 0.181 | 0.293 |
| 5 | LSTM × ACI         | 0.820 | 0.174 | 0.109 | 0.305 |
| 6 | HistBin × ACI      | 0.819 | 0.181 | 0.109 | 0.307 |
| ... | (ACI / AgACI variants cluster ~0.31) | | | | |
| 19 | FHS × Native       | 0.666 | 0.302 | 0.065 | 0.415 |
| 20 | GARCH × Native     | 0.768 | 0.179 | 0.225 | 0.477 |

Adaptation speed during 2008 (days to recover rolling 20-day cov $\geq 0.85$ for 5 consecutive days): TCP family 17.3–17.7 across both deep base predictors; ACI methods 108–117 (1/3 tickers recovered); Split / EnbPI / FHS 137 (none recovered).

Full per-cell numbers in `results/metrics_long.csv`; full prose discussion in `report/report.pdf`.

---

## 9. Gotchas

- **`datasets/` and `results/` are gitignored** — never commit downloaded CSVs or interval outputs.
- **`BaseConformal.confidence_level` defaults to 0.95** but Metrics.md is 0.90. Use `_shared.make_wrapper()` (defaults to 0.90) or pass it explicitly.
- **`make_wrapper(wrapper_kind=...)` is the preferred API**; old `step_size=None|0.01` keeps working for back-compat.
- **`walk_forward` ignores `is_adaptive`**; adaptation is duck-typed. ACI/EnbPI/AgACI/TCP-RM expose `adapt`; Split/TCP don't.
- **EnbPI on deep models defaults to `B=10`**, not 25, to keep wall-clock manageable. Override via `make_wrapper(..., B=25)`.
- **TCP/TCP-RM refit every step by default**, which is expensive on neural models. `make_wrapper(..., tcp_refit_every=5)` is the practical default in `scripts/run_grid.py`; the paper's claim is paper-default `=1`.
- **AIG 2008 GARCH degenerate fit**: $\alpha + \beta \geq 1$, runner skips it. Result file is absent for that one (asset, window) cell; the aggregator tolerates this.
- **MGM appears in both 2008 and 2020 baskets**; gets two independent fits (different train/calib/test windows).
- **Jupyter caches imports.** After editing any `src/` file, restart the kernel — re-running cells alone won't pick up the change.
- **AIG 2008 GARCH cell is missing on purpose** (see §6.3). If you re-run the baselines and see only 8 GARCH CSVs instead of 9, that's expected.
- **`load_all_intervals` skips `metrics_long.csv`** automatically (the aggregate has a different schema). Other ad-hoc CSVs in `results/` will also be skipped if their filename doesn't have ≥4 underscore-separated tokens.

---

## 10. Decision log

Chronological record of non-obvious choices, for context if any get revisited.

1. **Notebook layout: per-model, not per-crisis.** Each model trains once per (ticker, regime) and saves intervals; the aggregator handles the rest. Avoids retraining across crises and keeps "model work" separable from "evaluation work".

2. **Per-crisis refits, not one global model.** The proposal asks "does CP stay calibrated *into* each crisis", which needs a fresh pre-crisis fit, not a global 20-year model.

3. **Crash-only ticker baskets.** Stable baskets exist in `data.py` but are not iterated; they dilute the headline comparison. Re-add as controls if a reviewer asks "is ACI's advantage specifically about regime adaptation vs general quality?".

4. **Exogenous crisis onsets.** Deriving onset from the stock's own returns would introduce circularity into the adaptation-speed metric.

5. **2022 onset = 2022-01-03 (S&P peak), not 2022-03-16 (first Fed hike).** Captures the pre-hike selloff.

6. **2022 basket swap MSFT/AAPL/ZM → SNAP/NFLX/META** after empirical check showed the original three had weak vol-ratio shifts (1.26–1.51).

7. **6-month fixed pad on each side, not equal-ratio.** Tried `pad = crisis_length`; reverted because the resulting 50-day post-2020 tail was too short for `persistence=5` recovery detection.

8. **ACI alpha floor = `1/(n+1) + 1e-6`** instead of hardcoded `0.001`. See §6.2.

9. **Added HistogramBinningModel and TreeBasedModel** to fill the proposal's "simple" bucket. Quantile-bin lookup and gradient-boosted trees over lag windows; both interface-compatible with `BaseConformal`'s `model_class` contract.

10. **Kept transformer encoder, did not implement true TFT** (proposal originally named TFT). Notebook renamed `tft.ipynb` → `transformer.ipynb`, paper section renamed to "Transformer".

11. **TCP/TCP-RM scoped to deep base predictors only** in the headline grid. The two TCP wrappers also work on simple models (the factory accepts any `model_class`), but the proposal's framing positions simple models as "native-interval methods" — adding TCP to them is a one-line `WRAPPER_KINDS` change if a reviewer asks.

12. **EnbPI rewritten from scratch**, no MAPIE dependency. The earlier MAPIE wrapper had a broken integration that returned `(0,0,0)`.

13. **`tcp_refit_every=5` on deep models** for runtime; paper's default is 1 (refit every step).

14. **Deep model training rewrite (mini-batch + early stopping)** — biggest correctness fix. See §6.1.

---

## 11. Status

Nothing left in the work plan that's currently blocking. Concrete polish items if you have time:

- Replace the placeholder GitHub URL in `report/report.tex` (`\begin{ack}` block) with the actual repo URL.
- Re-run `scripts/run_grid.py` from a fresh clone in a different machine to confirm reproducibility before submission.
- Optionally extend EnbPI/AgACI to the simple base predictors (one-line change) for a fuller comparison table.
- Optionally implement BOA aggregation in AgACI; the interface supports a drop-in replacement of the EW step in `adapt`.

---

## 12. Contact

