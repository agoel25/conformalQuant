# Simple + Conformal Prediction

A simple point predictor (Linear Auto-Regressive on `num_lags = 5`
lagged log-returns) wrapped with three conformal prediction methods.
Tests whether **just adding a CP layer** to a deliberately weak point
predictor is enough to match the parametric baselines.

## Base predictor

* `linear_ar` — `sklearn.linear_model.LinearRegression` on the last 5
  lagged log-returns.  No regularisation, no covariate engineering.
  Implementation: `src/models/linear_ar.py`.

## Conformal wrappers

### `split` — Split conformal prediction

* Source: Vovk, Gammerman, Shafer (2005); Lei et al. (2018).
* Fit on training data, compute absolute residuals `|y_t - ŷ_t|` on the
  calibration set, store the
  `⌈(n+1)(1−α)⌉ / n`-quantile as a fixed half-width `q̂`.
  Interval = `ŷ_t ± q̂` for every test step.  No online update.
* Implementation: `src/conformal/split_conformal.py`.

### `aci` — Adaptive Conformal Inference (Gibbs & Candès 2021)

* Source: **Gibbs & Candès, "Adaptive Conformal Inference Under
  Distribution Shift", NeurIPS 2021**.
* Identical to split-CP except `α` is updated after every prediction:
  `α_{t+1} = α_t + γ · (1{r_t ∉ [l_t, u_t]} − α)`.
  Reduces α (widens intervals) after a miss; increases α (narrows
  intervals) after a hit.  Default learning rate `γ = 0.01`.
* Implementation: `src/conformal/adaptive_conformal.py`.

### `agaci` — Aggregated Adaptive Conformal Inference (Zaffran et al. 2022)

* Source: **Zaffran, Féron, Goude, Josse, Dieuleveut, "Adaptive
  Conformal Predictions for Time Series", ICML 2022**.
* Run `K` ACI experts in parallel with different `γ` values (default:
  geometric grid from 0.001 to 0.1, K=12).  At each step, aggregate
  their intervals via exponential weights based on per-side pinball
  loss (η = 1.0).  Removes the burden of choosing a single `γ`.
* Implementation: `src/conformal/agaci_conformal.py`.

## Why only these three CP wrappers?

Earlier experiments included `enbpi`, `tcp`, `tcp_rm` as well.

* **TCP / TCP-RM** were dropped from the main grid after a separate
  comparison (see `src/analysis/TCP_Paper/`) showed they lose to a
  properly fitted rolling GARCH; the TCP paper's claim of dominance was
  driven by an unfair (hardcoded-parameter) GARCH baseline.
* **EnbPI** was dropped to keep the grid manageable on CPU; it can be
  re-added in `scripts/run_grid.py` with one line if you want.

`split / aci / agaci` cover the spectrum from "no online adaptation"
(split) → "single-knob adaptation" (aci) → "robust multi-rate
aggregated adaptation" (agaci), so they tell the most informative story
within a small grid.
