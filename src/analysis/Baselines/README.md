# Baselines — industry-standard volatility-based interval methods

These two methods produce prediction intervals natively from a fitted
volatility model — no conformal wrapper.  They are the comparison
target for every method in `Simple_Native`, `Simple_CP`, and
`Complex_CP`.

Both implementations delegate to the
[`arch`](https://arch.readthedocs.io/) library so the GARCH recursion,
optimisation, and one-step-ahead forecasts come from the same code
quant practitioners actually use.

## `garch_roll` — Rolling GARCH(1,1) with Student-t innovations

*Source*: Bollerslev (1986); Student-t innovations follow the standard
econometrics practice for fat-tailed financial returns (Nelson 1991).

* `mean='Constant'`, `vol='Garch'`, `p=q=1`, `dist='studentst'`
* Re-estimated on a trailing 504-day window (≈ 2 calendar years) every
  `REFIT_EVERY = 21` predict steps (≈ monthly).
* Between refits, the *current* parameters are applied via
  `arch_model.fix(params)` to the running history and the one-step-ahead
  variance is read off `forecast(horizon=1)`.
* Interval: `μ ± z_ν · σ_{t+1}` where `z_ν` is the parametric
  (1−α/2)-quantile of the fitted Student-t with `ν` degrees of freedom.
* **No additional conformal q_hat layer** — adding one on top of a
  properly calibrated parametric model just inflates widths.

This is the "fair industry baseline" the TCP paper should have used.
Implementation: `src/conformal/garch.py` :: `RollingGarch`.

## `fhs` — Filtered Historical Simulation

*Source*: Barone-Adesi, Giannopoulos & Vosper (1999), "VaR without
Correlations for Portfolios of Derivative Securities".

* Fit a GARCH(1,1) (constant mean, normal innovations) on
  train+calibration.
* Compute standardised residuals η_t = ε_t / σ_t and store the
  empirical (1−α/2) quantile of |η_t|.
* At each test step, hand the fitted parameters to
  `arch_model.fix(params)` over the running history to get a fresh
  σ_{t+1}.  Interval = `μ ± q_|η| · σ_{t+1}`.

FHS is a strong, simple baseline that combines parametric volatility
filtering with non-parametric tail handling. Fantazzini (2024) found it
struggles in crypto due to GARCH convergence failures; in equities it
typically performs very well.  Implementation: `src/conformal/fhs.py`.

## Why no `garch_static` (MLE-fit, frozen parameters)?

A frozen-parameter GARCH always loses to its rolling counterpart over a
multi-year window containing regime changes.  Including it as a baseline
would muddy the headline comparison; we measure the true cost of
freezing parameters explicitly via the `tcp_paper` standalone sanity
notebook (`src/analysis/TCP_Paper/`).
