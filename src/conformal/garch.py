"""GARCH-based interval predictors built on the ``arch`` library.

This file delegates the entire GARCH recursion to ``arch.arch_model``
and uses ``fix()`` / ``fit()`` / ``forecast()`` so the dynamics come
from the same code that every quant practitioner relies on.

Scaling convention
------------------
arch handles small numbers poorly (log returns are O(0.01), and the
optimiser struggles).  We multiply returns by 100 (convert to "percent"
units) before passing to arch, and divide output sigmas by 100 to come
back to fractional log-return units.  ``rescale=False`` is used so arch
does not silently apply a second rescale on top.

Two concrete classes:

  GarchHardcoded
      The TCP paper's GARCH baseline (Aich et al. 2026 eq. 5–6):
      hardcoded (ω=1e-6, α=0.05, β=0.9) on fractional returns,
      equivalently (ω=0.01, α=0.05, β=0.9) on percent returns; zero
      mean, normal innovations.  Implemented via
      ``arch_model.fix()`` so the recursion comes straight from the
      library — we are not re-implementing GARCH ourselves.

  RollingGarch
      Properly fitted GARCH(1,1) with constant mean, Student-t
      innovations, re-estimated on a trailing window every
      ``refit_every`` steps via ``arch_model.fit()``.  Intervals use
      the parametric Student-t quantile of the fitted distribution
      (no additional conformal layer — adding a conformal q_hat to a
      properly calibrated parametric model just inflates widths).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm as _norm
from scipy.stats import t as _student_t
from typing import Tuple, override

from src.conformal.base import IntervalPredictor


# Conversion: fractional log-returns × _SCALE  →  percent units used by arch.
_SCALE = 100.0


# ---------------------------------------------------------------------------
# 1. Hardcoded-parameter GARCH — exactly the TCP paper's baseline (eq. 5–6)
# ---------------------------------------------------------------------------

class GarchHardcoded(IntervalPredictor):
  """GARCH(1,1) with hardcoded parameters as specified by Aich et al. 2026 eq. 5-6.

      σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}     (fractional returns: ω=1e-6, α=0.05, β=0.9)
      Interval [-z_{1-α/2}·σ_t, +z_{1-α/2}·σ_t]   (zero mean, normal)

  In percent units (returns × 100) the equivalent parameters are
  (ω' = 0.01, α=0.05, β=0.9), which is what we hand to arch.
  """

  # Params in PERCENT units (arch will operate on returns × 100).
  _PARAMS_PCT = pd.Series({
    "omega": 1e-6 * (_SCALE ** 2),  # = 0.01
    "alpha[1]": 0.05,
    "beta[1]": 0.9,
  })

  def __init__(
    self,
    label_col_name: str,
    num_lags: int = 1,
    confidence_level: float = 0.95,
    history_cap: int = 504,
  ):
    super().__init__(
      label_col_name=label_col_name,
      num_lags=num_lags,
      confidence_level=confidence_level,
    )
    self._z = float(_norm.ppf(1 - self.alpha / 2))
    self._fitted = False
    self.history_cap = history_cap  # GARCH(1,1) state converges quickly

  @override
  def fit_and_calibrate(
    self,
    train_data: pd.DataFrame,
    calib_data: pd.DataFrame | None = None,
  ) -> None:
    all_data = pd.concat([train_data, calib_data]) if calib_data is not None else train_data
    values = all_data[self.label_col_name].dropna().values
    if len(values) < 50:
      raise RuntimeError(f"GarchHardcoded: need ≥ 50 warm-up rows, got {len(values)}")
    self._fitted = True

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    assert self._fitted, "Must call fit_and_calibrate before predict."
    history = data[self.label_col_name].dropna().values.astype(np.float64)
    if len(history) < 50:
      raise RuntimeError("GarchHardcoded: insufficient history at predict time.")
    history_pct = history[-self.history_cap:] * _SCALE

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      am = arch_model(history_pct, mean="Zero", vol="Garch", p=1, q=1,
                      dist="normal", rescale=False)
      result = am.fix(self._PARAMS_PCT)
      forecast = result.forecast(horizon=1, reindex=False)

    sigma_next_pct = float(np.sqrt(forecast.variance.values[-1, 0]))
    sigma_next = sigma_next_pct / _SCALE  # back to fractional log-return scale
    half_width = self._z * sigma_next
    return 0.0, -half_width, half_width


# ---------------------------------------------------------------------------
# 2. Rolling-fit GARCH — the fair industry-standard baseline
# ---------------------------------------------------------------------------

class RollingGarch(IntervalPredictor):
  """Constant-mean GARCH(1,1) with Student-t innovations, rolling re-estimation.

  At fit time and every ``refit_every`` predict calls:  run
  ``arch_model.fit()`` on a trailing ``garch_window`` of percent-scaled
  data and store the parameters.

  At every predict step:  hand the stored parameters to
  ``arch_model.fix()`` over the latest history and read the
  one-step-ahead variance via ``forecast(horizon=1)``.

  Intervals use the parametric (1−α/2) quantile of the fitted Student-t
  distribution, i.e.  ``μ ± z_ν · σ_{t+1}``.
  """

  def __init__(
    self,
    label_col_name: str,
    num_lags: int = 1,
    confidence_level: float = 0.95,
    garch_window: int = 504,
    refit_every: int = 21,
  ):
    super().__init__(
      label_col_name=label_col_name,
      num_lags=num_lags,
      confidence_level=confidence_level,
    )
    self.garch_window = garch_window
    self.refit_every = refit_every

    self._params: pd.Series | None = None  # in PERCENT-scale units
    self._mu_frac: float = 0.0             # in fractional units (for return value)
    self._z_param: float | None = None     # parametric Student-t quantile
    self._steps_since_refit: int = refit_every

  def _quantile_from_nu(self, nu: float) -> float:
    nu = max(float(nu), 2.5)  # avoid blow-up at very heavy tails
    return float(_student_t.ppf(1 - self.alpha / 2, df=nu))

  def _refit(self, series: pd.Series) -> bool:
    """Refit GARCH on trailing garch_window rows.  Returns False on failure."""
    window_pct = series.dropna().iloc[-self.garch_window:] * _SCALE
    if len(window_pct) < 100:
      return False
    try:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(window_pct, mean="Constant", vol="Garch", p=1, q=1,
                        dist="studentst", rescale=False)
        res = am.fit(disp="off", show_warning=False)
    except Exception:
      return False

    p = res.params
    ab = float(p.get("alpha[1]", 0.0)) + float(p.get("beta[1]", 0.0))
    if not np.isfinite(ab) or ab >= 0.999:
      return False

    self._params = p.copy()
    self._mu_frac = float(p.get("mu", 0.0)) / _SCALE
    self._z_param = self._quantile_from_nu(p.get("nu", 8.0))
    return True

  @override
  def fit_and_calibrate(
    self,
    train_data: pd.DataFrame,
    calib_data: pd.DataFrame | None = None,
  ) -> None:
    all_data = pd.concat([train_data, calib_data]) if calib_data is not None else train_data
    if not self._refit(all_data[self.label_col_name]):
      raise RuntimeError("RollingGarch: initial fit failed (insufficient data or non-stationary).")
    self._steps_since_refit = 0

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    assert self._params is not None, "Must call fit_and_calibrate first."

    series = data[self.label_col_name]

    if self._steps_since_refit >= self.refit_every:
      self._refit(series)  # best-effort; keep previous params on failure
      self._steps_since_refit = 0
    self._steps_since_refit += 1

    history_pct = series.dropna().iloc[-self.garch_window:] * _SCALE
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      am = arch_model(history_pct, mean="Constant", vol="Garch", p=1, q=1,
                      dist="studentst", rescale=False)
      fit_rolling = am.fix(self._params)
      forecast = fit_rolling.forecast(horizon=1, reindex=False)

    sigma_next_pct = float(np.sqrt(forecast.variance.values[-1, 0]))
    sigma_next = sigma_next_pct / _SCALE
    margin = self._z_param * sigma_next
    return self._mu_frac, self._mu_frac - margin, self._mu_frac + margin


# ---------------------------------------------------------------------------
# 3. Legacy static GARCH — kept for back-compat, NOT used in the main grid.
# ---------------------------------------------------------------------------

class Garch(IntervalPredictor):
  """One-shot MLE-fit GARCH with conformal q_hat.  Deprecated; use RollingGarch."""

  def __init__(self, label_col_name, num_lags=1, confidence_level=0.95):
    super().__init__(label_col_name=label_col_name, num_lags=num_lags,
                     confidence_level=confidence_level)
    self._params = None
    self._mu_frac = 0.0
    self.q_hat = None

  @override
  def fit_and_calibrate(self, train_data, calib_data=None):
    if calib_data is None:
      raise ValueError("Calibration data required.")
    train_pct = train_data[self.label_col_name].dropna() * _SCALE
    am = arch_model(train_pct, mean="Constant", vol="Garch", p=1, q=1,
                    dist="studentst", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    ab = float(res.params["alpha[1]"]) + float(res.params["beta[1]"])
    if ab >= 1.0:
      raise RuntimeError(f"non-stationary fit (α+β={ab:.3f})")
    self._params = res.params.copy()
    self._mu_frac = float(res.params["mu"]) / _SCALE

    full_pct = pd.concat([train_data[self.label_col_name], calib_data[self.label_col_name]]).dropna() * _SCALE
    am_full = arch_model(full_pct, mean="Constant", vol="Garch", p=1, q=1,
                         dist="studentst", rescale=False)
    fit_full = am_full.fix(self._params)
    vol_pct = np.asarray(fit_full.conditional_volatility)
    calib_n = len(calib_data[self.label_col_name].dropna())
    calib_vol = vol_pct[-calib_n:] / _SCALE
    actuals = calib_data[self.label_col_name].dropna().values
    resids = np.abs(actuals - self._mu_frac)
    self.q_hat = float(np.quantile(resids / calib_vol, 1 - self.alpha))

  @override
  def predict_with_interval(self, data):
    history_pct = data[self.label_col_name].dropna().values * _SCALE
    am = arch_model(history_pct, mean="Constant", vol="Garch", p=1, q=1,
                    dist="studentst", rescale=False)
    fit_rolling = am.fix(self._params)
    forecast = fit_rolling.forecast(horizon=1, reindex=False)
    sigma_next = float(np.sqrt(forecast.variance.values[-1, 0])) / _SCALE
    margin = self.q_hat * sigma_next
    return self._mu_frac, self._mu_frac - margin, self._mu_frac + margin
