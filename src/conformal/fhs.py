"""Filtered Historical Simulation using the ``arch`` library.

  1. Fit a GARCH(1,1) (constant mean, normal innovations) on the
     pre-test training+calibration period (returns × 100, percent units).
  2. Compute standardized residuals η_t = ε_t / σ_t and store the
     (1−α/2) empirical quantile of |η_t|.
  3. At each test step, hand the *fitted* GARCH parameters to
     ``arch_model.fix()`` over the running history to obtain a fresh
     one-step-ahead σ, then form the interval as μ ± q_|η| · σ_{t+1}.

The percent-unit convention (×100, then ÷100 at the boundary) avoids
arch's ``rescale=True`` mode silently changing scales between
``fit()`` and ``fix()`` calls.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from typing import Tuple, override

from src.conformal.base import IntervalPredictor


_SCALE = 100.0


class FHS(IntervalPredictor):

  def __init__(
    self,
    label_col_name: str,
    confidence_level: float = 0.95,
    history_cap: int = 504,
  ):
    super().__init__(label_col_name, num_lags=0, confidence_level=confidence_level)
    self.history_cap = history_cap
    self._params: pd.Series | None = None
    self._mu_frac: float = 0.0
    self._q_eta: float | None = None  # (1-α/2) empirical quantile of |std residuals|

  @override
  def fit_and_calibrate(
    self,
    train_data: pd.DataFrame,
    calib_data: pd.DataFrame | None = None,
  ) -> None:
    series = train_data[self.label_col_name]
    if calib_data is not None:
      series = pd.concat([series, calib_data[self.label_col_name]])
    series_pct = series[np.isfinite(series)].astype(np.float64) * _SCALE
    if len(series_pct) < 100:
      raise RuntimeError(f"FHS: need ≥ 100 rows for fit, got {len(series_pct)}")

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      am = arch_model(series_pct, mean="Constant", vol="Garch", p=1, q=1,
                      dist="normal", rescale=False)
      res = am.fit(disp="off", show_warning=False)

    ab = float(res.params.get("alpha[1]", 0.0)) + float(res.params.get("beta[1]", 0.0))
    if not np.isfinite(ab) or ab >= 0.999:
      raise RuntimeError(f"FHS: non-stationary GARCH fit (α+β={ab:.3f})")

    self._params = res.params.copy()
    self._mu_frac = float(res.params.get("mu", 0.0)) / _SCALE
    eta = np.asarray(res.resid) / np.asarray(res.conditional_volatility)
    self._q_eta = float(np.quantile(np.abs(eta), 1 - self.alpha / 2))

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    assert self._params is not None and self._q_eta is not None
    history_pct = data[self.label_col_name].dropna().iloc[-self.history_cap:] * _SCALE
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      am = arch_model(history_pct, mean="Constant", vol="Garch", p=1, q=1,
                      dist="normal", rescale=False)
      fit_rolling = am.fix(self._params)
      forecast = fit_rolling.forecast(horizon=1, reindex=False)

    sigma_next = float(np.sqrt(forecast.variance.values[-1, 0])) / _SCALE
    margin = self._q_eta * sigma_next
    return self._mu_frac, self._mu_frac - margin, self._mu_frac + margin
