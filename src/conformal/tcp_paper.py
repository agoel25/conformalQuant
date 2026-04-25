"""TCP exactly as implemented in Aich et al. (2026), for sanity-check use only.

The paper's TCP uses two gradient-boosted quantile regressors (lower at alpha/2,
upper at 1-alpha/2) as the base model, with a rolling split-conformal calibration
layer on top.  Nonconformity score: max(lower(x)-r, r-upper(x), 0).
Interval: [lower(x) - C_t, upper(x) + C_t].

This is NOT the same as TCPConformal in tcp_conformal.py, which wraps a generic
point-predictor.  This class is faithful to the paper's specific base model choice.

Intended use: run alongside GarchHardcoded (the paper's GARCH baseline) and
RollingGarch to confirm the ordering:
    GarchHardcoded  >  RollingGarch  ~  TCPPaper
in Winkler score (lower = better), showing that a properly estimated rolling
GARCH closes most of the gap the TCP paper attributed to their method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from typing import Tuple, override

from src.conformal.base import IntervalPredictor


class TCPPaperConformal(IntervalPredictor):
  """Paper-faithful TCP: quantile GBM base + rolling split-conformal calibration.

  Matches Aich et al. 2026 Algorithm 1:
    - window=252-day rolling window split into train (window-calib) + calib rows
    - Fits lower/upper quantile GBM regressors on the training slice
    - Conformal threshold C_t = s_{(k)}, s_i = max(lower(x_i)-r_i, r_i-upper(x_i), 0)
    - Interval: [lower(x_{t+1}) - C_t, upper(x_{t+1}) + C_t]
    - Refits every refit_every steps (paper refits every step; we use shared REFIT_EVERY
      cadence so the comparison with rolling GARCH is apples-to-apples)
  """

  def __init__(
    self,
    label_col_name: str,
    num_lags: int = 5,
    confidence_level: float = 0.95,
    window: int = 252,
    calib: int = 60,
    refit_every: int = 21,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.05,
  ):
    super().__init__(
      label_col_name=label_col_name,
      num_lags=num_lags,
      confidence_level=confidence_level,
    )
    if calib <= 0 or calib >= window:
      raise ValueError(f"Need 0 < calib < window; got calib={calib}, window={window}")

    self.window = window
    self.calib = calib
    self.refit_every = refit_every

    _gbm = dict(n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, random_state=0)
    self._lower_model = GradientBoostingRegressor(loss="quantile", alpha=self.alpha / 2, **_gbm)
    self._upper_model = GradientBoostingRegressor(loss="quantile", alpha=1 - self.alpha / 2, **_gbm)
    self._C_t: float | None = None
    self._steps_since_refit: int = refit_every  # trigger refit on first call

  def _build_features(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lag-window feature matrix and next-step targets from a 1-D values array."""
    windows = np.lib.stride_tricks.sliding_window_view(values, self.num_lags)
    X = windows[:-1]
    y = values[self.num_lags:]
    return X, y

  def _refit(self, series: pd.Series) -> None:
    """Refit quantile models and recompute C_t on trailing window."""
    window_data = series.iloc[-self.window:] if len(series) >= self.window else series
    values = window_data.dropna().values.astype(np.float64)

    if len(values) <= self.num_lags + self.calib:
      raise RuntimeError("TCPPaper: not enough history for refit.")

    # Index in the feature array where calibration begins
    train_feat_end = len(values) - self.calib - self.num_lags

    X_all, y_all = self._build_features(values)
    X_train, y_train = X_all[:train_feat_end], y_all[:train_feat_end]
    X_calib, y_calib = X_all[train_feat_end:], y_all[train_feat_end:]

    if len(X_train) == 0 or len(X_calib) == 0:
      raise RuntimeError("TCPPaper: empty train or calib slice after feature construction.")

    self._lower_model.fit(X_train, y_train)
    self._upper_model.fit(X_train, y_train)

    lo_pred = self._lower_model.predict(X_calib)
    hi_pred = self._upper_model.predict(X_calib)
    # Two-sided nonconformity (Aich et al. eq. in §3)
    scores = np.maximum(np.maximum(lo_pred - y_calib, y_calib - hi_pred), 0.0)

    m = len(scores)
    k = int(np.ceil((m + 1) * (1 - self.alpha))) - 1  # convert to 0-index
    k = int(np.clip(k, 0, m - 1))
    self._C_t = float(np.sort(scores)[k])

  @override
  def fit_and_calibrate(
    self,
    train_data: pd.DataFrame,
    calib_data: pd.DataFrame | None = None,
  ) -> None:
    combined = pd.concat([train_data, calib_data]) if calib_data is not None else train_data
    self._refit(combined[self.label_col_name])
    self._steps_since_refit = 0

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    assert self._C_t is not None, "Must call fit_and_calibrate first."

    series = data[self.label_col_name]

    if self._steps_since_refit >= self.refit_every:
      try:
        self._refit(series)
      except RuntimeError:
        pass  # keep previous models if refit fails
      self._steps_since_refit = 0
    self._steps_since_refit += 1

    x = series.dropna().values[-self.num_lags:].reshape(1, -1)
    lo = float(self._lower_model.predict(x)[0])
    hi = float(self._upper_model.predict(x)[0])
    if lo > hi:
      lo, hi = hi, lo  # quantile crossing

    point = (lo + hi) / 2.0
    return point, lo - self._C_t, hi + self._C_t
