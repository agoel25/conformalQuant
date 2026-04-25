"""Histogram-binning native interval predictor — online variant.

At fit time: fits quantile bins on training data (bin boundaries fixed).
At predict time: looks up the bin for the current lag-window mean and
returns quantiles of the stored returns for that bin.
After each step: adapt(realized) appends the new return to the matched bin
and the global pool, so quantiles widen organically as crisis returns
accumulate — no scheduled refit needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple, override

from src.conformal.base import IntervalPredictor


class HistogramBinningNative(IntervalPredictor):
  """Online histogram-binning: bin boundaries fixed at fit time, stored
  returns per bin grow every step via adapt().

  Prediction uses the (alpha/2, 1-alpha/2) quantiles of accumulated returns
  in the matched bin.  As crisis returns enter the appropriate bins the
  quantiles widen automatically — no refit cadence required.
  """

  def __init__(
    self,
    label_col_name: str,
    num_lags: int = 5,
    confidence_level: float = 0.95,
    n_bins: int = 10,
    min_bin_samples: int = 5,
  ):
    super().__init__(
      label_col_name=label_col_name,
      num_lags=num_lags,
      confidence_level=confidence_level,
    )
    self.n_bins = n_bins
    self.min_bin_samples = min_bin_samples

    self.binner: KBinsDiscretizer | None = None
    self.bin_returns: list[np.ndarray] | None = None
    self._all_returns: np.ndarray | None = None
    self.global_lower: float = 0.0
    self.global_upper: float = 0.0
    self.global_point: float = 0.0
    self._last_bin_id: int = 0  # bin used at last predict — needed by adapt()

  @override
  def fit_and_calibrate(
    self,
    train_data: pd.DataFrame,
    calib_data: pd.DataFrame | None = None,
  ) -> None:
    all_data = pd.concat([train_data, calib_data]) if calib_data is not None else train_data
    values = all_data[self.label_col_name].dropna().values.astype(np.float64)
    if len(values) <= self.num_lags:
      raise ValueError(
        f"HistogramBinningNative: need > {self.num_lags} rows, got {len(values)}"
      )

    windows = np.lib.stride_tricks.sliding_window_view(values, self.num_lags)
    X_mean = windows[:-1].mean(axis=1).reshape(-1, 1)
    y = values[self.num_lags:]

    self.binner = KBinsDiscretizer(
      n_bins=self.n_bins, encode="ordinal",
      strategy="quantile", quantile_method="linear",
    )
    bin_ids = self.binner.fit_transform(X_mean).astype(int).ravel()

    self.bin_returns = []
    for b in range(self.n_bins):
      mask = bin_ids == b
      self.bin_returns.append(y[mask] if mask.any() else np.array([], dtype=np.float64))

    self._all_returns = y.copy()
    self.global_lower = float(np.quantile(y, self.alpha / 2))
    self.global_upper = float(np.quantile(y, 1 - self.alpha / 2))
    self.global_point = float(np.mean(y))

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    assert self.binner is not None

    last_window = data[self.label_col_name].values[-self.num_lags:]
    x = np.array([[float(np.mean(last_window))]])
    bin_id = int(self.binner.transform(x)[0, 0])
    bin_id = max(0, min(bin_id, self.n_bins - 1))
    self._last_bin_id = bin_id  # stored for adapt()

    bin_y = self.bin_returns[bin_id]
    if len(bin_y) >= self.min_bin_samples:
      lower = float(np.quantile(bin_y, self.alpha / 2))
      upper = float(np.quantile(bin_y, 1 - self.alpha / 2))
      point = float(np.mean(bin_y))
    else:
      lower = self.global_lower
      upper = self.global_upper
      point = self.global_point

    return point, lower, upper

  def adapt(self, realized: float) -> None:
    """Append the observed return to the bin matched at the last predict call
    and update global quantiles.  walk_forward calls this automatically."""
    self.bin_returns[self._last_bin_id] = np.append(
      self.bin_returns[self._last_bin_id], realized
    )
    self._all_returns = np.append(self._all_returns, realized)
    self.global_lower = float(np.quantile(self._all_returns, self.alpha / 2))
    self.global_upper = float(np.quantile(self._all_returns, 1 - self.alpha / 2))
    self.global_point = float(np.mean(self._all_returns))
