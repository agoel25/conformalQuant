"""Histogram-binning native interval predictor.

For each test point, finds the quantile bin that the current lag-window mean
falls in (bins fitted on training data), then returns the alpha/2 and
1-alpha/2 empirical quantiles of the training returns that fell in that bin.
No conformal wrapping — interval width comes entirely from the within-bin
return distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple, override

from src.conformal.base import IntervalPredictor


class HistogramBinningNative(IntervalPredictor):
  """Native prediction intervals via histogram-binning of lag features.

  At fit time: discretises the mean of the last `num_lags` returns into
  `n_bins` quantile bins and stores every training return associated with
  each bin.  At predict time: looks up the bin for the current lag-window
  mean and returns the (alpha/2, 1-alpha/2) quantile pair of stored returns.

  Falls back to global quantiles when the matching bin has fewer than
  `min_bin_samples` observations.
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
    self.global_lower: float = 0.0
    self.global_upper: float = 0.0
    self.global_point: float = 0.0

  @override
  def fit_and_calibrate(
    self,
    train_data: pd.DataFrame,
    calib_data: pd.DataFrame | None = None,
  ) -> None:
    # Combine train and calib — both are pre-test so more data = better bin estimates.
    all_data = pd.concat([train_data, calib_data]) if calib_data is not None else train_data
    values = all_data[self.label_col_name].dropna().values.astype(np.float64)
    if len(values) <= self.num_lags:
      raise ValueError(
        f"HistogramBinningNative.fit_and_calibrate: need > num_lags={self.num_lags} rows, "
        f"got {len(values)}"
      )

    windows = np.lib.stride_tricks.sliding_window_view(values, self.num_lags)
    X_mean = windows[:-1].mean(axis=1).reshape(-1, 1)
    y = values[self.num_lags:]

    self.binner = KBinsDiscretizer(
      n_bins=self.n_bins,
      encode="ordinal",
      strategy="quantile",
      quantile_method="linear",
    )
    bin_ids = self.binner.fit_transform(X_mean).astype(int).ravel()

    self.bin_returns = []
    for b in range(self.n_bins):
      mask = bin_ids == b
      self.bin_returns.append(y[mask] if mask.any() else np.array([], dtype=np.float64))

    self.global_lower = float(np.quantile(y, self.alpha / 2))
    self.global_upper = float(np.quantile(y, 1 - self.alpha / 2))
    self.global_point = float(np.mean(y))

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    assert self.binner is not None, "Must call fit_and_calibrate first."

    last_window = data[self.label_col_name].values[-self.num_lags:]
    x = np.array([[float(np.mean(last_window))]])
    bin_id = int(self.binner.transform(x)[0, 0])
    bin_id = max(0, min(bin_id, self.n_bins - 1))

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
