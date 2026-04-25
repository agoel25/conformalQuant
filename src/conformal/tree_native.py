"""Quantile Regression Forest — online variant.

At fit time: fits a random forest on train+calib lag-window features.
At predict time: uses Meinshausen (2006) leaf-weighting to read quantiles
from the accumulated training labels.
After each step: adapt(realized) adds the new (feature, return) pair to
a growing buffer and refits the forest every refit_every steps on the
full expanding dataset (initial train+calib + all observed test returns).

Reference: Meinshausen (2006), "Quantile Regression Forests", JMLR 7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, override

from src.conformal.base import IntervalPredictor


class QuantileTreeNative(IntervalPredictor):
  """Online Quantile Regression Forest.

  The forest is re-fitted every refit_every steps on the expanding dataset
  (initial train+calib data plus all test returns observed so far).
  Between refits, predictions use the current forest and the accumulated
  training labels — so quantiles drift toward recent data each refit.
  """

  def __init__(
    self,
    label_col_name: str,
    num_lags: int = 5,
    confidence_level: float = 0.95,
    n_estimators: int = 200,
    max_depth: int = 6,
    min_samples_leaf: int = 10,
    random_state: int = 0,
    refit_every: int = 21,
  ):
    super().__init__(
      label_col_name=label_col_name,
      num_lags=num_lags,
      confidence_level=confidence_level,
    )
    self.refit_every = refit_every
    self.rf = RandomForestRegressor(
      n_estimators=n_estimators,
      max_depth=max_depth,
      min_samples_leaf=min_samples_leaf,
      random_state=random_state,
      n_jobs=-1,
    )
    self._train_y: np.ndarray | None = None
    self._train_leaves: np.ndarray | None = None
    self._leaf_sizes: list[dict[int, int]] | None = None

    # Online buffer
    self._init_X: np.ndarray | None = None   # feature matrix from fit_and_calibrate
    self._init_y: np.ndarray | None = None   # labels from fit_and_calibrate
    self._buf_X: list[np.ndarray] = []       # feature rows added via adapt()
    self._buf_y: list[float] = []            # labels added via adapt()
    self._last_x: np.ndarray | None = None   # lag window used at last predict — for adapt()
    self._steps_since_refit: int = 0

  def _build_features(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    windows = np.lib.stride_tricks.sliding_window_view(values, self.num_lags)
    return windows[:-1], values[self.num_lags:]

  def _fit_forest(self, X: np.ndarray, y: np.ndarray) -> None:
    self.rf.fit(X, y)
    self._train_y = y
    self._train_leaves = self.rf.apply(X)
    self._leaf_sizes = []
    for t in range(self._train_leaves.shape[1]):
      leaves_t = self._train_leaves[:, t]
      unique, counts = np.unique(leaves_t, return_counts=True)
      self._leaf_sizes.append(dict(zip(unique.tolist(), counts.tolist())))

  @override
  def fit_and_calibrate(
    self,
    train_data: pd.DataFrame,
    calib_data: pd.DataFrame | None = None,
  ) -> None:
    all_data = pd.concat([train_data, calib_data]) if calib_data is not None else train_data
    values = all_data[self.label_col_name].dropna().values.astype(np.float64)
    if len(values) <= self.num_lags + 50:
      raise ValueError(
        f"QuantileTreeNative: need > {self.num_lags + 50} rows, got {len(values)}"
      )

    X, y = self._build_features(values)
    self._init_X = X.copy()
    self._init_y = y.copy()
    self._buf_X = []
    self._buf_y = []
    self._steps_since_refit = 0
    self._fit_forest(X, y)

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    assert self._train_y is not None and self._train_leaves is not None

    x = data[self.label_col_name].values[-self.num_lags:].astype(np.float64).reshape(1, -1)
    self._last_x = x.copy()  # stored for adapt()

    test_leaves = self.rf.apply(x).flatten()
    n_train = len(self._train_y)
    n_trees = len(test_leaves)

    mask = self._train_leaves == test_leaves[np.newaxis, :]
    leaf_sizes_for_test = mask.sum(axis=0)
    safe_sizes = np.where(leaf_sizes_for_test > 0, leaf_sizes_for_test, 1)
    weights = (mask / safe_sizes[np.newaxis, :]).sum(axis=1) / n_trees

    order = np.argsort(self._train_y)
    sorted_y = self._train_y[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    total_w = cum_w[-1]
    if total_w <= 0:
      lo = float(np.quantile(self._train_y, self.alpha / 2))
      hi = float(np.quantile(self._train_y, 1 - self.alpha / 2))
      return float(np.mean(self._train_y)), lo, hi
    cum_w /= total_w

    lo_idx = min(int(np.searchsorted(cum_w, self.alpha / 2)), n_train - 1)
    hi_idx = min(int(np.searchsorted(cum_w, 1 - self.alpha / 2)), n_train - 1)

    lower = float(sorted_y[lo_idx])
    upper = float(sorted_y[hi_idx])
    if lower > upper:
      lower, upper = upper, lower
    point = float(np.average(self._train_y, weights=weights / total_w))
    return point, lower, upper

  def adapt(self, realized: float) -> None:
    """Add the observed return to the buffer; refit forest every refit_every steps.
    walk_forward calls this automatically."""
    if self._last_x is not None:
      self._buf_X.append(self._last_x.flatten())
      self._buf_y.append(realized)

    self._steps_since_refit += 1
    if self._steps_since_refit >= self.refit_every:
      all_X = (np.vstack([self._init_X] + [x.reshape(1, -1) for x in self._buf_X])
               if self._buf_X else self._init_X)
      all_y = (np.concatenate([self._init_y, self._buf_y])
               if self._buf_y else self._init_y)
      self._fit_forest(all_X, all_y)
      self._steps_since_refit = 0
