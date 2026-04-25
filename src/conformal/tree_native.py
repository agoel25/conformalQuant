"""Quantile Regression Forest (Meinshausen 2006) as a native interval predictor.

Reference: Nicolai Meinshausen (2006), "Quantile Regression Forests",
Journal of Machine Learning Research 7, pp. 983-999.

Algorithm (one fit, every quantile readable):
  1. Fit a standard random forest regressor on lag-window features X →
     next-step return y.
  2. Record, for every training point, which leaf it landed in for each
     tree (an n_train × n_trees integer matrix).
  3. At test time, drop the test point through every tree to get its
     n_trees leaf indices.  For each tree, the training points in the
     same leaf get weight 1 / leaf_size; sum across trees and normalise.
  4. Read off any quantile from the *weighted* empirical CDF of the
     training labels.  Lower bound = α/2 quantile, upper bound = 1−α/2
     quantile, point = weighted mean.

This is the user's intuition (~"see which leaves a test point falls
into; aggregate training labels in the same leaves; take percentiles")
formalised — the de facto standard tree-based interval method.

We implement it directly on top of ``sklearn.ensemble.RandomForestRegressor``
to avoid adding a dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, override

from src.conformal.base import IntervalPredictor


class QuantileTreeNative(IntervalPredictor):
  """Quantile Regression Forest native interval predictor.

  Hyperparameter defaults are chosen to be similar in capacity to the
  point-prediction ``TreeBasedModel`` so the comparison with simple_cp
  remains apples-to-apples.
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
  ):
    super().__init__(
      label_col_name=label_col_name,
      num_lags=num_lags,
      confidence_level=confidence_level,
    )
    self.rf = RandomForestRegressor(
      n_estimators=n_estimators,
      max_depth=max_depth,
      min_samples_leaf=min_samples_leaf,
      random_state=random_state,
      n_jobs=-1,
    )
    self._train_y: np.ndarray | None = None
    self._train_leaves: np.ndarray | None = None  # shape (n_train, n_trees)
    self._leaf_sizes: list[dict[int, int]] | None = None  # per-tree {leaf_id: count}

  def _build_features(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    windows = np.lib.stride_tricks.sliding_window_view(values, self.num_lags)
    X = windows[:-1]
    y = values[self.num_lags:]
    return X, y

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
        f"QuantileTreeNative.fit_and_calibrate: need > {self.num_lags + 50} rows, got {len(values)}"
      )

    X, y = self._build_features(values)
    self.rf.fit(X, y)
    self._train_y = y

    # Leaf indices for each training point in each tree: (n_train, n_trees)
    self._train_leaves = self.rf.apply(X)

    # Pre-compute leaf sizes per tree for fast weighting at predict time
    self._leaf_sizes = []
    for t in range(self._train_leaves.shape[1]):
      leaves_t = self._train_leaves[:, t]
      unique, counts = np.unique(leaves_t, return_counts=True)
      self._leaf_sizes.append(dict(zip(unique.tolist(), counts.tolist())))

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    assert self._train_y is not None and self._train_leaves is not None

    x = data[self.label_col_name].values[-self.num_lags:].astype(np.float64).reshape(1, -1)
    test_leaves = self.rf.apply(x).flatten()  # (n_trees,)

    n_train = len(self._train_y)
    n_trees = len(test_leaves)
    weights = np.zeros(n_train, dtype=np.float64)
    for t, leaf_id in enumerate(test_leaves):
      leaf_size = self._leaf_sizes[t].get(int(leaf_id), 0)
      if leaf_size == 0:
        continue
      mask = self._train_leaves[:, t] == leaf_id
      weights[mask] += 1.0 / leaf_size
    weights /= n_trees  # average across trees

    # Weighted empirical CDF
    order = np.argsort(self._train_y)
    sorted_y = self._train_y[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    total_w = cum_w[-1]
    if total_w <= 0:
      # Fallback: should never happen, but be defensive
      lo = float(np.quantile(self._train_y, self.alpha / 2))
      hi = float(np.quantile(self._train_y, 1 - self.alpha / 2))
      pt = float(np.mean(self._train_y))
      return pt, lo, hi
    cum_w /= total_w  # normalise to [0, 1]

    lo_idx = int(np.searchsorted(cum_w, self.alpha / 2))
    hi_idx = int(np.searchsorted(cum_w, 1 - self.alpha / 2))
    lo_idx = min(lo_idx, n_train - 1)
    hi_idx = min(hi_idx, n_train - 1)

    lower = float(sorted_y[lo_idx])
    upper = float(sorted_y[hi_idx])
    point = float(np.average(self._train_y, weights=weights / total_w))

    if lower > upper:
      lower, upper = upper, lower
    return point, lower, upper
