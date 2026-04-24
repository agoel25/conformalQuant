"""EnbPI: Ensemble batch Prediction Intervals for time series (Xu & Xie 2021).

From-scratch implementation of Algorithm 1 in "Conformal prediction interval
for dynamic time-series" (Xu & Xie 2021). Replaces an earlier MAPIE-based stub
that did not integrate cleanly with our single-step `TimeSeriesModel` contract.

Idea: train B bootstrap-resampled predictors and form intervals from their
leave-one-out residuals. At test time the full B-ensemble produces the point
forecast; the interval half-width is the (1-alpha) quantile of a rolling window
of LOO residuals. After each observation is revealed, the oldest residual is
dropped and the new one appended, so the interval tracks local volatility
without re-fitting the base predictors.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Tuple, Type
from typing import override

import numpy as np
import pandas as pd

from src.conformal.base import BaseConformal
from src.models.base_model import TimeSeriesModel

_AGGREGATORS: dict[str, Callable[[np.ndarray], float]] = {
  "mean": lambda a: float(np.mean(a)),
  "median": lambda a: float(np.median(a)),
}


class EnbPIConformal(BaseConformal):
  """Ensemble batch Prediction Intervals (Xu & Xie 2021)."""

  @override
  def __init__(
    self,
    model_class: Type[TimeSeriesModel],
    label_col_name: str,
    num_lags: int = 1,
    confidence_level: float = 0.95,
    B: int = 25,
    block_size: int = 1,
    aggregation: str = "mean",
    window_length: int | None = None,
    random_state: int = 59,
  ):
    """
    :param B: ensemble size
    :param block_size: 1 for i.i.d. bootstrap, >1 for moving-block (better for
                       strongly autocorrelated series)
    :param aggregation: "mean" or "median"
    :param window_length: size of the sliding residual window; defaults to the
                          number of LOO residuals built during calibration
    :param random_state: seed for the bootstrap RNG
    """
    super().__init__(model_class, label_col_name, num_lags, confidence_level)

    if aggregation not in _AGGREGATORS:
      raise ValueError(
        f"aggregation must be one of {sorted(_AGGREGATORS)}, got {aggregation!r}"
      )
    if B < 2:
      raise ValueError(f"B must be >= 2, got {B}")
    if block_size < 1:
      raise ValueError(f"block_size must be >= 1, got {block_size}")

    self._model_class = model_class
    self.B = B
    self.block_size = block_size
    self.aggregation = aggregation
    self.window_length = window_length
    self.random_state = random_state

    self._agg_fn = _AGGREGATORS[aggregation]
    self._rng = np.random.default_rng(random_state)

    # State filled in by fit_and_calibrate
    self.models: list[TimeSeriesModel] = []
    self.bootstrap_index_sets: list[set[int]] = []
    self.residuals_window: deque[float] | None = None
    self.last_prediction: float | None = None

    # The BaseConformal-provided `self.model` is unused (EnbPI maintains its
    # own list of ensemble members in `self.models`); kept for interface
    # compatibility only.

  def _sample_bootstrap(self, T: int) -> np.ndarray:
    if self.block_size == 1:
      return self._rng.integers(0, T, size=T)
    n_blocks = (T + self.block_size - 1) // self.block_size
    starts = self._rng.integers(0, T - self.block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + self.block_size) for s in starts])
    return idx[:T]

  @override
  def fit_and_calibrate(
    self,
    train_data: pd.DataFrame,
    calib_data: pd.DataFrame | None = None,
  ):
    """Train B bootstrap predictors and seed the initial LOO residual window.

    EnbPI does not need a separate calibration slice: residuals are computed on
    all training rows via leave-one-out aggregation. If `calib_data` is provided
    it is concatenated to `train_data` (both are eligible to enter bootstrap
    samples), matching the team's wrapper contract where `fit_and_calibrate`
    receives both slices.
    """
    if calib_data is not None and len(calib_data) > 0:
      combined = pd.concat([train_data, calib_data])
    else:
      combined = train_data.copy()

    T = len(combined)
    if T <= self.num_lags:
      raise ValueError(f"Need > num_lags={self.num_lags} rows; got {T}")

    # 1. Draw B bootstrap samples and fit one base model per sample.
    self.models = []
    self.bootstrap_index_sets = []
    for _ in range(self.B):
      S_b = self._sample_bootstrap(T)
      self.bootstrap_index_sets.append(set(int(i) for i in S_b))
      # Keep time order for time-series models; drop duplicates (a common
      # simplification vs. weighted fitting, which most sklearn/torch models
      # don't uniformly support).
      boot_df = combined.iloc[sorted(np.unique(S_b))]
      model = self._model_class(
        label_col_name=self.label_col_name,
        num_lags=self.num_lags,
      )
      model.fit(boot_df)
      self.models.append(model)

    # 2. Compute leave-one-out residuals on every training row >= num_lags.
    y = combined[self.label_col_name].to_numpy()
    loo_residuals: list[float] = []
    for i in range(self.num_lags, T):
      history = combined.iloc[:i]
      contributing = [
        self.models[b].predict(history)
        for b in range(self.B)
        if i not in self.bootstrap_index_sets[b]
      ]
      if not contributing:
        # Exceptionally rare: every bootstrap happened to include i. Skip.
        continue
      loo_pred = self._agg_fn(np.asarray(contributing, dtype=float))
      r = float(abs(y[i] - loo_pred))
      if not np.isnan(r):
        loo_residuals.append(r)

    if not loo_residuals:
      raise RuntimeError(
        "No leave-one-out residuals produced — increase B or training size."
      )

    maxlen = self.window_length if self.window_length is not None else len(loo_residuals)
    self.residuals_window = deque(loo_residuals, maxlen=maxlen)
    self.residuals = np.asarray(self.residuals_window)
    self.quantile_residuals = self.conformal_quantile(self.residuals, self.alpha)

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    """Aggregate all B predictors; symmetric interval with half-width = current
    sliding quantile of LOO residuals (Xu & Xie 2021 Alg. 1, line 13–15).
    """
    assert len(data) >= self.num_lags
    if self.quantile_residuals is None or self.residuals_window is None:
      raise RuntimeError("EnbPI must be calibrated; call fit_and_calibrate first.")

    preds = np.asarray([m.predict(data) for m in self.models], dtype=float)
    point = self._agg_fn(preds)
    q = float(self.quantile_residuals)
    self.last_prediction = point
    return point, point - q, point + q

  def adapt(self, real_value: float) -> None:
    """Slide the residual window: append |real_value - last_prediction|. The
    deque's `maxlen` auto-drops the oldest entry, implementing Alg. 1 lines
    16–18. Alpha is not updated (EnbPI is not alpha-adaptive).
    """
    if self.last_prediction is None:
      # predict_with_interval has not been called; nothing to adapt from.
      return
    if self.residuals_window is None:
      raise RuntimeError("EnbPI must be calibrated; call fit_and_calibrate first.")

    new_residual = float(abs(real_value - self.last_prediction))
    self.residuals_window.append(new_residual)
    self.residuals = np.asarray(self.residuals_window)
    self.quantile_residuals = self.conformal_quantile(self.residuals, self.alpha)
