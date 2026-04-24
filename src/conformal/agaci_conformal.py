"""AgACI: Adaptive Conformal Prediction with Expert Aggregation (Zaffran et al. 2022).

Implements §4 of "Adaptive Conformal Predictions for Time Series" — runs K ACI
instances with different learning rates gamma_k in parallel and aggregates
their intervals via online expert weighting. The paper's "FACI" label in our
proposal refers to this family; the method itself is called AgACI.

Two-side aggregation: the lower and upper bounds each have their own weight
vector driven by the corresponding side's pinball loss, so the final interval
can be asymmetric even when each expert produces a symmetric half-width.

First-pass weighting is exponential (EW); swapping in BOA (Wintenberger 2017)
is a drop-in replacement for the two softmax lines in `adapt` — the interface
and state shape are unchanged.
"""
from __future__ import annotations

from typing import Tuple, Type
from typing import override

import numpy as np
import pandas as pd

from src.conformal.base import BaseConformal
from src.models.base_model import TimeSeriesModel


def _pinball(tau: float, y: float, q: float) -> float:
  """Pinball (quantile) loss of estimate q at level tau for observation y."""
  u = y - q
  return float(max(tau * u, (tau - 1.0) * u))


def _softmax_neg(x: np.ndarray, temperature: float) -> np.ndarray:
  """Numerically stable softmax of -temperature * x."""
  v = -temperature * np.asarray(x, dtype=float)
  v = v - np.max(v)
  w = np.exp(v)
  return w / np.sum(w)


class AgACIConformal(BaseConformal):
  """Adaptive conformal with online expert aggregation over K learning rates."""

  @override
  def __init__(
    self,
    model_class: Type[TimeSeriesModel],
    label_col_name: str,
    num_lags: int = 1,
    confidence_level: float = 0.95,
    gamma_grid: np.ndarray | None = None,
    eta: float = 1.0,
  ):
    """
    :param gamma_grid: K learning rates to run in parallel. Default: 12 values
                       on geomspace(0.001, 0.1), matching the paper.
    :param eta: inverse-temperature of the exponential-weights aggregator.
    """
    super().__init__(model_class, label_col_name, num_lags, confidence_level)

    self.gamma_grid = (
      np.asarray(gamma_grid, dtype=float)
      if gamma_grid is not None
      else np.geomspace(0.001, 0.1, 12)
    )
    if len(self.gamma_grid) < 2:
      raise ValueError("gamma_grid must have at least 2 entries.")
    self.eta = float(eta)

    K = len(self.gamma_grid)
    self.K = K
    self.initial_alpha = 1.0 - confidence_level

    self.alpha_k = np.full(K, self.initial_alpha, dtype=float)
    self.q_k = np.zeros(K, dtype=float)
    self.w_lower = np.full(K, 1.0 / K, dtype=float)
    self.w_upper = np.full(K, 1.0 / K, dtype=float)
    self.cum_loss_lower = np.zeros(K, dtype=float)
    self.cum_loss_upper = np.zeros(K, dtype=float)

    self.last_prediction: float | None = None
    self.last_q_k: np.ndarray | None = None

  def _alpha_floor(self) -> float:
    n = len(self.residuals) if self.residuals is not None else 0
    return 1.0 / (n + 1) + 1e-6

  def _clip_alpha(self, a: float) -> float:
    return max(min(float(a), 0.5), self._alpha_floor())

  @override
  def fit_and_calibrate(
    self,
    train_data: pd.DataFrame,
    calib_data: pd.DataFrame | None = None,
  ):
    """Fit base model once, compute calibration residuals once, seed all K
    experts at initial_alpha. Experts diverge online once `adapt` is called.
    """
    if calib_data is None or len(calib_data) == 0:
      raise ValueError("AgACI requires a calibration dataset (same as ACI/Split).")

    self.model.fit(train_data)

    calib_values = calib_data[self.label_col_name].to_numpy()
    residuals: list[float] = []
    for i in range(self.num_lags, len(calib_data)):
      history = calib_data.iloc[:i]
      pred = self.model.predict(history)
      residuals.append(float(abs(calib_values[i] - pred)))

    residuals_arr = np.array([r for r in residuals if not np.isnan(r)])
    if len(residuals_arr) == 0:
      raise RuntimeError("No calibration residuals produced.")

    self.residuals = residuals_arr
    self.q_k = np.array([
      self.conformal_quantile(self.residuals, self._clip_alpha(a))
      for a in self.alpha_k
    ])
    # Report a mean half-width as the headline quantile_residuals for consistency
    # with the BaseConformal contract; the actual intervals are asymmetric.
    self.quantile_residuals = float(np.dot((self.w_lower + self.w_upper) / 2.0, self.q_k))

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    """Single-model point forecast; per-side weighted combination of expert
    half-widths. Intervals may be asymmetric even with symmetric half-widths
    because the two sides' weight vectors can differ.
    """
    assert len(data) >= self.num_lags
    if self.residuals is None:
      raise RuntimeError("AgACI must be calibrated; call fit_and_calibrate first.")

    mu = float(self.model.predict(data))
    q_lo = float(np.dot(self.w_lower, self.q_k))
    q_up = float(np.dot(self.w_upper, self.q_k))

    self.last_prediction = mu
    self.last_q_k = self.q_k.copy()
    return mu, mu - q_lo, mu + q_up

  def adapt(self, real_value: float) -> None:
    """Three-part online update: accumulate per-expert pinball losses on each
    side, refresh aggregation weights, then update each expert's ACI alpha
    against its own interval (not the aggregated one).
    """
    if self.last_prediction is None or self.last_q_k is None:
      return
    if self.residuals is None:
      raise RuntimeError("AgACI must be calibrated; call fit_and_calibrate first.")

    mu = self.last_prediction
    q_k = self.last_q_k

    tau_lo = self.initial_alpha / 2.0
    tau_up = 1.0 - self.initial_alpha / 2.0
    for k in range(self.K):
      self.cum_loss_lower[k] += _pinball(tau_lo, real_value, mu - q_k[k])
      self.cum_loss_upper[k] += _pinball(tau_up, real_value, mu + q_k[k])

    self.w_lower = _softmax_neg(self.cum_loss_lower, self.eta)
    self.w_upper = _softmax_neg(self.cum_loss_upper, self.eta)

    residual = abs(real_value - mu)
    for k in range(self.K):
      miss_k = 1.0 if residual > q_k[k] else 0.0
      new_alpha = self.alpha_k[k] + self.gamma_grid[k] * (self.initial_alpha - miss_k)
      self.alpha_k[k] = self._clip_alpha(new_alpha)
      self.q_k[k] = self.conformal_quantile(self.residuals, self.alpha_k[k])
