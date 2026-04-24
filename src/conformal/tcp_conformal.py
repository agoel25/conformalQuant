"""TCP and TCP-RM: Temporal Conformal Prediction (Aich et al. 2026).

From-scratch implementation of the two variants in "Temporal Conformal
Prediction (TCP): A Distribution-Free Statistical and Machine Learning
Framework for Adaptive Risk Forecasting".

- TCPConformal: rolling split-conformal on a fixed-width trailing window. At
  each step the last `window` rows are split into `window - calib` training
  rows and `calib` calibration rows; the base model is refit on the training
  slice, absolute residuals are computed on the calibration slice, and the
  (1-alpha) quantile becomes the symmetric half-width. No online alpha update.

- TCPRMConformal: adds an online Robbins-Monro offset to the conformal
  threshold. Each step the offset is updated based on the previous step's
  coverage error; the effective half-width is projected to stay non-negative.

The paper's quantile forecaster is replaced by our point-predictor interface;
this reduces to the standard split-conformal threshold (same conformity score
that Aich et al. call "two-sided nonconformity" collapses to |y - mu| for a
mean-style forecaster).
"""
from __future__ import annotations

from typing import Tuple, Type
from typing import override

import numpy as np
import pandas as pd

from src.conformal.base import BaseConformal
from src.models.base_model import TimeSeriesModel


class TCPConformal(BaseConformal):
  """Rolling split-conformal (Aich et al. 2026, §5.2.1 — TCP without RM)."""

  @override
  def __init__(
    self,
    model_class: Type[TimeSeriesModel],
    label_col_name: str,
    num_lags: int = 1,
    confidence_level: float = 0.95,
    window: int = 252,
    calib: int = 60,
    refit_every: int = 1,
  ):
    """
    :param window: rolling-window length (paper default: 252 trading days)
    :param calib: number of rows at the end of the window used for calibration
                  (paper default: 60; so 192 training rows)
    :param refit_every: refit the base model every N predict calls. 1 (paper)
                        is expensive for deep models; 5 is a practical fallback.
    """
    super().__init__(model_class, label_col_name, num_lags, confidence_level)

    if calib <= 0 or calib >= window:
      raise ValueError(f"calib must satisfy 0 < calib < window; got calib={calib}, window={window}")
    if window <= num_lags + calib:
      raise ValueError(
        f"window={window} too small: need window > num_lags+calib = {num_lags + calib}"
      )
    if refit_every < 1:
      raise ValueError(f"refit_every must be >= 1; got {refit_every}")

    self._model_class = model_class
    self.window = window
    self.calib = calib
    self.refit_every = refit_every

    self._steps_since_refit = 0  # tracks refit_every cadence

  @override
  def fit_and_calibrate(
    self,
    train_data: pd.DataFrame,
    calib_data: pd.DataFrame | None = None,
  ):
    """TCP is inherently online — this is a no-op beyond a sanity check that
    enough warmup history will be available. The actual fit + calibrate cycle
    runs inside every `predict_with_interval` call.
    """
    warmup_len = len(train_data) + (0 if calib_data is None else len(calib_data))
    if warmup_len < self.window:
      # Not strictly fatal: the first predict call will just use whatever
      # history is available, but the paper's guarantees rely on w rows.
      # Warn via raising RuntimeError only in egregious cases.
      if warmup_len < self.num_lags + self.calib + 10:
        raise ValueError(
          f"TCP needs enough warmup to form at least num_lags+calib+10={self.num_lags+self.calib+10} rows; "
          f"train+calib only provides {warmup_len}."
        )

    # Tell BaseConformal we're in a pseudo-calibrated state. The residuals and
    # quantile_residuals populate lazily on the first predict call.
    self.residuals = np.array([], dtype=float)
    self.quantile_residuals = None

  def _refit_and_calibrate_on_window(self, history: pd.DataFrame) -> float:
    """Slice the trailing `window` rows, split into train/calib, refit, and
    return the fresh conformal half-width.
    """
    window_df = history.iloc[-self.window:] if len(history) >= self.window else history
    train_end = len(window_df) - self.calib
    train_slice = window_df.iloc[:train_end]
    calib_slice = window_df.iloc[train_end:]

    model = self._model_class(
      label_col_name=self.label_col_name,
      num_lags=self.num_lags,
    )
    model.fit(train_slice)

    y = calib_slice[self.label_col_name].to_numpy()
    residuals: list[float] = []
    for i in range(self.num_lags, len(calib_slice)):
      hist_i = calib_slice.iloc[:i]
      pred_i = model.predict(hist_i)
      residuals.append(float(abs(y[i] - pred_i)))

    residuals_arr = np.array([r for r in residuals if not np.isnan(r)])
    if len(residuals_arr) == 0:
      raise RuntimeError("TCP produced 0 residuals on the calibration slice.")

    self.model = model  # keep the latest fit for predict_with_interval
    self.residuals = residuals_arr
    self.quantile_residuals = self.conformal_quantile(residuals_arr, self.alpha)
    return float(self.quantile_residuals)

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    """Refit + recalibrate on the trailing window (every `refit_every` calls),
    then emit a symmetric interval using the fresh quantile.
    """
    assert len(data) >= self.num_lags

    # Refit on first call, and every `refit_every` calls thereafter.
    if self.quantile_residuals is None or self._steps_since_refit >= self.refit_every:
      self._refit_and_calibrate_on_window(data)
      self._steps_since_refit = 0
    self._steps_since_refit += 1

    point = float(self.model.predict(data))
    q = float(self.quantile_residuals)
    self._last_prediction = point
    self._last_q = q
    return point, point - q, point + q


class TCPRMConformal(TCPConformal):
  """TCP + online Robbins-Monro offset on the conformal threshold (Aich et al.
  2026, Eq. 1 and §4.4).

  After each observation, updates an offset C_rm against the miscoverage error
  `miss - alpha`, and uses the effective threshold q_eff = max(q + C_rm, 0) as
  the interval half-width. Step sizes follow gamma_t = gamma_0 / (1+lambda*t)^beta.
  """

  @override
  def __init__(
    self,
    model_class: Type[TimeSeriesModel],
    label_col_name: str,
    num_lags: int = 1,
    confidence_level: float = 0.95,
    window: int = 252,
    calib: int = 60,
    refit_every: int = 1,
    gamma_0: float = 0.01,
    rm_lambda: float = 1.0,
    rm_beta: float = 0.6,
  ):
    super().__init__(
      model_class=model_class,
      label_col_name=label_col_name,
      num_lags=num_lags,
      confidence_level=confidence_level,
      window=window,
      calib=calib,
      refit_every=refit_every,
    )
    if not (0.5 <= rm_beta <= 1.0):
      raise ValueError(f"rm_beta should be in [0.5, 1.0] (paper); got {rm_beta}")

    self.gamma_0 = float(gamma_0)
    self.rm_lambda = float(rm_lambda)
    self.rm_beta = float(rm_beta)

    self.C_rm: float = 0.0
    self._t: int = 0

    self._last_prediction: float | None = None
    self._last_lower: float | None = None
    self._last_upper: float | None = None
    self._last_q: float | None = None

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    point, _, _ = super().predict_with_interval(data)
    q = float(self.quantile_residuals)
    q_eff = max(q + self.C_rm, 0.0)
    lo, up = point - q_eff, point + q_eff

    self._last_prediction = point
    self._last_lower = lo
    self._last_upper = up
    self._last_q = q  # the un-offset base quantile
    return point, lo, up

  def adapt(self, real_value: float) -> None:
    """Robbins-Monro update on C_rm. Projection C_rm >= -q keeps q_eff >= 0."""
    if self._last_lower is None or self._last_upper is None or self._last_q is None:
      return

    miss = 0.0 if (self._last_lower <= real_value <= self._last_upper) else 1.0
    gamma_t = self.gamma_0 / ((1.0 + self.rm_lambda * self._t) ** self.rm_beta)
    self.C_rm = max(
      self.C_rm + gamma_t * (miss - self.alpha),
      -self._last_q,
    )
    self._t += 1
