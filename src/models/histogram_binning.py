import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from typing import override

from src.models.base_model import TimeSeriesModel


class HistogramBinningModel(TimeSeriesModel):
  """Simple histogram-binning point predictor.

  Buckets the mean of the last `num_lags` observations into `num_bins` quantile
  bins during training and memorises the mean next-step value per bin. At
  prediction time, bins the incoming lag-window mean and returns the learned
  bin mean. Intended as a weak, interpretable baseline that plugs into the
  conformal wrappers alongside LinearAR / LSTM / Transformer.
  """
  def __init__(
    self,
    label_col_name: str,
    num_lags: int = 1,
    num_bins: int = 10,
  ):
    super().__init__(label_col_name, num_lags)
    self.num_bins = num_bins
    self.binner: KBinsDiscretizer | None = None
    self.bin_to_next: np.ndarray | None = None
    self.fallback: float = 0.0

  @override
  def fit(self, data: pd.DataFrame) -> None:
    values = data[self.label_col_name].dropna().values.astype(np.float64)
    if len(values) <= self.num_lags:
      raise ValueError(
        f"HistogramBinningModel.fit: need more than num_lags={self.num_lags} rows, got {len(values)}"
      )

    windows = np.lib.stride_tricks.sliding_window_view(values, self.num_lags)
    X_mean = windows[:-1].mean(axis=1).reshape(-1, 1)
    y = values[self.num_lags:]

    self.binner = KBinsDiscretizer(
      n_bins=self.num_bins,
      encode="ordinal",
      strategy="quantile",
      quantile_method="linear",
    )
    bin_ids = self.binner.fit_transform(X_mean).astype(int).ravel()

    self.bin_to_next = np.full(self.num_bins, fill_value=float(y.mean()), dtype=np.float64)
    for b in range(self.num_bins):
      mask = bin_ids == b
      if mask.any():
        self.bin_to_next[b] = float(y[mask].mean())
    self.fallback = float(y.mean())

  @override
  def predict(self, data: pd.DataFrame) -> float:
    if self.binner is None or self.bin_to_next is None:
      raise RuntimeError("HistogramBinningModel.predict called before fit()")

    last_window = data[self.label_col_name].values[-self.num_lags:]
    x = np.array([[float(np.mean(last_window))]])
    bin_id = int(self.binner.transform(x)[0, 0])
    bin_id = max(0, min(bin_id, self.num_bins - 1))
    return float(self.bin_to_next[bin_id])
