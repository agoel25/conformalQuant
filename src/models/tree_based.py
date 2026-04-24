import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from typing import override

from src.models.base_model import TimeSeriesModel


class TreeBasedModel(TimeSeriesModel):
  """Gradient-boosted trees point predictor over `num_lags`-sized lag windows.

  Nonlinear baseline in the simple bucket. Uses sklearn's GradientBoostingRegressor
  with conservative defaults; if the team wants a quantile forest later, swap in
  RandomForestRegressor or QuantileRegressor — the interface stays the same.
  """
  def __init__(
    self,
    label_col_name: str,
    num_lags: int = 1,
    n_estimators: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    random_state: int = 0,
  ):
    super().__init__(label_col_name, num_lags)
    self.model = GradientBoostingRegressor(
      n_estimators=n_estimators,
      max_depth=max_depth,
      learning_rate=learning_rate,
      random_state=random_state,
    )

  @override
  def fit(self, data: pd.DataFrame) -> None:
    values = data[self.label_col_name].dropna().values.astype(np.float64)
    if len(values) <= self.num_lags:
      raise ValueError(
        f"TreeBasedModel.fit: need more than num_lags={self.num_lags} rows, got {len(values)}"
      )

    windows = np.lib.stride_tricks.sliding_window_view(values, self.num_lags)
    X = windows[:-1]
    y = values[self.num_lags:]
    self.model.fit(X, y)

  @override
  def predict(self, data: pd.DataFrame) -> float:
    last_window = data[self.label_col_name].values[-self.num_lags:].reshape(1, -1)
    return float(self.model.predict(last_window)[0])
