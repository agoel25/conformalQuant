from src.models.base_model import TimeSeriesModel
from conformal.base import BaseConformal
from typing import Type, override
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap

class EnbPIConformal(BaseConformal):
  @override
  def __init__(
    self, 
    model_class: Type[TimeSeriesModel],
    label_col_name: str, 
    num_bootstraps: int,
    aggregation_func: str = "mean",
    num_blocks: int = 10,
    num_lags: int = 1,
    confidence_level: float = 0.95
  ):
    super().__init__(model_class, label_col_name, num_lags, confidence_level)
    self.aggregation_func = aggregation_func
    
    self.num_bootstraps = num_bootstraps
    
    cv_mapietimeseries = BlockBootstrap(
      n_resamplings=num_bootstraps, n_blocks=num_blocks, overlapping=False, random_state=59
    )

    self.mapie_enpbi = TimeSeriesRegressor(
        self.model,
        method="enbpi",
        cv=cv_mapietimeseries,
        agg_function=aggregation_func,
        n_jobs=1,
    )
  
  @override
  def fit_and_calibrate(self, train_data: pd.DataFrame, calib_data: pd.DataFrame | None = None):
    # our model doesn't need labels as it predicts just based on the input data
    self.mapie_enpbi.fit(train_data, range(len(train_data)))
  
  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    super().predict_with_interval(data)
    prediction, bounds = self.mapie_enpbi.predict(
      data, 
      confidence_level=1 - self.alpha, 
      ensemble=True, 
      optimize_beta=True
    )
    
    print(prediction)
    print(bounds)
    return (0.0, 0.0, 0.0)