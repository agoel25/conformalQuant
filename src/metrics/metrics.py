import pandas as pd
from typing import cast
from src.metrics.prediction_cols import PredictionCols
from src.metrics.adaptation_metrics import AdaptationMetrics
from src.metrics.coverage_metrics import CoverageMetrics
from src.metrics.interval_metrics import IntervalMetrics
from src.metrics.winkler_metrics import WinklerMetrics

class Metrics:
  def __init__(
    self, 
    data: pd.DataFrame, 
    alpha: float = 0.10
  ):
    """
    :param data: DataFrame indexed by date containing realized returns and bounds
    :param returns_col: Name of column with realized returns (r_t)
    :param upper_col: Name of column with upper interval bound (u_t)
    :param lower_col: Name of column with lower interval bound (l_t)
    :param alpha: Miscoverage rate
    """
    self.data = data.copy()
    self.alpha = alpha
    self.dates = cast(pd.DatetimeIndex, data.index)
    
    # Pre-compute Daily Coverage Indicator
    self.is_covered = (
      (self.data[PredictionCols.LOG_RETURNS] >= 
        self.data[PredictionCols.LOWER_BOUND]) & \
      (self.data[PredictionCols.LOG_RETURNS] <= 
        self.data[PredictionCols.UPPER_BOUND])
    )
    
    # Pre-compute Volatility Masks (Shared Setup)
    rolling_var = self.data[PredictionCols.LOG_RETURNS].rolling(20).var()
    threshold = rolling_var.quantile(0.75)
    self.is_high_vol = rolling_var > threshold
    self.is_low_vol = ~self.is_high_vol

  def get_coverage_metrics(self) -> 'CoverageMetrics':
    return CoverageMetrics(
      self.is_covered, 
      self.is_high_vol, 
      self.is_low_vol
    )

  def get_winkler_metrics(self) -> 'WinklerMetrics':
    return WinklerMetrics(
      self.data[PredictionCols.LOG_RETURNS],
      self.data[PredictionCols.UPPER_BOUND],
      self.data[PredictionCols.LOWER_BOUND],
      self.is_high_vol, 
      self.is_low_vol, 
      self.alpha
    )

  def get_interval_metrics(self) -> 'IntervalMetrics':
    return IntervalMetrics(
      self.data[PredictionCols.UPPER_BOUND],
      self.data[PredictionCols.LOWER_BOUND],
      self.is_high_vol, self.is_low_vol
    )

  def get_adaptation_metrics(self) -> 'AdaptationMetrics':
    return AdaptationMetrics(self.is_covered, self.dates)