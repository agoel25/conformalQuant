from __future__ import annotations
import pandas as pd
import datetime
from typing import Tuple

class WinklerMetrics:
  def __init__(
    self, 
    real_returns: pd.Series[float], 
    upper_bounds: pd.Series[float], 
    lower_bounds: pd.Series[float], 
    is_high_vol: pd.Series[bool], 
    is_low_vol: pd.Series[bool], 
    alpha: float
  ):
    width = upper_bounds - lower_bounds
    lower_violations = (lower_bounds - real_returns).clip(lower=0)
    upper_violations = (real_returns - upper_bounds).clip(lower=0)
    
    self.scores = width + (2 / alpha) * lower_violations + (2 / alpha) * upper_violations
    self.high_mask = is_high_vol
    self.low_mask = is_low_vol

  def get_mean_scores(self) -> Tuple[float, float, float]:
    return (
      self.scores.mean(), 
      self.scores[self.high_mask].mean(), 
      self.scores[self.low_mask].mean()
    )

  def get_period_score(self, start: datetime.date, end: datetime.date) -> float:
    return self.scores.loc[start:end].mean()