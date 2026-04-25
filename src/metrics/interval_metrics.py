from __future__ import annotations
import datetime
import pandas as pd
from typing import Tuple, List

class IntervalMetrics:
  def __init__(
    self, 
    upper_bounds: pd.Series[float], 
    lower_bounds: pd.Series[float], 
    is_high_vol: pd.Series[bool], 
    is_low_vol: pd.Series[bool]
  ):
    self.widths = upper_bounds - lower_bounds
    self.is_high_vol = is_high_vol
    self.is_low_vol = is_low_vol

  def get_mean_widths(self) -> Tuple[float, float, float]:
    return self.widths.mean(), self.widths[self.is_high_vol].mean(), self.widths[self.is_low_vol].mean()

  def get_period_width(self, start: datetime.date, end: datetime.date) -> float:
    return self.widths.loc[start:end].mean()