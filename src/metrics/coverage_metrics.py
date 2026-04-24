import datetime
import pandas as pd
from typing import Tuple

class CoverageMetrics:
  def __init__(
    self, 
    is_covered: pd.Series[bool], 
    is_high_vol: pd.Series[bool], 
    is_low_vol: pd.Series[bool]
  ):
    self.is_covered = is_covered
    self.is_high_vol = is_high_vol
    self.is_low_vol = is_low_vol

  def get_marginal_coverage(self) -> float:
    return self.is_covered.mean()

  def get_high_low_vol_coverage(self) -> Tuple[float, float, float]:
    cov_high = self.is_covered[self.is_high_vol].mean()
    cov_low = self.is_covered[self.is_low_vol].mean()
    return cov_high, cov_low, (cov_low - cov_high)

  def get_period_coverage(self, start: datetime.date, end: datetime.date) -> float:
    return self.is_covered.loc[start:end].mean()