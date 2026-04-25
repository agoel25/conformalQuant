from __future__ import annotations
import pandas as pd
import datetime
from typing import Tuple

class AdaptationMetrics:
  def __init__(
    self, 
    is_covered: pd.Series[bool], 
    dates: pd.DatetimeIndex
  ):
    self.is_covered = is_covered
    self.dates = dates

  def compute_adaptation_speed(
    self, 
    crisis_start: datetime.date, 
    crisis_end: datetime.date,
    threshold: float = 0.85, 
    persistence: int = 5, 
    window: int = 20
  ) -> Tuple[int, float, bool]:
      
    rolling_cov = self.is_covered.astype(float).rolling(window).mean()
    
    # Subset to the crisis period
    crisis_rolling = rolling_cov.loc[crisis_start:crisis_end]
    if crisis_rolling.empty:
        return 0, 0.0, False
        
    is_well_covered = (crisis_rolling >= threshold).to_numpy()
    recovery_point = len(is_well_covered)
    has_recovered = False

    for i in range(len(is_well_covered) - persistence + 1):
      if is_well_covered[i : i + persistence].all():
        recovery_point = i
        has_recovered = True
        break

    max_drawdown = crisis_rolling.iloc[:recovery_point].min()
    return recovery_point, max_drawdown, has_recovered