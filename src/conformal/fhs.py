import numpy as np
import pandas as pd
from typing import Tuple
from arch import arch_model
from src.conformal.base import IntervalPredictor

class FHS(IntervalPredictor):
  """
  A 'Pure' Filtered Historical Simulation implementation.
  Skips a predictive mean model and uses raw log returns as residuals,
  standardized by GARCH volatility.
  """
    
  def __init__(
      self,
      label_col_name: str,
      confidence_level: float = 0.95,
      vol_model: str = 'Garch',
      p: int = 1,
      q: int = 1
  ):
    super().__init__(label_col_name, 0, confidence_level)
    self.vol_model_type = vol_model
    self.p = p
    self.q = q
    self.garch_fit = None

  def fit_and_calibrate(self, train_data: pd.DataFrame, calib_data: pd.DataFrame | None = None):
    """
    1. Takes raw log returns from the training data.
    2. Fits a GARCH model directly to these returns.
    3. Extracts standardized residuals (filtered innovations).
    """
    # In FHS, if returns are serially correlated, we use the residuals
    # of the return process. Drop NaN/inf (e.g. from pre-IPO rows in the
    # training window) — arch_model rejects non-finite values.
    series = train_data[self.label_col_name]
    returns = series[np.isfinite(series)].values
    
    # Fit GARCH directly to the returns. 
    # We use a Constant mean (or Zero) as the 'model'
    am = arch_model(
      returns, 
      vol=self.vol_model_type, 
      p=self.p, 
      q=self.q, 
      mean='Constant'
    )
    self.garch_fit = am.fit(disp="off")
    
    # η_t = ε_t / √h_t
    # arch returns pandas Series in some versions and numpy ndarray in others.
    # np.asarray normalizes both shapes so the subsequent arithmetic works.
    raw_resid = np.asarray(self.garch_fit.resid)
    volatility = np.asarray(self.garch_fit.conditional_volatility)
    
    # This filtered process yields approximately i.i.d. returns
    self.residuals = raw_resid / volatility
    
    if self.residuals is None:
      return
    
    # Calculate the conformal quantile for the desired confidence level
    # We use the absolute standardized residuals to form a symmetric interval.
    abs_std_residuals = np.abs(self.residuals)
    self.quantile_residuals = FHS.conformal_quantile(abs_std_residuals, self.alpha)

  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Calculates the prediction interval by scaling the historical standardized 
    quantile by the current forecast of volatility
    
    Instead of doing the sampling described in the paper, it directly uses the 
    required quantile from the standardized past returns. This is equivalent to
    sampling + finding a range of forecasts + finding quantile of forecasts
    """
    assert self.garch_fit is not None, "Model must be calibrated before prediction."
    
    # 1. The 'Point Forecast' here is the constant mean estimated by GARCH.
    point_pred = self.garch_fit.params['mu']
    
    # 2. Forecast the conditional volatility for the next period (√h_{T+1}).
    # This allows the interval to reflect current market conditions.
    forecast = self.garch_fit.forecast(horizon=1, reindex=False)
    forecasted_vol = np.sqrt(forecast.variance.values[-1, -1])
    
    # 3. Scale the standardized quantile by current volatility
    margin = self.quantile_residuals * forecasted_vol
    
    lower_bound = point_pred - margin
    upper_bound = point_pred + margin
    
    return float(point_pred), float(lower_bound), float(upper_bound)