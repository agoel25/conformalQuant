import numpy as np
import pandas as pd
from arch import arch_model
from typing import Tuple, override
from src.conformal.base import IntervalPredictor

class Garch(IntervalPredictor):
  def __init__(
    self, 
    label_col_name: str, 
    num_lags: int = 1,
    confidence_level: float = 0.95
  ):
    super().__init__(
      label_col_name=label_col_name, 
      num_lags=num_lags, 
      confidence_level=confidence_level
    )
    
    self.mu = 0.0
    self.garch_results = None
    self.q_hat = None

  @override
  def fit_and_calibrate(
    self,
    train_data: pd.DataFrame, 
    calib_data: pd.DataFrame | None = None
  ):
    """
    Fits a Constant Mean GARCH(1,1) and calculates conformal scores.
    """
    if calib_data is None:
        raise ValueError("Calibration data is required for conformal scores.")

    # 1. Calculate Constant Mean (mu) from training data
    self.mu = train_data[self.label_col_name].mean()

    # 2. Fit GARCH(1,1) with Student-t
    # mean='Constant' in arch_model precisely matches y_t = mu + epsilon_t
    am = arch_model(
        train_data[self.label_col_name],
        mean='Constant',
        vol='Garch',
        p=1,
        q=1,
        dist='studentst',
        rescale=False
    )
    self.garch_results = am.fit(disp='off')
    
    # Update our mu to the optimized mean from the GARCH fit
    self.mu = self.garch_results.params['mu']

    # 3. Calibration: Standardized Residuals
    # score s_t = |y_t - mu| / sigma_t
    actuals_calib = calib_data[self.label_col_name].values
    
    # Get conditional volatility for the calibration period
    # We use the 'forecast' method to project volatility onto the calibration set
    forecasts = self.garch_results.model.forecast(
        self.garch_results.params,
        horizon=1,
        data=calib_data[self.label_col_name],
        align='target'
    )
    # Note: we use the volatility from the step immediately preceding 
    # each calibration point to stay 'out-of-sample'
    calib_vol = np.sqrt(forecasts.variance.values.flatten())
    
    resids = np.abs(actuals_calib - self.mu)
    self.calibration_scores = resids / calib_vol

    # 4. Calculate the Conformal Quantile
    self.q_hat = np.quantile(self.calibration_scores, 1 - self.alpha)

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Returns (mu, lower, upper) using the paper's constant mean and GARCH variance.
    """
    super().predict_with_interval(data)
    
    # 1. Constant Mean Prediction
    mu_pred = self.mu

    # 2. Forecast next-day Variance sigma^2_{t+1}
    # Based on: sigma^2_t = omega + alpha*eps^2_{t-1} + beta*sigma^2_{t-1}
    forecasts = self.garch_results.model.forecast(
        self.garch_results.params,
        horizon=1,
        data=data[self.label_col_name],
        align='target'
    )
    
    sigma_next = np.sqrt(forecasts.variance.values[-1, 0])

    # 3. Construct Intervals
    # Bound = mu +/- (q_hat * sigma)
    margin = self.q_hat * sigma_next
    
    lower_bound = mu_pred - margin
    upper_bound = mu_pred + margin

    return float(mu_pred), float(lower_bound), float(upper_bound)