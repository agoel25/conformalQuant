from src.conformal.base_conformal import BaseConformal
from typing import override
import numpy as np
import pandas as pd
from typing import Tuple

class SplitConformal(BaseConformal):
  """
  Implements Split Conformal Prediction for Time Series.
  Calculates a fixed interval width based on calibration residuals.
  """
  
  @override
  def fit_and_calibrate(self, train_data: pd.DataFrame, calib_data: pd.DataFrame | None = None):
    """
    Trains the underlying model.
    Uses the calibration dataset, predicts on each set of previous samples
    and finds residuals + quantiles for those residuals based on the given confidence level
    Uses these quantiles as the prediction intervals for all future predictions
    """
    if calib_data is None:
      print("Split Conformal Wrapper requires a calibration dataset, please provide one!")
      return
    
    self.model.fit(train_data)
    
    # we need to generate predictions for every set of previous samples
    # to find residuals
    calib_values = calib_data[self.label_col_name].values
    residuals = []
    
    # We start from n_lags because the model needs history to predict
    for i in range(self.num_lags, len(calib_data)):
      current_history = calib_data.iloc[:i]
      actual = calib_values[i]

      prediction = self.model.predict(current_history)
      residuals.append(abs(actual - prediction))
    
    residuals = np.array(residuals)
    residuals = residuals[~np.isnan(residuals)]
    
    # We use the formula: quantile = ceil((n+1)(1-alpha)) / n
    n = len(residuals)
    q_level = (1 - self.alpha) * (1 + 1/n)
    # Clip to ensure it stays within [0, 1]
    q_level = min(max(q_level, 0.0), 1.0)
    
    self.residuals = residuals    
    self.quantile_residuals = np.quantile(residuals, q_level)
    
  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Returns (prediction, lower_bound, upper_bound) using 
    the static interval calibrated during fit.
    """
    super().predict_with_interval(data)
    if self.quantile_residuals is None:
        print("Model must be calibrated before prediction. Call fit_and_calibrate first.")
        return (0, 0, 0)
    
    # Get the point prediction from the underlying model
    prediction = self.model.predict(data)
    
    # Apply the static margin (the 'quantile' of residuals)
    lower_bound = prediction - self.quantile_residuals
    upper_bound = prediction + self.quantile_residuals
    
    return float(prediction), float(lower_bound), float(upper_bound)