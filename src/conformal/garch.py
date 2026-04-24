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

    # 2. Fit GARCH(1,1) with Student-t.
    # mean='Constant' in arch_model matches y_t = mu + epsilon_t. Drop
    # NaN/inf from the training window (e.g. pre-IPO rows) before fitting
    # — arch_model rejects non-finite values.
    train_series = train_data[self.label_col_name]
    train_series = train_series[np.isfinite(train_series)]
    am = arch_model(
        train_series,
        mean='Constant',
        vol='Garch',
        p=1,
        q=1,
        dist='studentst',
        rescale=True
    )
    self.garch_results = am.fit(disp='off')

    # Update our mu to the optimized mean from the GARCH fit
    self.mu = self.garch_results.params['mu']

    # Validate the fit: the optimizer occasionally converges to
    # non-stationary GARCH params (alpha+beta >= 1) and a bogus constant
    # mean several orders of magnitude outside the returns range. When
    # that happens the forward-in-time volatility recursion explodes.
    # Raise so the caller can decide to skip this (ticker, regime) pair
    # rather than emitting interval widths of ~10^9.
    alpha_plus_beta = float(
      self.garch_results.params['alpha[1]'] + self.garch_results.params['beta[1]']
    )
    if alpha_plus_beta >= 1.0:
      raise RuntimeError(
        f"GARCH fit non-stationary: alpha+beta={alpha_plus_beta:.3f} >= 1. "
        f"Forward volatility recursion will explode; skip this (asset, window)."
      )

    # 3. Calibration: Standardized Residuals  (score s_t = |y_t - mu| / sigma_t)
    # arch 8.x dropped the `data=` kwarg to forecast(); we use fix() instead
    # to apply the trained GARCH parameters over train+calib and read off the
    # conditional_volatility trajectory for the calibration range.
    actuals_calib = calib_data[self.label_col_name].values

    combined = pd.concat([train_data[self.label_col_name], calib_data[self.label_col_name]])
    combined = combined[np.isfinite(combined)]
    am_full = arch_model(
      combined, mean='Constant', vol='Garch', p=1, q=1, dist='studentst', rescale=True,
    )
    fit_full = am_full.fix(self.garch_results.params)
    # Slice the volatility path to the calibration rows only.
    vol_all = np.asarray(fit_full.conditional_volatility)
    calib_vol = vol_all[-len(calib_data):]

    # Cap sigma forecasts at 5x the training-period 99th percentile. Some
    # asset/window combos (notably AIG 2008) produce non-stationary GARCH
    # params where volatility explodes when fix() is applied to new data —
    # this bounds the failure mode to sane widths. Stores the cap for reuse
    # at predict time.
    train_vol = np.asarray(self.garch_results.conditional_volatility)
    train_vol = train_vol[np.isfinite(train_vol)]
    if len(train_vol) > 0:
      self._sigma_cap = float(5.0 * np.quantile(train_vol, 0.99))
    else:
      self._sigma_cap = float("inf")
    calib_vol = np.minimum(calib_vol, self._sigma_cap)

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
    # arch 8.x: fix() lets us apply the trained params to the rolling history
    # without refitting, then forecast() extrapolates one step ahead.
    rolling_series = data[self.label_col_name]
    rolling_series = rolling_series[np.isfinite(rolling_series)]
    am_rolling = arch_model(
      rolling_series,
      mean='Constant', vol='Garch', p=1, q=1, dist='studentst', rescale=True,
    )
    fit_rolling = am_rolling.fix(self.garch_results.params)
    forecasts = fit_rolling.forecast(horizon=1, reindex=False)

    sigma_next = np.sqrt(forecasts.variance.values[-1, 0])
    sigma_next = min(float(sigma_next), getattr(self, "_sigma_cap", float("inf")))

    # 3. Construct Intervals
    # Bound = mu +/- (q_hat * sigma)
    margin = self.q_hat * sigma_next
    
    lower_bound = mu_pred - margin
    upper_bound = mu_pred + margin

    return float(mu_pred), float(lower_bound), float(upper_bound)