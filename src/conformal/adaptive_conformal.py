from src.conformal.split_conformal import SplitConformal
from typing import override
import pandas as pd
from typing import Tuple, Type
from src.models.base_model import TimeSeriesModel

class AdaptiveConformal(SplitConformal):
  """
  Implements Adaptive Conformal Prediction for Time Series.
  Calculates residuals using a calibration set
  but continually adjusts the confidence level used to calculate prediction intervals
  based on observed reality
  """
  def __init__(
    self,
    model_class: Type[TimeSeriesModel],
    label_col_name: str,
    step_size: float,
    num_lags: int = 1,
    confidence_level: float = 0.95
  ):
    """
    All args same as parent init

    :param step_size: the rate at which the wrapper should adjust its confidence intervals
    """
    super().__init__(
      model_class,
      label_col_name,
      num_lags,
      confidence_level
    )

    self.initial_alpha = 1 - confidence_level
    self.step_size = step_size
    self.last_prediction: float | None = None

  @override
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Predicts the next sample based on the given data and current calibration scores
    Stores the new prediction for future adapting
    """
    prediction, lower_bound, upper_bound = super().predict_with_interval(data)

    self.last_prediction = prediction

    return float(prediction), float(lower_bound), float(upper_bound)

  def adapt(self, real_value: float) -> None:
    """
    Adapts the wrapper by providing the real observed value for the last prediction
    If the residual between these is too large, model adjusts its confidence level lower
    Else, increases confidence level
    """
    if self.last_prediction is None:
      print("Please predict a value first before adapting the model!")
      return
    if self.residuals is None or self.quantile_residuals is None:
      print("Model must be calibrated before adapting. Call fit_and_calibrate first.")
      return

    residual = abs(self.last_prediction - real_value)

    error = 1 if residual > self.quantile_residuals else 0

    self.alpha = self.alpha + self.step_size * (self.initial_alpha - error)
    self.alpha = max(min(self.alpha, 0.5), 1.0 / (len(self.residuals) + 1) + 1e-6)

    self.quantile_residuals = self.conformal_quantile(self.residuals, self.alpha)
