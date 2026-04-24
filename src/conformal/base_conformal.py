from abc import ABC, abstractmethod
from typing import Tuple, Type
import pandas as pd
from src.models.base_model import TimeSeriesModel

class BaseConformal(ABC):
  """
  A wrapper for conformal prediction that provides uncertainty bounds.
  """
  def __init__(
    self, 
    model_class: Type[TimeSeriesModel],
    label_col_name: str, 
    num_lags: int = 1,
    confidence_level: float = 0.95
  ):
    """
    :param model_class: The class of model to initialize train, and predict with
    :param label_col_name: the name of the column with the labels
    :param lags: Integer, number of previous time steps to use as features
    :param confidence_level: The desired probability (1 - alpha)
    """
    self.model = model_class(num_lags=num_lags, label_col_name=label_col_name)
    self.num_lags = num_lags
    self.label_col_name = label_col_name
    self.confidence_level = confidence_level
    self.alpha = 1 - confidence_level
    
    self.calibration_scores = None

  @abstractmethod
  def fit_and_calibrate(self, train_data: pd.DataFrame, calib_data: pd.DataFrame | None = None):
    """
    1. Fits the underlying model using train_data.
    2. Generates predictions for calib_data.
    3. Calculates and stores non-conformity scores (residuals).
    
    :param train_data: the data to train the model on
    :param calib_data: the data to calibrate with (calculate prediction intervals with)
                       None if conformal implementation doesn't use a calibration set
    """
    pass

  @abstractmethod
  def predict_with_interval(self, data: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Returns a tuple of (prediction, lower_bound, upper_bound).
    
    The bounds are calculated by applying the calibrated quantile 
    to the model's point prediction.
    
    :param data: the data to predict with
    """
    assert len(data) >= self.num_lags
    pass