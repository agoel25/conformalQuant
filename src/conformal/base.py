from abc import ABC, abstractmethod
from typing import Tuple, Type
import numpy as np
import pandas as pd
from src.models.base_model import TimeSeriesModel

class IntervalPredictor(ABC):
  """
  A predictor that trains and outputs a point prediction
  along with a confidence interval
  """
  def __init__(
    self,
    label_col_name: str,
    num_lags: int = 1,
    confidence_level: float = 0.95
  ):
    """
    :param label_col_name: the name of the column with the labels
    :param lags: Integer, number of previous time steps to use as features
    :param confidence_level: The desired probability (1 - alpha)
    """
    self.num_lags = num_lags
    self.label_col_name = label_col_name
    self.confidence_level = confidence_level
    self.alpha = 1 - confidence_level

    self.residuals: np.ndarray | None = None
    self.quantile_residuals: float | None = None
    
  @staticmethod
  def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """
    Finite-sample split-conformal quantile: the value at rank ceil((n+1)(1-alpha))
    of the sorted absolute residuals. Returns +inf if the rank exceeds n, which
    indicates the calibration set is too small to guarantee (1-alpha) coverage.
    """
    n = len(residuals)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
      return float("inf")
    return float(np.sort(residuals)[rank - 1])

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


class BaseConformal(IntervalPredictor):
  """
  A wrapper for a base model to perform conformal prediction 
  that provides uncertainty bounds.
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
    super().__init__(
      label_col_name=label_col_name,
      num_lags=num_lags,
      confidence_level=confidence_level
    )
    self.model = model_class(num_lags=num_lags, label_col_name=label_col_name)

  