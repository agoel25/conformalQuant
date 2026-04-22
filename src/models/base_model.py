from abc import ABC, abstractmethod
import pandas as pd

class TimeSeriesModel(ABC):
  def __init__(self, label_col_name: str, num_lags: int = 1):
    """
    Intializes the model
    
    :param label_col_name: the name of the column with the labels
    :param lags: Integer, number of previous time steps to use as features
    """
    self.num_lags = num_lags
    self.label_col_name = label_col_name

  @abstractmethod
  def fit(self, data: pd.DataFrame):
    """Train the model on the provided dataframe."""
    pass

  @abstractmethod
  def predict(self, data: pd.DataFrame) -> float:
    """Predict the single next value following the last num_lags entries from the training data."""
    pass