import pandas as pd
import numpy as np
from typing import Tuple

def build_lstm_X(
  data: pd.DataFrame, 
  label_col_name: str,
  num_lags: int = 1
) -> np.typing.NDArray[np.float64]:
  """
  Reshapes the input feature data for deep learning

  It expects a 3-D matrix of lagged features, with each lagged feature in its own list
    
  :param data: a pandas dataframe with the timeseries data
  :param label_col_name: the name of the column with the labels
  :param num_lags: how many previous samples to use to predict the next one
  :return: An X datasets following the above format
  """
  values = data[label_col_name].values
  
  X = []
  
  for i in range(len(data) - num_lags):
    lagged_samples = values[i:(i + num_lags)]
    
    X.append(lagged_samples)
    
  X = np.array(X)
  X = X.reshape((X.shape[0], X.shape[1], 1))
  
  return X

def build_lstm_data(
    data: pd.DataFrame, 
    label_col_name: str,
    num_lags: int = 1
) -> Tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
  """
  Reshapes data for deep learning

  It expects a 3-D matrix of lagged features, with each lagged feature in its own list
    
  :param data: a pandas dataframe with the timeseries data
  :param label_col_name: the name of the column with the labels
  :param num_lags: how many previous samples to use to predict the next one
  :return: X and y datasets following the above format
  """  
  values = data[label_col_name].values
  
  X = build_lstm_X(data, label_col_name, num_lags)
  y = []
  
  for i in range(len(data) - num_lags):
    y.append(values[i + num_lags])
    
  y = np.array(y)
    
  return (X, y)