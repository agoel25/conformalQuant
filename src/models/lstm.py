import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from data.transform import build_deep_data

def train_lstm(
  data: pd.DataFrame, 
  label_col_name: str,
  num_lags: int = 1,
  output_dim: int = 50,
  activation: str = 'relu',
  optimizer: str = 'adam',
  loss: str = 'mse',
  epochs: int = 20):
  """
  Trains an LSTM model on financial timeseries data
    
  :param data: a pandas dataframe with the timeseries data
  :param label_col_name: the name of the column with the labels
  :param num_lags: how many previous samples to use to predict the next one
  :return: the model
  """
  
  X, y = build_deep_data(data, label_col_name, num_lags)
  
  model = Sequential([
      LSTM(output_dim, activation=activation, input_shape=(num_lags, 1)),
      Dense(1)
  ])
  
  model.compile(optimizer=optimizer, loss=loss)
  
  model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
  