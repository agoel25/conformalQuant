import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from data.transform import build_deep_data
from base_model import TimeSeriesModel

class LSTMModel(TimeSeriesModel):
  def __init__(
    self, 
    label_col_name: str,
    num_lags: int = 1,
    output_dim: int = 50,
    activation: str = 'relu',
    optimizer: str = 'adam',
    loss: str = 'mse',
    epochs: int = 20
  ):
    """
    Initializes an LSTM model
    
    :param label_col_name: the name of the column with the labels
    :param num_lags: how many previous samples to use to predict the next one
    :other args: optional configuration parameters for the model
    """
    super().__init__(label_col_name, num_lags)
    
    self.epochs = epochs
    
    self.model = Sequential([
      LSTM(output_dim, activation=activation, input_shape=(num_lags, 1)),
      Dense(1)
    ])
    self.model.compile(optimizer=optimizer, loss=loss)

  def fit(self, data: pd.DataFrame):
    """
    Trains an LSTM model on financial timeseries data
      
    :param data: a pandas dataframe with the timeseries data
    """
    X, y = build_deep_data(data, self.label_col_name, self.num_lags)
    self.model.fit(X, y, epochs=self.epochs, batch_size=16, verbose=0)

  def predict(self, data: pd.DataFrame) -> float:
    last_window = data[self.label_col_name][-self.num_lags:].reshape(1, self.num_lags, 1)
    prediction = self.model.predict(last_window, verbose=0)
    return float(prediction[0])
  