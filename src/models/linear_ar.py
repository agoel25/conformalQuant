import pandas as pd
from sklearn.linear_model import LinearRegression
from src.models.base_model import TimeSeriesModel
from typing import override

class LinearARModel(TimeSeriesModel):
  def __init__(
    self,
    label_col_name: str,
    num_lags: int = 1
  ):
    super().__init__(label_col_name, num_lags)
    self.model = LinearRegression()

  @override
  def fit(self, data: pd.DataFrame):
    """
    Trains a linear autoregression model.
    
    :param data: Financial Timeseries data as a Dataframe
    """    
    training_data = data[[self.label_col_name]].copy()
    
    for i in range(1, self.num_lags + 1):
        training_data[f'lag_{i}'] = data[self.label_col_name].shift(i)
    
    training_data.dropna(inplace=True)
    
    X = training_data.drop(self.label_col_name, axis=1).values
    y = training_data[self.label_col_name].values
  
    self.model.fit(X, y)
    
  @override
  def predict(self, data: pd.DataFrame) -> float:
    last_window = data[self.label_col_name][-self.num_lags:].reshape(1, -1)
    prediction = self.model.predict(last_window)
    return float(prediction[0])