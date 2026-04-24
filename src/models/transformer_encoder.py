import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import override
from src.models.base_model import TimeSeriesModel
from src.data.transform import build_deep_data

class Transformer(nn.Module):
  """
  Class that trains a Transformer to predict financial timeseries data
  
  Expects data in the shape (num_samples, num_lags, 1)
  """
  def __init__(self, num_lags: int, model_dim: int = 32, num_heads: int = 4):
    super().__init__()
    self.num_lags = num_lags
    
    self.input_layer = nn.Linear(1, model_dim)
    
    self.pos_embedding = nn.Parameter(torch.zeros(1, num_lags, model_dim))
    
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=model_dim, 
        nhead=num_heads, 
        batch_first=True
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
    
    self.regressor = nn.Linear(model_dim, 1)

  def forward(self, x):
    x = self.input_layer(x) + self.pos_embedding
    x = self.transformer(x)
    
    last_step_output = x[:, -1, :] 
    return self.regressor(last_step_output)
  
class TransformerModel(TimeSeriesModel):
  def __init__( 
    self, 
    label_col_name: str,
    num_lags: int = 1,
    model_dim: int = 32, 
    num_heads: int = 4,
    epochs: int = 20
  ):
    """
    Initializes an Transformer based model
    
    :param label_col_name: the name of the column with the labels
    :param num_lags: how many previous samples to use to predict the next one
    :other args: optional configuration parameters for the model
    """
    super().__init__(label_col_name, num_lags)
    
    self.epochs = epochs

    self.network = Transformer(num_lags, model_dim, num_heads) 
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
    self.criterion = torch.nn.MSELoss()

  @override    
  def fit(self, data: pd.DataFrame, *args, **kwargs):
    super().fit(data)
    
    X, y = build_deep_data(data, self.label_col_name, self.num_lags)
    
    X_tensor = torch.from_numpy(np.array(X)).reshape(-1, self.num_lags, 1)
    y_tensor = torch.from_numpy(np.array(y))

    self.network.train()
    for _ in range(self.epochs):
      self.optimizer.zero_grad()
      output = self.network(X_tensor)
      loss = self.criterion(output, y_tensor)
      loss.backward()
      self.optimizer.step()

  @override
  def predict(self, data: pd.DataFrame) -> float:
    self.network.eval()
    
    last_window = data[self.label_col_name][-self.num_lags:]
    input_tensor = torch.from_numpy(last_window).reshape(1, self.num_lags, 1)
    
    with torch.no_grad():
        prediction = self.network(input_tensor).numpy()
    return float(prediction[0])