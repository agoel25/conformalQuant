import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.transform import build_deep_data
from src.models.base_model import TimeSeriesModel

class LSTMNetwork(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    self.final_layer = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    # x shape: (batch, seq_len, features)
    # out shape: (batch, seq_len, hidden_dim)
    out, (hn, cn) = self.lstm(x)
    # We only take the output from the last time step
    out = self.final_layer(out[:, -1, :])
    return out

class LSTMModel(TimeSeriesModel):
  def __init__(
    self, 
    label_col_name: str,
    num_lags: int = 1,
    hidden_dim: int = 50,
    learning_rate: float = 0.001,
    loss_fn = nn.MSELoss(),
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
    self.learning_rate = learning_rate
    
    self.model = LSTMNetwork(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    self.criterion = loss_fn

  def fit(self, data: pd.DataFrame):
    """
    Trains an LSTM model on financial timeseries data
      
    :param data: a pandas dataframe with the timeseries data
    """
    X, y = build_deep_data(data, self.label_col_name, self.num_lags)
    
    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    self.model.train()
    for _ in range(self.epochs):
      self.optimizer.zero_grad()
      outputs = self.model(X_tensor)
      loss = self.criterion(outputs, y_tensor)
      loss.backward()
      self.optimizer.step()

  def predict(self, data: pd.DataFrame) -> float:
    self.model.eval()
        
    last_window = data[self.label_col_name].values[-self.num_lags:]
    last_window_tensor = torch.from_numpy(last_window.astype(np.float32)).view(1, self.num_lags, 1)
    
    with torch.no_grad():
      prediction = self.model(last_window_tensor)
        
    return float(prediction.item())
  