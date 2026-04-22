import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Tuple

class Transformer(nn.Module):
  """
  Class that trains a Transformer to predict financial timeseries data
  
  Expects data in the shape (num_samples, num_lags, 1)
  """
  def __init__(self, num_lags: int, model_dim: int = 32, n_heads: int = 4):
    super().__init__()
    self.num_lags = num_lags
    
    self.input_layer = nn.Linear(1, model_dim)
    
    self.pos_embedding = nn.Parameter(torch.zeros(1, num_lags, model_dim))
    
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=model_dim, 
        nhead=n_heads, 
        batch_first=True
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
    
    self.regressor = nn.Linear(model_dim, 1)

  def forward(self, x):
    x = self.input_layer(x) + self.pos_embedding
    x = self.transformer(x)
    
    last_step_output = x[:, -1, :] 
    return self.regressor(last_step_output)