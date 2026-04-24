import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import override
from src.data.transform import build_deep_data
from src.models.base_model import TimeSeriesModel


class Transformer(nn.Module):
  """Transformer encoder over a `num_lags`-length sequence of scalar returns.

  Input shape (batch, num_lags, 1). A small linear input projection + learned
  positional embedding feeds a single TransformerEncoderLayer; the last token's
  hidden state is projected to a scalar prediction.
  """
  def __init__(self, num_lags: int, model_dim: int = 32, num_heads: int = 4):
    super().__init__()
    self.num_lags = num_lags

    self.input_layer = nn.Linear(1, model_dim)
    self.pos_embedding = nn.Parameter(torch.zeros(1, num_lags, model_dim))
    nn.init.normal_(self.pos_embedding, std=0.02)  # small non-zero init

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=model_dim,
      nhead=num_heads,
      dim_feedforward=2 * model_dim,
      dropout=0.1,
      batch_first=True,
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
    self.regressor = nn.Linear(model_dim, 1)

  def forward(self, x):
    x = self.input_layer(x) + self.pos_embedding
    x = self.transformer(x)
    return self.regressor(x[:, -1, :])


class TransformerModel(TimeSeriesModel):
  def __init__(
    self,
    label_col_name: str,
    num_lags: int = 1,
    model_dim: int = 32,
    num_heads: int = 4,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_frac: float = 0.15,
    patience: int = 5,
    grad_clip: float = 1.0,
  ):
    """1-layer Transformer encoder + linear head.

    Trained with proper mini-batch SGD, chronological train/val split, and
    early stopping on validation loss. Prior versions ran a single full-batch
    gradient step per "epoch" (~20 total updates), which left the model
    untrained; predictions then collapsed to a near-constant value.
    """
    super().__init__(label_col_name, num_lags)

    self.epochs = epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.val_frac = val_frac
    self.patience = patience
    self.grad_clip = grad_clip

    self.network = Transformer(num_lags, model_dim, num_heads)
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
    self.criterion = torch.nn.MSELoss()

  @override
  def fit(self, data: pd.DataFrame, *args, **kwargs):
    super().fit(data)

    clean = data[np.isfinite(data[self.label_col_name])]
    X, y = build_deep_data(clean, self.label_col_name, self.num_lags)
    if len(X) <= self.num_lags:
      return

    X_t = torch.from_numpy(X.astype(np.float32)).reshape(-1, self.num_lags, 1)
    y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    n = len(X_t)
    n_val = max(5, int(self.val_frac * n)) if n >= 40 else 0

    if n_val > 0:
      X_tr, X_va = X_t[:-n_val], X_t[-n_val:]
      y_tr, y_va = y_t[:-n_val], y_t[-n_val:]
    else:
      X_tr, y_tr = X_t, y_t
      X_va, y_va = None, None

    loader = DataLoader(
      TensorDataset(X_tr, y_tr),
      batch_size=min(self.batch_size, len(X_tr)),
      shuffle=True,
    )

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for _ in range(self.epochs):
      self.network.train()
      for batch_X, batch_y in loader:
        self.optimizer.zero_grad()
        out = self.network(batch_X)
        loss = self.criterion(out, batch_y)
        loss.backward()
        if self.grad_clip is not None:
          torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

      if X_va is None:
        continue

      self.network.eval()
      with torch.no_grad():
        val_out = self.network(X_va)
        val_loss = float(self.criterion(val_out, y_va).item())

      if val_loss < best_val - 1e-8:
        best_val = val_loss
        best_state = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
        patience_counter = 0
      else:
        patience_counter += 1
        if patience_counter >= self.patience:
          break

    if best_state is not None:
      self.network.load_state_dict(best_state)

  @override
  def predict(self, data: pd.DataFrame) -> float:
    self.network.eval()
    last_window = data[self.label_col_name].values[-self.num_lags:].astype(np.float32)
    x = torch.from_numpy(last_window).reshape(1, self.num_lags, 1)
    with torch.no_grad():
      prediction = self.network(x)
    return float(prediction.item())
