import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.data.transform import build_deep_data
from src.models.base_model import TimeSeriesModel
from typing import override


class LSTMNetwork(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    self.final_layer = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    out, (hn, cn) = self.lstm(x)
    out = self.final_layer(out[:, -1, :])
    return out


class LSTMModel(TimeSeriesModel):
  def __init__(
    self,
    label_col_name: str,
    num_lags: int = 1,
    hidden_dim: int = 50,
    learning_rate: float = 1e-3,
    loss_fn = nn.MSELoss(),
    epochs: int = 50,
    batch_size: int = 32,
    val_frac: float = 0.15,
    patience: int = 5,
    grad_clip: float = 1.0,
  ):
    """Single-layer LSTM + linear head.

    Trained with proper mini-batch SGD, chronological train/val split, and
    early stopping on validation loss. Prior versions of this file ran a
    single full-batch gradient step per "epoch" (~20 total updates), which
    left the model untrained; downstream predictions collapsed to a
    near-constant value.
    """
    super().__init__(label_col_name, num_lags)

    self.hidden_dim = hidden_dim
    self.learning_rate = learning_rate
    self.loss_fn = loss_fn
    self.epochs = epochs
    self.batch_size = batch_size
    self.val_frac = val_frac
    self.patience = patience
    self.grad_clip = grad_clip

    self.model = LSTMNetwork(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

  @override
  def fit(self, data: pd.DataFrame, *args, **kwargs):
    super().fit(data)

    # Drop rows with NaN/inf in the label (e.g. pre-IPO rows)
    clean = data[np.isfinite(data[self.label_col_name])]
    X, y = build_deep_data(clean, self.label_col_name, self.num_lags)
    if len(X) <= self.num_lags:
      return  # not enough data; leave model in init state

    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    # Chronological train/val split — hold out the last val_frac as validation
    # to select the checkpoint that best generalizes forward in time.
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
      self.model.train()
      for batch_X, batch_y in loader:
        self.optimizer.zero_grad()
        out = self.model(batch_X)
        loss = self.loss_fn(out, batch_y)
        loss.backward()
        if self.grad_clip is not None:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

      if X_va is None:
        continue

      self.model.eval()
      with torch.no_grad():
        val_out = self.model(X_va)
        val_loss = float(self.loss_fn(val_out, y_va).item())

      if val_loss < best_val - 1e-8:
        best_val = val_loss
        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        patience_counter = 0
      else:
        patience_counter += 1
        if patience_counter >= self.patience:
          break

    if best_state is not None:
      self.model.load_state_dict(best_state)

  @override
  def predict(self, data: pd.DataFrame) -> float:
    self.model.eval()
    last_window = data[self.label_col_name].values[-self.num_lags:]
    x = torch.from_numpy(last_window.astype(np.float32)).view(1, self.num_lags, 1)
    with torch.no_grad():
      prediction = self.model(x)
    return float(prediction.item())
