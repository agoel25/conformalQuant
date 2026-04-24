"""Shared utilities for the analysis notebooks.

Centralises (a) the train / calib / test split convention, (b) data loading,
(c) the per-day walk-forward loop, and (d) interval-array I/O. Every per-model
notebook imports from here so all models are evaluated on identical splits and
all saved files share one schema.
"""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Type

import numpy as np
import pandas as pd

from src.conformal.adaptive_conformal import AdaptiveConformal
from src.conformal.base import BaseConformal, IntervalPredictor
from src.conformal.split_conformal import SplitConformal
from src.data.data import (
  Cols,
  Ticker,
  TICKERS_2008_CRASH,
  TICKERS_2008_FINANCE_RESILIENT,
  TICKERS_2008_STABLE,
  TICKERS_2022_CRASH,
  TICKERS_2022_STABLE,
  TICKERS_COVID_CRASH,
  TICKERS_COVID_STABLE,
  datasets_directory,
  download_data,
  load_data,
)
from src.models.base_model import TimeSeriesModel

_SHARED_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(os.path.abspath(os.path.join(_SHARED_DIR, "..", "..", "results")))

ALL_TICKERS: list[Ticker] = sorted(
  set(
    TICKERS_2008_STABLE + TICKERS_2008_FINANCE_RESILIENT + TICKERS_2008_CRASH
    + TICKERS_COVID_STABLE + TICKERS_COVID_CRASH
    + TICKERS_2022_STABLE + TICKERS_2022_CRASH
  ),
  key=lambda t: t.value,
)

CRISIS_WINDOWS: dict[str, dict] = {
  "2008": {
    "onset": pd.Timestamp("2008-09-15"),
    "end": pd.Timestamp("2009-03-31"),
    "tickers": TICKERS_2008_CRASH,
  },
  "2020": {
    "onset": pd.Timestamp("2020-02-20"),
    "end": pd.Timestamp("2020-04-30"),
    "tickers": TICKERS_COVID_CRASH,
  },
  "2022": {
    "onset": pd.Timestamp("2022-01-03"),
    "end": pd.Timestamp("2022-12-31"),
    "tickers": TICKERS_2022_CRASH,
  },
}

REQUIRED_COLUMNS = {"date", "point", "l", "u", "r", "model", "wrapper", "ticker", "regime"}


@dataclass(frozen=True)
class ExperimentSplit:
  """One (ticker, crisis regime) experiment with train / calib / test windows.

  Windows are sequential and non-overlapping:
    train     = [train_start, train_end)
    calibrate = [calib_start, calib_end)
    test      = [test_start,  test_end ]
  """
  ticker: Ticker
  regime: str
  train_start: pd.Timestamp
  train_end: pd.Timestamp
  calib_start: pd.Timestamp
  calib_end: pd.Timestamp
  test_start: pd.Timestamp
  test_end: pd.Timestamp

  @property
  def label_col(self) -> str:
    return f"{Cols.LogReturns.value}_{self.ticker.value}"

  def slice_train(self, data: pd.DataFrame) -> pd.DataFrame:
    return data[(data.index >= self.train_start) & (data.index < self.train_end)]

  def slice_calib(self, data: pd.DataFrame) -> pd.DataFrame:
    return data[(data.index >= self.calib_start) & (data.index < self.calib_end)]

  def slice_full(self, data: pd.DataFrame) -> pd.DataFrame:
    """Train + calib + test concatenated; needed by walk_forward for history lookups."""
    return data[(data.index >= self.train_start) & (data.index <= self.test_end)]


def test_window(
  onset: pd.Timestamp,
  end: pd.Timestamp,
  pad_months: int = 6,
) -> tuple[pd.Timestamp, pd.Timestamp]:
  """Return (test_start, test_end) for a crisis, with a fixed calendar-month pad on each side."""
  return (
    onset - pd.DateOffset(months=pad_months),
    end   + pd.DateOffset(months=pad_months),
  )


def iterate_experiments(
  train_years: int = 4,
  calib_months: int = 6,
  test_pad_months: int = 6,
) -> Iterator[ExperimentSplit]:
  """Yield an ExperimentSplit for every (ticker, regime) pair in CRISIS_WINDOWS.

  Geometry, anchored at each crisis onset:

    test_start, test_end = test_window(onset, end, test_pad_months)
    calib_end   = test_start
    calib_start = calib_end - calib_months
    train_end   = calib_start
    train_start = train_end - train_years

  Fixed pad (not scaled to crisis length) gives every regime the same ~6mo
  post-crisis tail for adaptation-speed measurement. Crisis : non-crisis
  ratios are therefore unequal across regimes (e.g. 2020 is ~16% crisis,
  2022 is ~50%); metrics are reported per-regime so this doesn't bias
  cross-model comparisons.
  """
  for regime, cfg in CRISIS_WINDOWS.items():
    onset: pd.Timestamp = cfg["onset"]
    end: pd.Timestamp = cfg["end"]

    test_start, test_end = test_window(onset, end, test_pad_months)
    calib_end = test_start
    calib_start = calib_end - pd.DateOffset(months=calib_months)
    train_end = calib_start
    train_start = train_end - pd.DateOffset(years=train_years)

    for ticker in cfg["tickers"]:
      yield ExperimentSplit(
        ticker=ticker,
        regime=regime,
        train_start=train_start,
        train_end=train_end,
        calib_start=calib_start,
        calib_end=calib_end,
        test_start=test_start,
        test_end=test_end,
      )


def ensure_data(
  file_name: str = "all_tickers.csv",
  start: datetime.date = datetime.date(2003, 1, 1),
  end: datetime.date | None = None,
) -> pd.DataFrame:
  """Download the full ALL_TICKERS panel covering [start, end) if not cached, then load it.

  If the cache exists but is missing columns for any current ALL_TICKERS entry
  (e.g. because the basket list was edited since the last download), the stale
  cache is deleted and re-downloaded. Prevents the "I added a ticker but my
  notebook kept reading the old CSV" footgun.
  """
  if end is None:
    end = datetime.date.today()

  path = os.path.join(datasets_directory, file_name)

  if os.path.exists(path):
    cached_cols = set(pd.read_csv(path, nrows=0).columns)
    expected_cols = {
      f"{prefix}_{t.value}"
      for t in ALL_TICKERS
      for prefix in (Cols.Returns.value, Cols.LogReturns.value)
    }
    if expected_cols - cached_cols:
      os.remove(path)

  if not os.path.exists(path):
    download_data(tickers=ALL_TICKERS, start=start, end=end, file_name=file_name)
  return load_data(file_name)


def walk_forward(
  wrapper: IntervalPredictor,
  data: pd.DataFrame,
  test_start: pd.Timestamp,
  test_end: pd.Timestamp,
  is_adaptive: bool | None = None,
) -> pd.DataFrame:
  """Step a fitted+calibrated wrapper one trading day at a time across [test_start, test_end].

  For each day d in the test window:
    1. history = data[index < d]    (model only uses last num_lags rows internally)
    2. (point, l, u) = wrapper.predict_with_interval(history)
    3. r = realized log return on day d
    4. record (date, point, l, u, r)
    5. if wrapper has an adapt() method, call wrapper.adapt(r)

  Adaptation is duck-typed: any wrapper exposing .adapt(real_value) gets driven
  automatically (ACI, EnbPI's sliding window, AgACI experts, TCP-RM's RM offset).
  Split-CP and plain TCP have no adapt method and are left alone.

  The is_adaptive kwarg is deprecated (kept for backwards compatibility with
  existing notebooks) and is ignored — adaptation is now fully determined by
  the wrapper's interface.
  """
  del is_adaptive  # kept in signature for call-site compatibility only

  label = wrapper.label_col_name
  test_mask = (data.index >= test_start) & (data.index <= test_end)
  test_dates = data.index[test_mask]

  rows = []
  for d in test_dates:
    history = data[data.index < d]
    if len(history) < wrapper.num_lags:
      continue

    point, lower, upper = wrapper.predict_with_interval(history)
    realized = float(data.loc[d, label])

    rows.append({"date": d, "point": point, "l": lower, "u": upper, "r": realized})

    adapt_fn = getattr(wrapper, "adapt", None)
    if callable(adapt_fn):
      adapt_fn(realized)

  return pd.DataFrame(rows).set_index("date")


_DEEP_MODEL_NAMES = {"LSTMModel", "TransformerEncoderModel"}


def _resolve_wrapper_kind(wrapper_kind: str | None, step_size: float | None) -> str:
  """Back-compat bridge: notebooks that pass step_size=None/float get mapped to split/aci."""
  if wrapper_kind is not None:
    return wrapper_kind.lower()
  return "aci" if step_size is not None else "split"


def make_wrapper(
  model_class: Type[TimeSeriesModel],
  label_col: str,
  num_lags: int,
  confidence_level: float = 0.90,
  wrapper_kind: str | None = None,
  step_size: float | None = None,
  *,
  # EnbPI knobs
  B: int | None = None,
  block_size: int = 1,
  aggregation: str = "mean",
  window_length: int | None = None,
  random_state: int = 59,
  # AgACI knobs
  gamma_grid: np.ndarray | None = None,
  eta: float = 1.0,
  # TCP / TCP-RM knobs
  tcp_window: int = 252,
  tcp_calib: int = 60,
  tcp_refit_every: int = 1,
  gamma_0: float = 0.01,
  rm_lambda: float = 1.0,
  rm_beta: float = 0.6,
) -> IntervalPredictor:
  """Construct a conformal wrapper by kind.

  Preferred API: pass wrapper_kind ∈ {"split", "aci", "enbpi", "agaci", "tcp", "tcp_rm"}.
  Back-compat API: pass step_size=None (→ split) or step_size>0 (→ aci) per the
  old signature; existing notebooks work unchanged.

  Defaults to confidence_level=0.90 to match the proposal / Metrics.md target.
  """
  kind = _resolve_wrapper_kind(wrapper_kind, step_size)

  if kind == "split":
    return SplitConformal(
      model_class=model_class,
      label_col_name=label_col,
      num_lags=num_lags,
      confidence_level=confidence_level,
    )

  if kind == "aci":
    return AdaptiveConformal(
      model_class=model_class,
      label_col_name=label_col,
      num_lags=num_lags,
      confidence_level=confidence_level,
      step_size=step_size if step_size is not None else 0.01,
    )

  if kind == "enbpi":
    from src.conformal.enbpi_conformal import EnbPIConformal
    default_B = 10 if model_class.__name__ in _DEEP_MODEL_NAMES else 25
    return EnbPIConformal(
      model_class=model_class,
      label_col_name=label_col,
      num_lags=num_lags,
      confidence_level=confidence_level,
      B=B if B is not None else default_B,
      block_size=block_size,
      aggregation=aggregation,
      window_length=window_length,
      random_state=random_state,
    )

  if kind == "agaci":
    from src.conformal.agaci_conformal import AgACIConformal
    grid = gamma_grid if gamma_grid is not None else np.geomspace(0.001, 0.1, 12)
    return AgACIConformal(
      model_class=model_class,
      label_col_name=label_col,
      num_lags=num_lags,
      confidence_level=confidence_level,
      gamma_grid=grid,
      eta=eta,
    )

  if kind == "tcp":
    from src.conformal.tcp_conformal import TCPConformal
    return TCPConformal(
      model_class=model_class,
      label_col_name=label_col,
      num_lags=num_lags,
      confidence_level=confidence_level,
      window=tcp_window,
      calib=tcp_calib,
      refit_every=tcp_refit_every,
    )

  if kind == "tcp_rm":
    from src.conformal.tcp_conformal import TCPRMConformal
    return TCPRMConformal(
      model_class=model_class,
      label_col_name=label_col,
      num_lags=num_lags,
      confidence_level=confidence_level,
      window=tcp_window,
      calib=tcp_calib,
      refit_every=tcp_refit_every,
      gamma_0=gamma_0,
      rm_lambda=rm_lambda,
      rm_beta=rm_beta,
    )

  raise ValueError(
    f"Unknown wrapper_kind={wrapper_kind!r}. "
    "Expected one of: 'split', 'aci', 'enbpi', 'agaci', 'tcp', 'tcp_rm'."
  )


def save_intervals(
  df: pd.DataFrame,
  model_name: str,
  wrapper_name: str,
  ticker: Ticker | str,
  regime: str,
  results_dir: Path | str = RESULTS_DIR,
) -> Path:
  """Tag with metadata columns, validate schema, write CSV under results_dir.

  Output filename: <model>_<wrapper>_<ticker>_<regime>.csv
  """
  ticker_str = ticker.value if isinstance(ticker, Ticker) else ticker

  out = df.reset_index() if df.index.name == "date" else df.copy()
  out["model"] = model_name
  out["wrapper"] = wrapper_name
  out["ticker"] = ticker_str
  out["regime"] = regime

  missing = REQUIRED_COLUMNS - set(out.columns)
  if missing:
    raise ValueError(f"save_intervals: missing required columns {missing}")

  results_dir = Path(results_dir)
  results_dir.mkdir(parents=True, exist_ok=True)
  path = results_dir / f"{model_name}_{wrapper_name}_{ticker_str}_{regime}.csv"
  out.to_csv(path, index=False)
  return path


def load_all_intervals(results_dir: Path | str = RESULTS_DIR) -> pd.DataFrame:
  """Concatenate every interval CSV under results_dir. Validates schema per file."""
  results_dir = Path(results_dir)
  # Interval CSVs follow the pattern <model>_<wrapper>_<ticker>_<regime>.csv,
  # i.e. >=4 underscore-separated tokens. The aggregator's metrics_long.csv
  # (and any other ad-hoc summaries) live in the same directory but have
  # different schemas — skip them by name.
  files = [
    f for f in sorted(results_dir.glob("*.csv"))
    if len(f.stem.split("_")) >= 4 and f.name != "metrics_long.csv"
  ]
  if not files:
    raise FileNotFoundError(f"No interval CSVs found in {results_dir}")

  dfs = []
  for f in files:
    df = pd.read_csv(f, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
      raise ValueError(f"{f.name} missing required columns: {missing}")
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)
