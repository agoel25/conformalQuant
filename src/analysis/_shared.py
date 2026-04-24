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

import pandas as pd

from src.conformal.adaptive_conformal import AdaptiveConformal
from src.conformal.base_conformal import BaseConformal
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
    "tickers": TICKERS_2008_STABLE + TICKERS_2008_FINANCE_RESILIENT + TICKERS_2008_CRASH,
  },
  "2020": {
    "onset": pd.Timestamp("2020-02-20"),
    "end": pd.Timestamp("2020-04-30"),
    "tickers": TICKERS_COVID_STABLE + TICKERS_COVID_CRASH,
  },
  "2022": {
    "onset": pd.Timestamp("2022-03-16"),
    "end": pd.Timestamp("2022-12-31"),
    "tickers": TICKERS_2022_STABLE + TICKERS_2022_CRASH,
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


def iterate_experiments(
  train_years: int = 4,
  calib_months: int = 6,
  test_pad_months: int = 6,
) -> Iterator[ExperimentSplit]:
  """Yield an ExperimentSplit for every (ticker, regime) pair in CRISIS_WINDOWS.

  Geometry, anchored at each crisis onset:
    test_start  = onset - test_pad_months
    calib_end   = test_start
    calib_start = calib_end - calib_months
    train_end   = calib_start
    train_start = train_end - train_years
    test_end    = crisis_end + test_pad_months
  """
  for regime, cfg in CRISIS_WINDOWS.items():
    onset: pd.Timestamp = cfg["onset"]
    end: pd.Timestamp = cfg["end"]

    test_start = onset - pd.DateOffset(months=test_pad_months)
    test_end = end + pd.DateOffset(months=test_pad_months)
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
  """Download the full ALL_TICKERS panel covering [start, end) if not cached, then load it."""
  if end is None:
    end = datetime.date.today()

  path = os.path.join(datasets_directory, file_name)
  if not os.path.exists(path):
    download_data(tickers=ALL_TICKERS, start=start, end=end, file_name=file_name)
  return load_data(file_name)


def walk_forward(
  wrapper: BaseConformal,
  data: pd.DataFrame,
  test_start: pd.Timestamp,
  test_end: pd.Timestamp,
  is_adaptive: bool = False,
) -> pd.DataFrame:
  """Step a fitted+calibrated wrapper one trading day at a time across [test_start, test_end].

  For each day d in the test window:
    1. history = data[index < d]    (model only uses last num_lags rows internally)
    2. (point, l, u) = wrapper.predict_with_interval(history)
    3. r = realized log return on day d
    4. record (date, point, l, u, r)
    5. if is_adaptive: wrapper.adapt(r)   (ACI only; nudges alpha for the next day)

  Skips days where history is shorter than num_lags (early in the test window
  this should never trigger because train+calib provide ample history, but the
  guard keeps the function safe to use on truncated data).
  """
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

    if is_adaptive:
      assert isinstance(wrapper, AdaptiveConformal)
      wrapper.adapt(realized)

  return pd.DataFrame(rows).set_index("date")


def make_wrapper(
  model_class: Type[TimeSeriesModel],
  label_col: str,
  num_lags: int,
  confidence_level: float = 0.90,
  step_size: float | None = None,
) -> BaseConformal:
  """Construct SplitConformal (step_size=None) or AdaptiveConformal (step_size>0).

  Defaults to confidence_level=0.90 to match the proposal / Metrics.md target,
  not BaseConformal's 0.95 default.
  """
  if step_size is None:
    return SplitConformal(
      model_class=model_class,
      label_col_name=label_col,
      num_lags=num_lags,
      confidence_level=confidence_level,
    )
  return AdaptiveConformal(
    model_class=model_class,
    label_col_name=label_col,
    num_lags=num_lags,
    confidence_level=confidence_level,
    step_size=step_size,
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
  files = sorted(results_dir.glob("*.csv"))
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
