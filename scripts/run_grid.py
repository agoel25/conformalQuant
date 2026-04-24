"""Reproduce the full 180-CSV evaluation grid end-to-end.

One file, runnable from the project root with `python scripts/run_grid.py`.
Same code path as the notebooks under `src/analysis/`, just consolidated for
headless reproduction.

Grid topology (180 runs total):
    Simple models     (linear_ar, hist_binning, tree_based)  x  {split, aci}
    Deep models       (lstm, transformer)                    x  {split, aci, enbpi, agaci, tcp, tcp_rm}
    Industry baselines (fhs, garch)                          x  {native}
    each over 9 (ticker, regime) pairs from `iterate_experiments()`.

CSVs land in `results/<model>_<wrapper>_<ticker>_<regime>.csv` and are picked
up by `scripts/build_artifacts.py` to produce the aggregator tables + paper
figures.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

# Make `src.*` importable when run from the project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.analysis._shared import (
  ensure_data,
  iterate_experiments,
  walk_forward,
  make_wrapper,
  save_intervals,
)
from src.conformal.fhs import FHS
from src.conformal.garch import Garch
from src.models.histogram_binning import HistogramBinningModel
from src.models.linear_ar import LinearARModel
from src.models.lstm import LSTMModel
from src.models.transformer_encoder import TransformerModel
from src.models.tree_based import TreeBasedModel


# Hyperparameters mirror the per-model notebooks.
NUM_LAGS = 5
CONFIDENCE_LEVEL = 0.90  # alpha = 0.10, matches src/metrics/Metrics.md
ACI_STEP_SIZE = 0.01
TCP_REFIT_EVERY = 5  # lower = more faithful to the paper, higher = cheaper
MIN_TRAIN_ROWS = NUM_LAGS + 50  # skip splits with insufficient pre-IPO history

SIMPLE_MODELS: list[tuple[str, type]] = [
  ("linear_ar", LinearARModel),
  ("hist_binning", HistogramBinningModel),
  ("tree_based", TreeBasedModel),
]
DEEP_MODELS: list[tuple[str, type]] = [
  ("lstm", LSTMModel),
  ("transformer", TransformerModel),
]
SIMPLE_WRAPPERS = ["split", "aci"]
DEEP_WRAPPERS = ["split", "aci", "enbpi", "agaci", "tcp", "tcp_rm"]


def _run_wrapper_grid(
  data: pd.DataFrame,
  model_name: str,
  model_cls: type,
  wrapper_kinds: list[str],
) -> None:
  print(f"########## {model_name} ##########")
  t_model = time.perf_counter()

  for split in iterate_experiments():
    train_df = split.slice_train(data)
    calib_df = split.slice_calib(data)
    full_df = split.slice_full(data)

    if len(train_df) < MIN_TRAIN_ROWS:
      print(f"  skip {split.regime} {split.ticker.value}: only {len(train_df)} train rows")
      continue

    for kind in wrapper_kinds:
      np.random.seed(0)
      torch.manual_seed(0)
      t0 = time.perf_counter()
      wrapper = make_wrapper(
        model_cls,
        label_col=split.label_col,
        num_lags=NUM_LAGS,
        confidence_level=CONFIDENCE_LEVEL,
        wrapper_kind=kind,
        step_size=ACI_STEP_SIZE,
        tcp_refit_every=TCP_REFIT_EVERY,
      )
      wrapper.fit_and_calibrate(train_df, calib_df)
      intervals = walk_forward(wrapper, full_df, split.test_start, split.test_end)
      save_intervals(intervals, model_name, kind, split.ticker, split.regime)

      cov = ((intervals.r >= intervals.l) & (intervals.r <= intervals.u)).mean()
      mean_w = (intervals.u - intervals.l).mean()
      print(
        f"  {split.regime} {split.ticker.value:5s} {kind:7s}: "
        f"cov={cov:.3f}  mean_w={mean_w:.4f}  ({time.perf_counter() - t0:.1f}s)"
      )

  print(f"  --- {model_name} elapsed: {(time.perf_counter() - t_model) / 60:.1f} min ---")


def _run_baselines(data: pd.DataFrame) -> None:
  """FHS and GARCH don't follow the BaseConformal model_class contract."""
  for name, cls in [("fhs", FHS), ("garch", Garch)]:
    print(f"########## {name} (native) ##########")
    t_model = time.perf_counter()
    skipped: list[tuple[str, str]] = []

    for split in iterate_experiments():
      train_df = split.slice_train(data)
      calib_df = split.slice_calib(data)
      full_df = split.slice_full(data)

      try:
        if name == "fhs":
          baseline = FHS(label_col_name=split.label_col, confidence_level=CONFIDENCE_LEVEL)
          baseline.fit_and_calibrate(pd.concat([train_df, calib_df]))
        else:
          baseline = Garch(
            label_col_name=split.label_col, num_lags=1, confidence_level=CONFIDENCE_LEVEL,
          )
          baseline.fit_and_calibrate(train_df, calib_df)

        intervals = walk_forward(baseline, full_df, split.test_start, split.test_end)
        save_intervals(intervals, name, "native", split.ticker, split.regime)
        cov = ((intervals.r >= intervals.l) & (intervals.r <= intervals.u)).mean()
        mean_w = (intervals.u - intervals.l).mean()
        print(f"  {split.regime} {split.ticker.value:5s}: cov={cov:.3f}  mean_w={mean_w:.4f}")
      except Exception as e:
        skipped.append((split.regime, split.ticker.value))
        # Remove any stale CSV so the aggregator doesn't see a non-converged fit.
        stale = Path(ROOT) / "results" / f"{name}_native_{split.ticker.value}_{split.regime}.csv"
        if stale.exists():
          stale.unlink()
        print(f"  {split.regime} {split.ticker.value:5s}: SKIPPED ({type(e).__name__}: {e})")

    if skipped:
      print(f"  skipped: {skipped}")
    print(f"  --- {name} elapsed: {(time.perf_counter() - t_model) / 60:.1f} min ---")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--only",
    choices=("simple", "deep", "baselines", "all"),
    default="all",
    help="Subset of the grid to run (default: all)",
  )
  args = parser.parse_args()

  warnings.filterwarnings("ignore")
  data = ensure_data()
  print(f"data: {len(data):,} rows  |  {data.index.min().date()} -> {data.index.max().date()}")

  t_global = time.perf_counter()

  if args.only in ("all", "simple"):
    for model_name, model_cls in SIMPLE_MODELS:
      _run_wrapper_grid(data, model_name, model_cls, SIMPLE_WRAPPERS)

  if args.only in ("all", "deep"):
    for model_name, model_cls in DEEP_MODELS:
      _run_wrapper_grid(data, model_name, model_cls, DEEP_WRAPPERS)

  if args.only in ("all", "baselines"):
    _run_baselines(data)

  print(f"\nTOTAL elapsed: {(time.perf_counter() - t_global) / 60:.1f} min")


if __name__ == "__main__":
  main()
