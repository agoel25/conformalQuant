"""Reproduce the full evaluation grid end-to-end.

One file, runnable from the project root with `python scripts/run_grid.py`.
Same code path as the notebooks under `src/analysis/`, just consolidated for
headless reproduction.

Grid topology (model categories):
  Baselines      (fhs, garch, garch_roll)            x  native
  Simple_Native  (hist_binning, tree_based)           x  native   ← no CP
  Simple_CP      (linear_ar)                          x  {split, aci}
  Complex_CP     (lstm, transformer)                  x  {split, aci, enbpi, agaci, tcp, tcp_rm}
  each over 9 (ticker, regime) pairs from iterate_experiments().

All adaptive methods (TCP, rolling GARCH) share a single REFIT_EVERY cadence
(≈ monthly) for an apples-to-apples comparison.

CSVs land in results/<model>_<wrapper>_<ticker>_<regime>.csv and are picked
up by scripts/build_artifacts.py to produce aggregator tables + paper figures.

Run modes
---------
  --only {baselines,simple_native,simple_cp,complex_cp,all}
  --sanity   Fast subset: baselines + simple_native + linear_ar×split + lstm×tcp
             on 1 ticker per regime, quarterly refit cadence (63 steps).
             Use this to verify rolling-GARCH vs TCP direction before the full run.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.analysis._shared import (
  CRISIS_WINDOWS,
  Ticker,
  ensure_data,
  iterate_experiments,
  walk_forward,
  make_wrapper,
  save_intervals,
)
from src.conformal.fhs import FHS
from src.conformal.garch import GarchHardcoded, RollingGarch
from src.conformal.hist_binning_native import HistogramBinningNative
from src.conformal.tcp_paper import TCPPaperConformal
from src.conformal.tree_native import QuantileTreeNative
from src.models.linear_ar import LinearARModel
from src.models.lstm import LSTMModel
from src.models.transformer_encoder import TransformerModel

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

NUM_LAGS = 5
CONFIDENCE_LEVEL = 0.90  # alpha = 0.10, matches src/metrics/Metrics.md
ACI_STEP_SIZE = 0.01
# Single refit cadence used by EVERY adaptive method (TCP, rolling GARCH) so
# that the comparison is apples-to-apples: 21 trading days ≈ one calendar month.
REFIT_EVERY = 21
MIN_TRAIN_ROWS = NUM_LAGS + 50  # skip splits with insufficient pre-IPO history

SIMPLE_CP_MODELS: list[tuple[str, type]] = [
  ("linear_ar", LinearARModel),
]
COMPLEX_CP_MODELS: list[tuple[str, type]] = [
  ("lstm", LSTMModel),
  ("transformer", TransformerModel),
]
SIMPLE_CP_WRAPPERS = ["split", "aci", "agaci"]
COMPLEX_CP_WRAPPERS = ["split", "aci", "agaci"]
# TCP / TCP-RM excluded from main grid (see TCP_Paper sanity for why);
# EnbPI also dropped to keep CPU runtime tractable.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _splits_for(ticker_filter: set[str] | None = None):
  """Yield ExperimentSplit, optionally restricting to a set of ticker names."""
  for split in iterate_experiments():
    if ticker_filter is None or split.ticker.value in ticker_filter:
      yield split


def _run_cp_grid(
  data: pd.DataFrame,
  model_name: str,
  model_cls: type,
  wrapper_kinds: list[str],
  ticker_filter: set[str] | None = None,
  refit_every: int | None = None,
) -> None:
  """Run a CP-wrapped model over the experiment grid."""
  _refit = refit_every if refit_every is not None else REFIT_EVERY
  print(f"########## {model_name} ##########")
  t_model = time.perf_counter()

  for split in _splits_for(ticker_filter):
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


def _run_native_grid(
  data: pd.DataFrame,
  native_specs: list[tuple[str, object]],
  ticker_filter: set[str] | None = None,
  refit_every: int | None = None,
) -> None:
  """Run native-interval models (baselines + simple_native). Each spec is
  (name, constructor_or_instance_factory) where calling the factory with
  (label_col_name, confidence_level, refit_every?) returns a fitted-ready
  IntervalPredictor."""
  _refit = refit_every if refit_every is not None else REFIT_EVERY
  for name, factory in native_specs:
    print(f"########## {name} (native) ##########")
    t_model = time.perf_counter()
    skipped: list[tuple[str, str]] = []

    for split in _splits_for(ticker_filter):
      train_df = split.slice_train(data)
      calib_df = split.slice_calib(data)
      full_df = split.slice_full(data)

      try:
        baseline = factory(split.label_col, _refit)
        baseline.fit_and_calibrate(train_df, calib_df)
        intervals = walk_forward(baseline, full_df, split.test_start, split.test_end)
        save_intervals(intervals, name, "native", split.ticker, split.regime)
        cov = ((intervals.r >= intervals.l) & (intervals.r <= intervals.u)).mean()
        mean_w = (intervals.u - intervals.l).mean()
        print(f"  {split.regime} {split.ticker.value:5s}: cov={cov:.3f}  mean_w={mean_w:.4f}")
      except Exception as e:
        skipped.append((split.regime, split.ticker.value))
        stale = Path(ROOT) / "results" / f"{name}_native_{split.ticker.value}_{split.regime}.csv"
        if stale.exists():
          stale.unlink()
        print(f"  {split.regime} {split.ticker.value:5s}: SKIPPED ({type(e).__name__}: {e})")

    if skipped:
      print(f"  skipped: {skipped}")
    print(f"  --- {name} elapsed: {(time.perf_counter() - t_model) / 60:.1f} min ---")


# ---------------------------------------------------------------------------
# Native model factories  (label_col, refit_every) -> IntervalPredictor
# ---------------------------------------------------------------------------

def _fhs_factory(label_col, _refit):
  m = FHS(label_col_name=label_col, confidence_level=CONFIDENCE_LEVEL)
  return m

# FHS overrides fit_and_calibrate to take a single combined df — patch via closure
class _FHSWrapper:
  """Thin shim: fit_and_calibrate(train, calib) → concatenates and calls FHS."""
  def __init__(self, label_col):
    self._inner = FHS(label_col_name=label_col, confidence_level=CONFIDENCE_LEVEL)
    self.num_lags = self._inner.num_lags
    self.label_col_name = label_col

  def fit_and_calibrate(self, train_data, calib_data=None):
    combined = pd.concat([train_data, calib_data]) if calib_data is not None else train_data
    self._inner.fit_and_calibrate(combined)

  def predict_with_interval(self, data):
    return self._inner.predict_with_interval(data)

  # Duck-typed adapt (FHS has none — this is fine, walk_forward checks hasattr)


def _make_native_specs(refit_every: int) -> list[tuple[str, object]]:
  """Return the (name, factory) list for all native-interval models."""
  return [
    ("fhs",        lambda lc, re: _FHSWrapper(lc)),
    ("garch_roll", lambda lc, re: RollingGarch(label_col_name=lc, num_lags=1,
                                               confidence_level=CONFIDENCE_LEVEL,
                                               garch_window=504, refit_every=re)),
    ("hist_binning", lambda lc, re: HistogramBinningNative(label_col_name=lc, num_lags=NUM_LAGS,
                                                            confidence_level=CONFIDENCE_LEVEL)),
    ("tree_based",   lambda lc, re: QuantileTreeNative(label_col_name=lc, num_lags=NUM_LAGS,
                                                        confidence_level=CONFIDENCE_LEVEL)),
  ]


# ---------------------------------------------------------------------------
# Sanity check: 1 ticker per regime, reduced model set, faster refit cadence
# ---------------------------------------------------------------------------

# One representative ticker per regime (chosen to be liquid / well-known)
_SANITY_TICKERS: set[str] = {
  cfg["tickers"][0].value for cfg in CRISIS_WINDOWS.values()
}
_SANITY_REFIT = 63  # quarterly — speeds up rolling GARCH & TCP without changing direction


def _run_tcp_paper_sanity(data: pd.DataFrame, ticker_filter: set[str], refit_every: int) -> None:
  """Run the three-way TCP-paper sanity comparison on a ticker subset.

  Compares:
    garch_hardcoded — TCP paper's exact GARCH (hardcoded ω,α,β, normal dist, no MLE)
    garch_roll      — properly estimated rolling GARCH
    tcp_paper       — paper-faithful TCP (quantile GBM + rolling split-conformal)

  Expected ordering in Winkler (lower = better):
    garch_hardcoded  >  garch_roll  ~  tcp_paper
  This demonstrates that rolling GARCH closes most of the gap TCP claimed over GARCH.
  """
  tcp_paper_specs = [
    ("garch_hardcoded", lambda lc, re: GarchHardcoded(label_col_name=lc, confidence_level=CONFIDENCE_LEVEL)),
    ("garch_roll",      lambda lc, re: RollingGarch(label_col_name=lc, num_lags=1,
                                                    confidence_level=CONFIDENCE_LEVEL,
                                                    garch_window=504, refit_every=re)),
    ("tcp_paper",       lambda lc, re: TCPPaperConformal(label_col_name=lc, num_lags=NUM_LAGS,
                                                          confidence_level=CONFIDENCE_LEVEL,
                                                          refit_every=re)),
  ]
  _run_native_grid(data, tcp_paper_specs, ticker_filter, refit_every)


def _run_sanity(data: pd.DataFrame) -> None:
  """Quick verification pass: all native models + linear_ar×split + TCP paper sanity.

  Restricted to 1 ticker per regime and quarterly refit cadence.
  """
  print(f"\n{'='*60}")
  print(f"SANITY CHECK  |  tickers: {_SANITY_TICKERS}  |  refit_every={_SANITY_REFIT}")
  print(f"{'='*60}\n")

  t0 = time.perf_counter()

  # All native baselines (fhs, garch, garch_roll, hist_binning, tree_based)
  _run_native_grid(data, _make_native_specs(_SANITY_REFIT), _SANITY_TICKERS, _SANITY_REFIT)

  # Simple CP
  _run_cp_grid(data, "linear_ar", LinearARModel, ["split"],
               ticker_filter=_SANITY_TICKERS)

  # TCP paper sanity: garch_hardcoded vs garch_roll vs tcp_paper
  print(f"\n--- TCP Paper sanity (garch_hardcoded / garch_roll / tcp_paper) ---")
  _run_tcp_paper_sanity(data, _SANITY_TICKERS, _SANITY_REFIT)

  elapsed = (time.perf_counter() - t0) / 60
  print(f"\nSanity check elapsed: {elapsed:.1f} min")
  print("\nQuick Winkler summary (lower = better):")
  _print_sanity_summary()


def _print_sanity_summary() -> None:
  """Print per-(model,wrapper) mean coverage and Winkler from sanity CSVs."""
  import glob
  results_dir = ROOT / "results"
  files = [
    f for f in sorted(glob.glob(str(results_dir / "*.csv")))
    if not f.endswith("metrics_long.csv")
  ]
  if not files:
    print("  (no result CSVs found)")
    return

  from src.metrics.metrics import Metrics
  from src.metrics.prediction_cols import PredictionCols

  rows = []
  for f in files:
    df = pd.read_csv(f, parse_dates=["date"])
    # only sanity tickers
    if df["ticker"].iloc[0] not in _SANITY_TICKERS:
      continue
    df2 = df.rename(columns={"r": PredictionCols.LOG_RETURNS,
                              "l": PredictionCols.LOWER_BOUND,
                              "u": PredictionCols.UPPER_BOUND}).set_index("date")
    widths = df2[PredictionCols.UPPER_BOUND] - df2[PredictionCols.LOWER_BOUND]
    df2 = df2[~np.isinf(widths)]
    if len(df2) == 0:
      continue
    m = Metrics(df2, alpha=0.10)
    cov = m.get_coverage_metrics().get_marginal_coverage()
    winkler, _, _ = m.get_winkler_metrics().get_mean_scores()
    rows.append({
      "model": df["model"].iloc[0],
      "wrapper": df["wrapper"].iloc[0],
      "ticker": df["ticker"].iloc[0],
      "cov": cov,
      "winkler": winkler,
    })

  if not rows:
    print("  (no sanity rows to summarise)")
    return

  summary = (
    pd.DataFrame(rows)
    .groupby(["model", "wrapper"])[["cov", "winkler"]]
    .mean()
    .sort_values("winkler")
    .reset_index()
  )
  print(summary.to_string(index=False, float_format="{:.3f}".format))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--only",
    choices=("baselines", "simple_native", "simple_cp", "complex_cp", "tcp_paper", "all"),
    default="all",
    help="Subset of the grid to run (default: all)",
  )
  parser.add_argument(
    "--sanity",
    action="store_true",
    help="Fast verification pass (1 ticker/regime, quarterly refit). "
         "Run this first to confirm rolling-GARCH vs TCP direction.",
  )
  args = parser.parse_args()

  warnings.filterwarnings("ignore")
  data = ensure_data()
  print(f"data: {len(data):,} rows  |  {data.index.min().date()} -> {data.index.max().date()}")

  if args.sanity:
    _run_sanity(data)
    return

  t_global = time.perf_counter()
  native_specs = _make_native_specs(REFIT_EVERY)

  if args.only in ("all", "baselines"):
    _run_native_grid(data, [s for s in native_specs if s[0] in ("fhs", "garch_roll")])

  if args.only in ("all", "simple_native"):
    _run_native_grid(data, [s for s in native_specs if s[0] in ("hist_binning", "tree_based")])

  if args.only in ("all", "simple_cp"):
    for model_name, model_cls in SIMPLE_CP_MODELS:
      _run_cp_grid(data, model_name, model_cls, SIMPLE_CP_WRAPPERS)

  if args.only in ("all", "complex_cp"):
    for model_name, model_cls in COMPLEX_CP_MODELS:
      _run_cp_grid(data, model_name, model_cls, COMPLEX_CP_WRAPPERS)

  if args.only == "tcp_paper":
    _run_tcp_paper_sanity(data, ticker_filter=None, refit_every=REFIT_EVERY)

  print(f"\nTOTAL elapsed: {(time.perf_counter() - t_global) / 60:.1f} min")


if __name__ == "__main__":
  main()
