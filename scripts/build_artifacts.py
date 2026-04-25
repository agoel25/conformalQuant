"""Aggregate the 180-CSV grid into paper artifacts.

Runnable from the project root with `python scripts/build_artifacts.py`.

Reads every interval CSV under `results/`, computes the five Metrics.md
metrics per (model, wrapper, ticker, regime) slice, and writes:

    results/metrics_long.csv          one row per slice with every metric
    report/figures/scatter_cov_lo_hi.{pdf,png}    Cov_low vs Cov_high scatter
    report/figures/rolling_coverage.{pdf,png}     three-panel rolling-coverage figure

Same logic as `src/analysis/results.ipynb`; this script is the headless
equivalent for CI / fresh-clone reproduction.
"""
from __future__ import annotations

import glob
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.analysis._shared import CRISIS_WINDOWS, RESULTS_DIR
from src.metrics.metrics import Metrics
from src.metrics.prediction_cols import PredictionCols


ALPHA = 0.10  # matches src/metrics/Metrics.md


WRAPPER_COLORS = {
  "split": "#d62728", "aci": "#2ca02c", "enbpi": "#9467bd",
  "agaci": "#1f77b4", "tcp": "#ff7f0e", "tcp_rm": "#e377c2", "native": "#7f7f7f",
}
MODEL_MARKERS = {
  "linear_ar": "o", "hist_binning": "s", "tree_based": "^",
  "lstm": "D", "transformer": "P", "fhs": "X", "garch_roll": "H",
}
WRAPPER_NAME = {
  "split": "Split", "aci": "ACI", "enbpi": "EnbPI", "agaci": "AgACI",
  "native": "Native",
}
MODEL_NAME = {
  "linear_ar": "LinAR",
  "hist_binning": "HistBin",
  "tree_based": "QTree",
  "lstm": "LSTM",
  "transformer": "Tx",
  "fhs": "FHS",
  "garch_roll": "GARCH(rolling)",
}

# Four-way categorisation matching src/analysis/_shared.MODEL_CATEGORIES:
#   baseline      — FHS, GARCH (rolling): the comparison targets
#   simple_native — HistBin and quantile-tree native intervals (no CP)
#   simple_cp     — LinearAR wrapped with CP
#   complex_cp    — LSTM / Transformer wrapped with CP
MODEL_CATEGORY: dict[str, str] = {
  "fhs": "baseline",
  "garch_roll": "baseline",
  "hist_binning": "simple_native",
  "tree_based": "simple_native",
  "linear_ar": "simple_cp",
  "lstm": "complex_cp",
  "transformer": "complex_cp",
}


def _load_intervals(results_dir: Path) -> pd.DataFrame:
  """Load every interval CSV (skipping aggregate files like metrics_long.csv)."""
  files = [
    f for f in sorted(glob.glob(str(results_dir / "*.csv")))
    if not f.endswith("metrics_long.csv")
  ]
  if not files:
    raise FileNotFoundError(f"No interval CSVs in {results_dir}")
  raw = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
  raw["regime"] = raw["regime"].astype(str)
  return raw


def _metrics_for_slice(slice_df: pd.DataFrame, regime: str) -> dict | None:
  s = slice_df.rename(columns={
    "r": PredictionCols.LOG_RETURNS,
    "l": PredictionCols.LOWER_BOUND,
    "u": PredictionCols.UPPER_BOUND,
  }).set_index("date").sort_index()

  widths = s[PredictionCols.UPPER_BOUND] - s[PredictionCols.LOWER_BOUND]
  n_inf = int(np.isinf(widths).sum())
  s_finite = s[~np.isinf(widths)]
  if len(s_finite) == 0:
    return None

  m = Metrics(s_finite, alpha=ALPHA)
  cov_high, cov_low, gap = m.get_coverage_metrics().get_high_low_vol_coverage()
  w_overall, w_high, w_low = m.get_winkler_metrics().get_mean_scores()
  width_overall, width_high, width_low = m.get_interval_metrics().get_mean_widths()

  cfg = CRISIS_WINDOWS[regime]
  crisis_cov = float(m.get_coverage_metrics().get_period_coverage(cfg["onset"], cfg["end"]))
  crisis_win = float(m.get_winkler_metrics().get_period_score(cfg["onset"], cfg["end"]))
  crisis_wid = float(m.get_interval_metrics().get_period_width(cfg["onset"], cfg["end"]))
  T_rec, max_dd, recovered = m.get_adaptation_metrics().compute_adaptation_speed(
    cfg["onset"], cfg["end"], threshold=0.85, persistence=5, window=20,
  )

  return {
    "n": len(s_finite),
    "inf_days": n_inf,
    "marg_cov": float(m.get_coverage_metrics().get_marginal_coverage()),
    "cov_high": float(cov_high),
    "cov_low": float(cov_low),
    "gap": float(gap),
    "winkler": float(w_overall),
    "w_high": float(w_high),
    "w_low": float(w_low),
    "width": float(width_overall),
    "width_high": float(width_high),
    "width_low": float(width_low),
    "crisis_cov": crisis_cov,
    "crisis_win": crisis_win,
    "crisis_wid": crisis_wid,
    "t_recovery": int(T_rec),
    "max_drawdown_cov": float(max_dd) if max_dd is not None and not np.isnan(max_dd) else float("nan"),
    "recovered": bool(recovered),
  }


def build_metrics_long(raw: pd.DataFrame, out_path: Path) -> pd.DataFrame:
  rows = []
  for (model, wrapper, ticker, regime), g in raw.groupby(["model", "wrapper", "ticker", "regime"]):
    m = _metrics_for_slice(g, regime)
    if m is None:
      continue
    rows.append({
      "model": model,
      "wrapper": wrapper,
      "category": MODEL_CATEGORY.get(model, "unknown"),
      "ticker": ticker,
      "regime": regime,
      **m,
    })
  long = pd.DataFrame(rows)
  long.to_csv(out_path, index=False)
  print(f"wrote {len(long)} rows -> {out_path}")
  return long


def build_scatter(long: pd.DataFrame, out_dir: Path) -> None:
  sc = long.groupby(["model", "wrapper"]).agg(
    cov_low=("cov_low", "mean"),
    cov_high=("cov_high", "mean"),
  ).reset_index()

  fig, ax = plt.subplots(figsize=(9, 6))
  order = sc.assign(
    wp=sc["wrapper"].map({"tcp": 2, "tcp_rm": 3}).fillna(1).astype(int)
  ).sort_values("wp").reset_index(drop=True)
  for _, row in order.iterrows():
    ax.scatter(
      row["cov_low"], row["cov_high"],
      color=WRAPPER_COLORS.get(row["wrapper"], "black"),
      marker=MODEL_MARKERS.get(row["model"], "o"),
      s=150, edgecolor="black", linewidth=0.6, zorder=3,
    )
  lims = [0.4, 1.0]
  ax.plot(lims, lims, "k--", alpha=0.35, zorder=1, label="y = x")
  ax.axhline(0.90, color="gray", linestyle=":", alpha=0.4, zorder=1)
  ax.axvline(0.90, color="gray", linestyle=":", alpha=0.4, zorder=1)
  ax.text(0.99, 0.91, "nominal = 0.90", ha="right", fontsize=8, color="gray")
  ax.set_xlim(lims); ax.set_ylim(lims)
  ax.set_xlabel("Coverage on low-volatility days", fontsize=11)
  ax.set_ylabel("Coverage on high-volatility days", fontsize=11)
  ax.grid(True, alpha=0.2)

  color_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black",
           markersize=10, label=WRAPPER_NAME[w]) for w, c in WRAPPER_COLORS.items()
  ]
  marker_handles = [
    Line2D([0], [0], marker=m, color="gray", markeredgecolor="black", markersize=9,
           linestyle="", label=MODEL_NAME[mdl]) for mdl, m in MODEL_MARKERS.items()
  ]
  leg1 = ax.legend(
    handles=color_handles, title="Wrapper (color)",
    loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, title_fontsize=9, frameon=True,
  )
  ax.add_artist(leg1)
  ax.legend(
    handles=marker_handles, title="Model (shape)",
    loc="upper left", bbox_to_anchor=(1.02, 0.55), fontsize=9, title_fontsize=9, frameon=True,
  )
  fig.tight_layout()
  fig.savefig(out_dir / "scatter_cov_lo_hi.pdf", bbox_inches="tight")
  fig.savefig(out_dir / "scatter_cov_lo_hi.png", dpi=140, bbox_inches="tight")
  plt.close(fig)
  print(f"wrote scatter -> {out_dir}/scatter_cov_lo_hi.{{pdf,png}}")


def build_rolling_coverage(raw: pd.DataFrame, out_dir: Path) -> None:
  representative = {"2008": "AIG", "2020": "MCD", "2022": "NFLX"}
  fig, axes = plt.subplots(3, 1, figsize=(9, 9))
  for ax, (regime, ticker) in zip(axes, representative.items()):
    onset, end = CRISIS_WINDOWS[regime]["onset"], CRISIS_WINDOWS[regime]["end"]
    sl = raw[
      (raw["regime"] == regime) & (raw["ticker"] == ticker)
      & raw["model"].isin(["lstm", "transformer", "fhs", "garch", "garch_roll"])
    ]
    for (model, wrapper), g in sl.groupby(["model", "wrapper"]):
      g = g.sort_values("date")
      covered = ((g["r"] >= g["l"]) & (g["r"] <= g["u"])).astype(float)
      rolling = covered.rolling(20).mean()
      color = WRAPPER_COLORS.get(wrapper, "black")
      ls = "-" if wrapper in ("tcp", "tcp_rm") else ("--" if wrapper == "aci" else ":")
      lw = 1.2 if wrapper in ("tcp", "tcp_rm") else 0.8
      ax.plot(
        g["date"], rolling, color=color, linewidth=lw, alpha=0.85, linestyle=ls,
        label=f"{MODEL_NAME[model]}/{WRAPPER_NAME[wrapper]}",
      )
    ax.axvspan(onset, end, alpha=0.12, color="red")
    ax.axhline(0.90, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_ylabel("Rolling 20-day coverage", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{regime} crisis - {ticker}", fontsize=10)
    ax.legend(fontsize=6, loc="lower right", ncols=3, framealpha=0.9)
  fig.tight_layout()
  fig.savefig(out_dir / "rolling_coverage.pdf", bbox_inches="tight")
  fig.savefig(out_dir / "rolling_coverage.png", dpi=130, bbox_inches="tight")
  plt.close(fig)
  print(f"wrote rolling-coverage -> {out_dir}/rolling_coverage.{{pdf,png}}")


def main() -> None:
  warnings.filterwarnings("ignore")

  results_dir = Path(RESULTS_DIR)
  figures_dir = ROOT / "report" / "figures"
  figures_dir.mkdir(parents=True, exist_ok=True)

  raw = _load_intervals(results_dir)
  print(f"loaded {len(raw):,} rows across {raw.groupby(['model','wrapper','ticker','regime']).ngroups} slices")

  long = build_metrics_long(raw, results_dir / "metrics_long.csv")
  build_scatter(long, figures_dir)
  build_rolling_coverage(raw, figures_dir)

  print("\nDone. To rebuild the report PDF, run:")
  print("    cd report && pdflatex report.tex && biber report && pdflatex report.tex && pdflatex report.tex")


if __name__ == "__main__":
  main()
