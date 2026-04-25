"""Aggregate the interval CSV grid into paper artifacts.

Reads every interval CSV under results/, computes per-cell metrics via
src.plotting.compute_cell_metrics_full (which delegates entirely to
src.metrics.Metrics — single source of truth), and writes:

    results/metrics_long.csv
    report/figures/winkler_comparison.{pdf,png}
    report/figures/cov_scatter.{pdf,png}
    report/figures/width_adaptation.{pdf,png}
    report/figures/rolling_coverage.{pdf,png}

Run from the project root:
    python scripts/build_artifacts.py
"""
from __future__ import annotations

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

from src.analysis._shared import CRISIS_WINDOWS, RESULTS_DIR, MODEL_CATEGORIES
from src.plotting import CATEGORY_COLORS, compute_cell_metrics_full

ALPHA = 0.10
FIGURES_DIR = ROOT / "report" / "figures"

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_METHOD_LABEL = {
  "fhs": "FHS", "garch_roll": "GARCH-Roll",
  "hist_binning": "Hist-Bin", "tree_based": "QRF",
  "linear_ar": "LinAR", "lstm": "LSTM", "transformer": "Tx",
}
_WRAPPER_SUFFIX = {
  "native": "", "split": " × Split", "aci": " × ACI", "agaci": " × AgACI",
}
_CATEGORY_LABEL = {
  "baseline": "Baselines", "simple_native": "Simple Native",
  "simple_cp": "Simple CP", "complex_cp": "Complex CP",
}
_CATEGORY_ORDER = ["baseline", "simple_native", "simple_cp", "complex_cp"]
_MODEL_MARKER = {
  "fhs": "X", "garch_roll": "P",
  "hist_binning": "s", "tree_based": "^",
  "linear_ar": "o", "lstm": "D", "transformer": "v",
}
_WRAPPER_LS = {
  "native": "-", "split": "--", "aci": ":", "agaci": "-.",
}


def _label(model: str, wrapper: str) -> str:
  return _METHOD_LABEL.get(model, model) + _WRAPPER_SUFFIX.get(wrapper, f" × {wrapper}")


def _save(fig: plt.Figure, stem: Path) -> None:
  stem.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
  fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
  print(f"  saved → {stem.relative_to(ROOT)}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# 1. Load CSVs
# ---------------------------------------------------------------------------

def _load_intervals(results_dir: Path) -> pd.DataFrame:
  files = sorted(f for f in results_dir.glob("*.csv")
                 if f.name != "metrics_long.csv"
                 and not f.name.startswith("garch_hardcoded_")
                 and not f.name.startswith("tcp_paper_"))
  if not files:
    raise FileNotFoundError(f"No interval CSVs in {results_dir}")
  raw = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
  raw["regime"] = raw["regime"].astype(str)
  raw["category"] = raw["model"].map(MODEL_CATEGORIES)
  return raw


# ---------------------------------------------------------------------------
# 2. Build metrics_long.csv  (single source: compute_cell_metrics_full)
# ---------------------------------------------------------------------------

def build_metrics_long(raw: pd.DataFrame, out_path: Path) -> pd.DataFrame:
  rows = []
  for (model, wrapper, ticker, regime), g in raw.groupby(
      ["model", "wrapper", "ticker", "regime"]
  ):
    cfg = CRISIS_WINDOWS[regime]
    widths = g["u"] - g["l"]
    n_inf = int(np.isinf(widths).sum())
    g2 = g[np.isfinite(widths)].copy()
    if len(g2) == 0:
      continue
    m = compute_cell_metrics_full(
      g2, crisis_start=cfg["onset"], crisis_end=cfg["end"], alpha_nominal=ALPHA
    )
    rows.append({
      "model": model, "wrapper": wrapper,
      "category": MODEL_CATEGORIES.get(model, "unknown"),
      "ticker": ticker, "regime": regime,
      "inf_days": n_inf,
      **m,
    })
  long = pd.DataFrame(rows)
  long.to_csv(out_path, index=False)
  print(f"  wrote {len(long)} rows → {out_path.relative_to(ROOT)}")
  return long


# ---------------------------------------------------------------------------
# 3. Figure: Winkler comparison — dot plot sorted by mean Winkler
#
#   Marks:   dots (one per model × wrapper)
#   Channels: x-position = mean Winkler (primary, lower = better)
#             horizontal error bar = ±1 std across cells
#             color = method category
#   Story:   baselines cluster left; CP methods cluster right
# ---------------------------------------------------------------------------

def build_winkler_chart(long: pd.DataFrame, out_dir: Path) -> None:
  agg = (
    long.groupby(["category", "model", "wrapper"])["winkler"]
    .agg(mean="mean", std="std")
    .reset_index()
    .sort_values("mean")
    .reset_index(drop=True)
  )
  agg["label"] = agg.apply(lambda r: _label(r["model"], r["wrapper"]), axis=1)
  agg["color"] = agg["category"].map(CATEGORY_COLORS)

  fig, ax = plt.subplots(figsize=(8, max(5, len(agg) * 0.45)))

  for i, row in agg.iterrows():
    ax.errorbar(
      row["mean"], i, xerr=row["std"],
      fmt="o", color=row["color"], ecolor=row["color"],
      elinewidth=1.5, capsize=3,
      markersize=8, markeredgecolor="black", markeredgewidth=0.5,
      zorder=3,
    )

  best_baseline = agg.loc[agg["category"] == "baseline", "mean"].min()
  ax.axvline(best_baseline, color="#444444", linestyle="--", alpha=0.45, linewidth=1.0,
             label=f"Best baseline  W={best_baseline:.3f}")

  ax.set_yticks(range(len(agg)))
  ax.set_yticklabels(agg["label"], fontsize=9)
  ax.set_xlabel("Mean Winkler score  (lower = better)", fontsize=10)
  ax.set_title(
    "Winkler score by method\n"
    "mean ± 1 std across 9 ticker × regime cells",
    fontsize=10,
  )
  ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  cat_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=CATEGORY_COLORS.get(c, "gray"),
           markeredgecolor="black", markersize=9,
           label=_CATEGORY_LABEL.get(c, c))
    for c in _CATEGORY_ORDER if c in agg["category"].values
  ]
  ax.legend(handles=cat_handles + [
    Line2D([0], [0], color="#444444", linestyle="--", alpha=0.55, label="Best baseline")
  ], loc="lower right", fontsize=9, framealpha=0.9)

  fig.tight_layout()
  _save(fig, out_dir / "winkler_comparison")
  plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Figure: Conditional coverage scatter — cov_low vs cov_high
#
#   Marks:   small dots (individual cells), large dots (per-method mean)
#   Channels: x = coverage on low-vol days
#             y = coverage on high-vol days
#             color = method category
#             shape = model
#   Reference: diagonal y=x (equal coverage across regimes)
#   Story:   CP methods below diagonal; GARCH/FHS near or above
# ---------------------------------------------------------------------------

def build_cov_scatter(long: pd.DataFrame, out_dir: Path) -> None:
  means = (
    long.groupby(["category", "model", "wrapper"])[["cov_low", "cov_high"]]
    .mean().reset_index()
  )

  fig, ax = plt.subplots(figsize=(6.5, 6.5))

  # Individual cells (small, transparent)
  for cat, sub in long.groupby("category"):
    ax.scatter(sub["cov_low"], sub["cov_high"],
               color=CATEGORY_COLORS.get(cat, "gray"),
               alpha=0.18, s=18, zorder=2)

  # Per-(model, wrapper) means (large, opaque)
  for _, row in means.iterrows():
    ax.scatter(
      row["cov_low"], row["cov_high"],
      color=CATEGORY_COLORS.get(row["category"], "gray"),
      marker=_MODEL_MARKER.get(row["model"], "o"),
      s=130, edgecolor="black", linewidth=0.6, zorder=4,
    )

  lims = [0.5, 1.02]
  ax.plot(lims, lims, "k--", alpha=0.25, linewidth=1.0, zorder=1)
  ax.axhline(0.90, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
  ax.axvline(0.90, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
  ax.text(0.515, 0.907, "0.90", fontsize=7, color="gray")
  ax.text(0.907, 0.515, "0.90", fontsize=7, color="gray", rotation=90)

  ax.set_xlim(lims); ax.set_ylim(lims)
  ax.set_aspect("equal")
  ax.set_xlabel("Coverage — low-volatility days", fontsize=11)
  ax.set_ylabel("Coverage — high-volatility days", fontsize=11)
  ax.set_title(
    "Conditional coverage by volatility regime\n"
    "below diagonal = under-covers during high vol",
    fontsize=10,
  )
  ax.grid(True, alpha=0.15, linewidth=0.4)

  cat_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=CATEGORY_COLORS.get(c, "gray"),
           markeredgecolor="black", markersize=9,
           label=_CATEGORY_LABEL.get(c, c))
    for c in _CATEGORY_ORDER if c in means["category"].values
  ]
  model_handles = [
    Line2D([0], [0], marker=m, color="gray", markeredgecolor="black",
           markersize=8, linestyle="", label=_METHOD_LABEL.get(mdl, mdl))
    for mdl, m in _MODEL_MARKER.items()
    if mdl in means["model"].values
  ]
  leg1 = ax.legend(handles=cat_handles, title="Category (color)",
                   loc="upper left", fontsize=8, title_fontsize=8, framealpha=0.9)
  ax.add_artist(leg1)
  ax.legend(handles=model_handles, title="Model (shape)",
            loc="lower right", fontsize=8, title_fontsize=8, framealpha=0.9)

  fig.tight_layout()
  _save(fig, out_dir / "cov_scatter")
  plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Figure: Width dumbbell — non-crisis vs crisis interval width per method
#
#   Marks:   connected dot pairs (one row per method)
#   Channels: x-position = interval width
#             left dot (open) = mean width over full test window
#             right dot (filled) = mean width during crisis only
#             line length = absolute expansion during crisis
#             color = method category (line + both dots)
#   Sorted:  descending expansion ratio (best expanders at top)
#   Story:   GARCH/FHS have long gaps (widen 1.4-1.6x); Split-CP dots
#             overlap perfectly (1.00x, literally frozen)
# ---------------------------------------------------------------------------

def build_width_adaptation(long: pd.DataFrame, out_dir: Path) -> None:
  agg = (
    long.groupby(["category", "model", "wrapper"])[["width", "width_crisis"]]
    .mean().reset_index()
  )
  agg["ratio"] = agg["width_crisis"] / agg["width"]
  agg["label"] = agg.apply(lambda r: _label(r["model"], r["wrapper"]), axis=1)
  # sort ascending so highest ratio ends up at top (highest y index)
  agg = agg.sort_values("ratio", ascending=True).reset_index(drop=True)

  n = len(agg)
  fig, ax = plt.subplots(figsize=(8, max(5, n * 0.52)))

  for i, row in agg.iterrows():
    color = CATEGORY_COLORS.get(row["category"], "gray")

    # connecting line — cap style round so it looks clean at the dots
    ax.plot(
      [row["width"], row["width_crisis"]], [i, i],
      color=color, linewidth=2.4, alpha=0.72,
      solid_capstyle="round", zorder=2,
    )
    # left dot: non-crisis width (open circle)
    ax.scatter(
      row["width"], i,
      facecolors="white", edgecolors=color,
      linewidths=1.6, s=72, zorder=4,
    )
    # right dot: crisis width (filled)
    ax.scatter(
      row["width_crisis"], i,
      color=color, s=72, zorder=4,
    )
    # expansion ratio annotation — right of the filled dot
    ax.text(
      row["width_crisis"] + 0.004, i,
      f"{row['ratio']:.2f}×",
      va="center", ha="left", fontsize=8,
      color=color, alpha=0.85,
    )

  ax.set_yticks(range(n))
  ax.set_yticklabels(agg["label"], fontsize=9)
  ax.set_xlabel("Mean interval width (log-return units)", fontsize=10)
  ax.set_title(
    "Crisis width expansion by method\n"
    "○ non-crisis width   ● crisis-period width   ratio labelled right",
    fontsize=10,
  )
  ax.grid(True, axis="x", alpha=0.18, linewidth=0.5)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  # x-axis room for ratio labels
  x_max = agg["width_crisis"].max() * 1.28
  ax.set_xlim(0, x_max)

  cat_handles = [
    Line2D([0], [0], color=CATEGORY_COLORS.get(c, "gray"),
           linewidth=2.2, label=_CATEGORY_LABEL.get(c, c))
    for c in _CATEGORY_ORDER if c in agg["category"].values
  ]
  ax.legend(handles=cat_handles, loc="lower right", fontsize=9, framealpha=0.9)

  fig.tight_layout()
  _save(fig, out_dir / "width_adaptation")
  plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Figure: Rolling 20-day coverage — 3-panel time series by crisis
#
#   Marks:   lines (one per model × wrapper)
#   Channels: x-position = date
#             y-position = rolling 20-day coverage
#             color = method category
#             line-style = wrapper type (solid/dashed/dotted/dash-dot)
#             line-width = thicker for baselines
#   Reference: 0.90 nominal line, red shaded crisis window
#   Story:   GARCH/FHS stay near 0.90 through crisis; CP dips and lags
# ---------------------------------------------------------------------------

def build_rolling_coverage(raw: pd.DataFrame, out_dir: Path) -> None:
  WINDOW = 20

  fig, axes = plt.subplots(3, 1, figsize=(10, 10))

  for ax, (regime, cfg) in zip(axes, CRISIS_WINDOWS.items()):
    onset, end = cfg["onset"], cfg["end"]
    sub = raw[raw["regime"] == regime].copy()
    sub["date"] = pd.to_datetime(sub["date"])

    for (model, wrapper), grp in sub.groupby(["model", "wrapper"]):
      category = MODEL_CATEGORIES.get(model, "unknown")
      color = CATEGORY_COLORS.get(category, "black")
      lw = 1.8 if category == "baseline" else 0.85
      alpha = 0.95 if category == "baseline" else 0.55
      ls = _WRAPPER_LS.get(wrapper, "-")

      # Rolling coverage averaged across tickers in this regime
      ticker_rolls = []
      for _, tgrp in grp.groupby("ticker"):
        tgrp = tgrp.sort_values("date").set_index("date")
        covered = ((tgrp["r"] >= tgrp["l"]) & (tgrp["r"] <= tgrp["u"])).astype(float)
        ticker_rolls.append(covered.rolling(WINDOW, min_periods=1).mean())
      if not ticker_rolls:
        continue
      avg = pd.concat(ticker_rolls, axis=1).mean(axis=1).sort_index()

      ax.plot(avg.index, avg.values, color=color, linewidth=lw,
              alpha=alpha, linestyle=ls, label=_label(model, wrapper))

    ax.axvspan(onset, end, alpha=0.10, color="red", zorder=0)
    ax.axhline(0.90, color="black", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Rolling 20-day coverage", fontsize=9)
    ax.set_ylim(0.3, 1.05)
    ax.set_title(
      f"{regime} crisis  ({onset.strftime('%b %Y')} – {end.strftime('%b %Y')})",
      fontsize=10,
    )
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

  # Single legend on last panel — wrapper line-style legend + category color legend
  ls_handles = [
    Line2D([0], [0], color="gray", linestyle=ls, linewidth=1.2, label=w.capitalize())
    for w, ls in _WRAPPER_LS.items()
  ]
  cat_handles = [
    Line2D([0], [0], color=CATEGORY_COLORS.get(c, "gray"), linewidth=2.0,
           label=_CATEGORY_LABEL.get(c, c))
    for c in _CATEGORY_ORDER
  ]
  leg1 = axes[-1].legend(handles=cat_handles, title="Color = category",
                         loc="lower left", fontsize=8, title_fontsize=8,
                         framealpha=0.9, ncols=2)
  axes[-1].add_artist(leg1)
  axes[-1].legend(handles=ls_handles, title="Style = wrapper",
                  loc="lower right", fontsize=8, title_fontsize=8,
                  framealpha=0.9)

  fig.tight_layout()
  _save(fig, out_dir / "rolling_coverage")
  plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
  warnings.filterwarnings("ignore")
  FIGURES_DIR.mkdir(parents=True, exist_ok=True)

  print("Loading CSVs ...")
  raw = _load_intervals(RESULTS_DIR)
  print(f"  {len(raw):,} rows across "
        f"{raw.groupby(['model','wrapper','ticker','regime']).ngroups} cells")

  print("\nBuilding metrics_long.csv ...")
  long = build_metrics_long(raw, RESULTS_DIR / "metrics_long.csv")

  print("\nBuilding figures ...")
  build_winkler_chart(long, FIGURES_DIR)
  build_cov_scatter(long, FIGURES_DIR)
  build_width_adaptation(long, FIGURES_DIR)
  build_rolling_coverage(raw, FIGURES_DIR)

  print("\nDone.")


if __name__ == "__main__":
  main()
