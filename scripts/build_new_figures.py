"""Two extra timeline figures for the report.

Both are 1×2 (left = method A, right = method B) timelines on MCD during the
2020 COVID crisis, mirroring the style of the existing
``report/figures/timeline_comparison.pdf``:

  * ``timeline_qrf_online``  — QRF-Online vs Split-CP (LSTM × Split):
    a native online tree method widens during the crisis without any
    conformal layer, while Split-CP's intervals are frozen.
  * ``timeline_tcp_vs_garch`` — GARCH-Roll vs TCP (paper-faithful):
    rolling GARCH widens instantly on crisis day 1; TCP widens gradually
    as its 60-day calibration window fills with crisis residuals.

Both figures share a y-range across panels so the visual contrast is in
*width and timing*, not in axis scaling.

Run from the project root:
    python scripts/build_new_figures.py

Saves to ``report/figures/<stem>.{png,pdf}``.  Does not touch
``build_artifacts.py``.
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
import numpy as np
import pandas as pd

from src.analysis._shared import CRISIS_WINDOWS, MODEL_CATEGORIES, RESULTS_DIR
from src.plotting import plot_interval_timeline

ALPHA = 0.10
FIGURES_DIR = ROOT / "report" / "figures"
PANEL_FIGSIZE = (11, 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cell_csv(path: Path) -> pd.DataFrame:
  """Load one interval CSV and finite-filter inf widths (matches build_artifacts)."""
  if not path.exists():
    raise FileNotFoundError(f"Missing interval CSV: {path}")
  df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
  widths = df["u"] - df["l"]
  return df[np.isfinite(widths)]


def _shared_ylim(*frames: pd.DataFrame, pad_frac: float = 0.05) -> tuple[float, float]:
  """Compute a common y-range covering every interval band and realised return."""
  lo = min(min(d["l"].min(), d["r"].min()) for d in frames)
  hi = max(max(d["u"].max(), d["r"].max()) for d in frames)
  pad = pad_frac * (hi - lo)
  return lo - pad, hi + pad


def _save(fig: plt.Figure, stem: Path) -> None:
  stem.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
  fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
  print(f"  saved → {stem.relative_to(ROOT)}.{{png,pdf}}")


def _build_two_panel(
  *,
  left_path: Path,
  right_path: Path,
  out_stem: Path,
  suptitle: str,
) -> None:
  """Render a 1x2 figure: left_path on left, right_path on right, shared y-axis."""
  left = _load_cell_csv(left_path)
  right = _load_cell_csv(right_path)

  # Each interval CSV has the metadata columns repeated on every row.
  def _meta(df: pd.DataFrame) -> dict:
    return {col: df[col].iloc[0] for col in ("model", "wrapper", "ticker", "regime")}

  L = _meta(left); R = _meta(right)
  if L["regime"] != R["regime"] or L["ticker"] != R["ticker"]:
    raise ValueError(
      f"Cells must share ticker+regime; got "
      f"{L['ticker']}/{L['regime']} vs {R['ticker']}/{R['regime']}"
    )

  cfg = CRISIS_WINDOWS[str(L["regime"])]
  crisis_start, crisis_end = cfg["onset"], cfg["end"]

  fig, axes = plt.subplots(
    1, 2,
    figsize=(PANEL_FIGSIZE[0] * 2, PANEL_FIGSIZE[1]),
    sharey=True,
  )
  ylim = _shared_ylim(left, right)

  for ax, df, meta in ((axes[0], left, L), (axes[1], right, R)):
    plot_interval_timeline(
      df,
      ax=ax,
      model=str(meta["model"]),
      wrapper=str(meta["wrapper"]),
      ticker=str(meta["ticker"]),
      regime=str(meta["regime"]),
      crisis_start=crisis_start,
      crisis_end=crisis_end,
      category=MODEL_CATEGORIES.get(str(meta["model"])),
      alpha_nominal=ALPHA,
    )
    ax.set_ylim(ylim)

  # The right panel inherits the y-tick labels from the left via sharey,
  # but the y-axis label was set on both — clear the duplicate on the right.
  axes[1].set_ylabel("")

  fig.suptitle(suptitle, fontsize=12, y=1.02)
  fig.tight_layout()
  _save(fig, out_stem)
  plt.close(fig)


# ---------------------------------------------------------------------------
# Figure recipes
# ---------------------------------------------------------------------------

def build_qrf_online_figure(out_dir: Path) -> None:
  """QRF-Online (tree_based × native) vs Split-CP (LSTM × split) — MCD COVID."""
  _build_two_panel(
    left_path=RESULTS_DIR / "tree_based_native_MCD_2020.csv",
    right_path=RESULTS_DIR / "lstm_split_MCD_2020.csv",
    out_stem=out_dir / "timeline_qrf_online",
    suptitle=(
      "QRF-Online (native) vs Split-CP — MCD, 2020 COVID crisis\n"
      "left: native tree intervals widen during crisis; right: Split-CP intervals are frozen"
    ),
  )


def build_tcp_vs_garch_figure(out_dir: Path) -> None:
  """GARCH-Roll vs TCP (paper-faithful) — MCD COVID.

  TCP CSVs are excluded from build_artifacts.py's main load
  (build_artifacts.py:85), so we load them by exact filename here.
  """
  _build_two_panel(
    left_path=RESULTS_DIR / "garch_roll_native_MCD_2020.csv",
    right_path=RESULTS_DIR / "tcp_paper_native_MCD_2020.csv",
    out_stem=out_dir / "timeline_tcp_vs_garch",
    suptitle=(
      "GARCH-Roll vs TCP (paper-faithful) — MCD, 2020 COVID crisis\n"
      "left: rolling GARCH widens instantly; right: TCP widens gradually as its calibration window absorbs crisis days"
    ),
  )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
  warnings.filterwarnings("ignore")
  FIGURES_DIR.mkdir(parents=True, exist_ok=True)

  print("Building report figures ...")
  build_qrf_online_figure(FIGURES_DIR)
  build_tcp_vs_garch_figure(FIGURES_DIR)
  print("Done.")


if __name__ == "__main__":
  main()
