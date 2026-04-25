"""Shared plotting helpers used by the per-category notebooks.

The flagship figure is ``plot_interval_timeline`` — one figure per
(model, wrapper, ticker, regime) cell.  Layout:

  * filled band  : prediction interval [l_t, u_t] in the method colour
  * dark line    : predicted point
  * blue scatter : realised log-return at every test day
  * red scatter  : realised return when it falls *outside* the interval
  * shaded vspan : crisis window onset → end
  * dashed verts : crisis onset and end
  * title        : "<model × wrapper>  —  <ticker> (<regime>)   cov=… W=…"

The figure is saved to disk and also returned so the caller can `display()`
it inside a notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Per-category colour palette (band + line colour for each method category)
CATEGORY_COLORS: dict[str, str] = {
  "baseline":      "#7f7f7f",  # gray
  "simple_native": "#9467bd",  # purple
  "simple_cp":     "#2ca02c",  # green
  "complex_cp":    "#ff7f0e",  # orange
}

# Per-method overrides if needed
METHOD_COLORS: dict[str, str] = {
  "fhs":             "#444444",
  "garch_roll":      "#1a1a1a",
  "hist_binning":    "#9467bd",
  "tree_based":      "#7d2eaa",
  "linear_ar":       "#2ca02c",
  "lstm":            "#ff7f0e",
  "transformer":     "#d62728",
}


def _interval_color(model: str, category: str | None = None) -> str:
  return METHOD_COLORS.get(model, CATEGORY_COLORS.get(category or "baseline", "C0"))


def plot_interval_timeline(
  intervals: pd.DataFrame,
  *,
  model: str,
  wrapper: str,
  ticker: str,
  regime: str,
  crisis_start: pd.Timestamp,
  crisis_end: pd.Timestamp,
  category: str | None = None,
  alpha_nominal: float = 0.10,
  save_path: Path | str | None = None,
  figsize: tuple[float, float] = (11, 4),
) -> plt.Figure:
  """Render the per-cell timeline figure and (optionally) save to disk.

  ``intervals`` must contain columns: date (in index or column), point, l, u, r.
  """
  df = intervals.copy()
  if df.index.name != "date" and "date" in df.columns:
    df = df.set_index(pd.to_datetime(df["date"]))
  df = df.sort_index()

  realized = df["r"].astype(float).values
  lo = df["l"].astype(float).values
  up = df["u"].astype(float).values
  pt = df["point"].astype(float).values
  in_iv = (realized >= lo) & (realized <= up)

  cov = float(in_iv.mean())
  width = float(np.mean(up - lo))
  miss = ~in_iv
  # Winkler with nominal alpha
  width_t = up - lo
  pen_lo = (2.0 / alpha_nominal) * np.maximum(lo - realized, 0.0)
  pen_hi = (2.0 / alpha_nominal) * np.maximum(realized - up, 0.0)
  winkler = float(np.mean(width_t + pen_lo + pen_hi))

  color = _interval_color(model, category)

  fig, ax = plt.subplots(figsize=figsize)

  # Crisis shading first so it's underneath
  ax.axvspan(crisis_start, crisis_end, color="red", alpha=0.06, zorder=1)
  ax.axvline(crisis_start, color="black", linestyle="--", linewidth=0.8, alpha=0.55, zorder=2)
  ax.axvline(crisis_end,   color="black", linestyle="--", linewidth=0.8, alpha=0.55, zorder=2)

  # Interval band
  ax.fill_between(df.index, lo, up, color=color, alpha=0.22,
                  label=f"{int((1 - alpha_nominal) * 100)}% interval", zorder=3)

  # Predicted point line
  ax.plot(df.index, pt, color=color, linewidth=1.3, alpha=0.9,
          label="Predicted return", zorder=4)

  # Realised returns: blue dots inside the interval, red dots when out
  ax.scatter(df.index[in_iv], realized[in_iv],
             s=10, color="#1f77b4", alpha=0.55, label="Realised (covered)", zorder=5)
  ax.scatter(df.index[miss], realized[miss],
             s=22, color="#d62728", edgecolor="black", linewidth=0.4,
             label="Realised (missed)", zorder=6)

  ax.set_xlabel("Date")
  ax.set_ylabel("Log return")
  ax.set_title(
    f"{model} × {wrapper}  —  {ticker} ({regime})    "
    f"cov={cov:.3f}   W={winkler:.3f}   width={width:.4f}",
    fontsize=10,
  )
  ax.grid(True, alpha=0.25, linewidth=0.5)
  ax.legend(loc="upper right", fontsize=8, ncols=2, framealpha=0.85)

  fig.tight_layout()
  if save_path is not None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")

  return fig


def compute_cell_metrics(intervals: pd.DataFrame, alpha_nominal: float = 0.10) -> dict:
  """Convenience: per-cell coverage / Winkler / width."""
  realized = intervals["r"].astype(float).values
  lo = intervals["l"].astype(float).values
  up = intervals["u"].astype(float).values
  in_iv = (realized >= lo) & (realized <= up)
  width_t = up - lo
  pen_lo = (2.0 / alpha_nominal) * np.maximum(lo - realized, 0.0)
  pen_hi = (2.0 / alpha_nominal) * np.maximum(realized - up, 0.0)
  return {
    "cov": float(in_iv.mean()),
    "winkler": float(np.mean(width_t + pen_lo + pen_hi)),
    "width": float(np.mean(width_t)),
    "n": int(len(intervals)),
  }


def render_cell_grid(
  raw: pd.DataFrame,
  *,
  model_filter: Iterable[str] | str,
  out_dir: Path | str,
  crisis_windows: dict,
  category_lookup: dict[str, str],
  alpha_nominal: float = 0.10,
) -> pd.DataFrame:
  """Generate one timeline figure per (model, wrapper, ticker, regime) cell
  matching ``model_filter``, save to ``out_dir``, return per-cell metric rows."""
  if isinstance(model_filter, str):
    models = {model_filter}
  else:
    models = set(model_filter)

  out_dir = Path(out_dir)
  rows = []
  for (model, wrapper, ticker, regime), grp in raw[
    raw["model"].isin(models)
  ].groupby(["model", "wrapper", "ticker", "regime"]):
    cfg = crisis_windows[regime]
    grp_idx = grp.set_index(pd.to_datetime(grp["date"])).sort_index()
    fig = plot_interval_timeline(
      grp_idx,
      model=model, wrapper=wrapper, ticker=ticker, regime=regime,
      crisis_start=cfg["onset"], crisis_end=cfg["end"],
      category=category_lookup.get(model),
      alpha_nominal=alpha_nominal,
      save_path=out_dir / f"{model}_{wrapper}_{ticker}_{regime}.png",
    )
    plt.close(fig)
    metrics = compute_cell_metrics(grp_idx, alpha_nominal=alpha_nominal)
    rows.append({"model": model, "wrapper": wrapper, "ticker": ticker, "regime": regime, **metrics})

  return pd.DataFrame(rows)
