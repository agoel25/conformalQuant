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

All metric computations delegate to ``src.metrics.Metrics`` (the spec in
``src/metrics/Metrics.md``), so plotting numbers and table numbers are
guaranteed to match.

The figure is saved to disk and also returned so the caller can ``display()``
it inside a notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics.metrics import Metrics
from src.metrics.prediction_cols import PredictionCols


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


def _to_metrics_frame(intervals: pd.DataFrame) -> pd.DataFrame:
  """Internal helper: ensure the DataFrame has a DatetimeIndex and the column
  names that ``src.metrics.Metrics`` expects (``log_returns``, ``lower_bound``,
  ``upper_bound``)."""
  df = intervals.copy()
  if df.index.name != "date" and "date" in df.columns:
    df = df.set_index(pd.to_datetime(df["date"]))
  df = df.sort_index()
  return df.rename(columns={
    "r": str(PredictionCols.LOG_RETURNS),
    "l": str(PredictionCols.LOWER_BOUND),
    "u": str(PredictionCols.UPPER_BOUND),
  })


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
  ax: plt.Axes | None = None,
) -> plt.Figure:
  """Render the per-cell timeline figure and (optionally) save to disk.

  ``intervals`` must contain columns: date (in index or column), point, l, u, r.
  Coverage / Winkler / width in the title come from ``src.metrics.Metrics``.

  When ``ax`` is given, draws onto it (caller owns the figure: ``figsize``,
  ``save_path``, and ``tight_layout`` are skipped). Returns ``ax.figure``.
  """
  df = _to_metrics_frame(intervals)
  realized = df[str(PredictionCols.LOG_RETURNS)].astype(float).values
  lo = df[str(PredictionCols.LOWER_BOUND)].astype(float).values
  up = df[str(PredictionCols.UPPER_BOUND)].astype(float).values
  pt = df["point"].astype(float).values
  in_iv = (realized >= lo) & (realized <= up)
  miss = ~in_iv

  m = Metrics(df, alpha=alpha_nominal)
  cov = float(m.get_coverage_metrics().get_marginal_coverage())
  winkler = float(m.get_winkler_metrics().get_mean_scores()[0])
  width = float(m.get_interval_metrics().get_mean_widths()[0])

  color = _interval_color(model, category)

  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
    own_fig = True
  else:
    fig = ax.figure
    own_fig = False

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

  if own_fig:
    fig.tight_layout()
    if save_path is not None:
      save_path = Path(save_path)
      save_path.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(save_path, dpi=140, bbox_inches="tight")
      fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")

  return fig


def compute_cell_metrics(intervals: pd.DataFrame, alpha_nominal: float = 0.10) -> dict:
  """Convenience: per-cell coverage / Winkler / width — delegates to
  ``src.metrics.Metrics`` so plotting and analysis tables share one source.
  """
  df = _to_metrics_frame(intervals)
  m = Metrics(df, alpha=alpha_nominal)
  return {
    "cov": float(m.get_coverage_metrics().get_marginal_coverage()),
    "winkler": float(m.get_winkler_metrics().get_mean_scores()[0]),
    "width": float(m.get_interval_metrics().get_mean_widths()[0]),
    "n": int(len(df)),
  }


def compute_cell_metrics_full(
  intervals: pd.DataFrame,
  *,
  crisis_start: pd.Timestamp,
  crisis_end: pd.Timestamp,
  alpha_nominal: float = 0.10,
) -> dict:
  """Per-cell metrics matching ``src/metrics/Metrics.md`` spec:
  marginal + high/low-vol-stratified coverage, Winkler and width
  (overall / high-vol / low-vol / in-crisis), plus adaptation speed.
  """
  df = _to_metrics_frame(intervals)
  m = Metrics(df, alpha=alpha_nominal)

  cov_high, cov_low, cov_gap = m.get_coverage_metrics().get_high_low_vol_coverage()
  w_overall, w_high, w_low = m.get_winkler_metrics().get_mean_scores()
  width_overall, width_high, width_low = m.get_interval_metrics().get_mean_widths()
  cov_crisis = float(m.get_coverage_metrics().get_period_coverage(crisis_start, crisis_end))
  win_crisis = float(m.get_winkler_metrics().get_period_score(crisis_start, crisis_end))
  width_crisis = float(m.get_interval_metrics().get_period_width(crisis_start, crisis_end))
  t_rec, max_dd, recovered = m.get_adaptation_metrics().compute_adaptation_speed(
    crisis_start, crisis_end, threshold=0.85, persistence=5, window=20,
  )

  return {
    "n": int(len(df)),
    "cov": float(m.get_coverage_metrics().get_marginal_coverage()),
    "cov_high": float(cov_high), "cov_low": float(cov_low), "cov_gap": float(cov_gap),
    "winkler": float(w_overall), "w_high": float(w_high), "w_low": float(w_low),
    "width": float(width_overall), "width_high": float(width_high), "width_low": float(width_low),
    "cov_crisis": cov_crisis, "winkler_crisis": win_crisis, "width_crisis": width_crisis,
    "t_recovery": int(t_rec),
    "max_drawdown_cov": float(max_dd) if max_dd is not None and not np.isnan(max_dd) else float("nan"),
    "recovered": bool(recovered),
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
  raw = raw.copy()
  raw["regime"] = raw["regime"].astype(str)  # CSV often parses regime as int
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
