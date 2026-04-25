"""Generate the per-category Jupyter notebooks from a single source of truth.

Run from project root:  ``python scripts/build_notebooks.py``

This rebuilds:
    src/analysis/Baselines/baselines.ipynb
    src/analysis/Simple_Native/simple_native.ipynb
    src/analysis/Simple_CP/simple_cp.ipynb
    src/analysis/Complex_CP/complex_cp.ipynb
    src/analysis/TCP_Paper/tcp_paper.ipynb
    src/analysis/results.ipynb

Each per-category notebook does the same three things:
  1. Bootstrap sys.path so ``from src...`` works regardless of CWD.
  2. Run the relevant slice of ``scripts/run_grid.py`` (skipped if CSVs
     already exist), so the notebook is idempotent.
  3. Load the resulting CSVs from ``results/``, render one timeline
     figure per (model, wrapper, ticker, regime) cell into
     ``report/figures/timelines/<Category>/``, and print a per-cell
     metrics table.

The ``results.ipynb`` aggregator does cross-category scatter / rolling
coverage / table 1-2-3 (overall, in-crisis, out-of-crisis).

The ``tcp_paper.ipynb`` is standalone — it reproduces the
``garch_hardcoded`` vs ``garch_roll`` vs ``tcp_paper`` sanity check that
documents *why* TCP was dropped from the main grid.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent


def _md(text: str) -> dict:
  return nbf.v4.new_markdown_cell(text)

def _code(src: str) -> dict:
  return nbf.v4.new_code_cell(src)


# -----------------------------------------------------------------------
# Shared boilerplate cells
# -----------------------------------------------------------------------

BOOTSTRAP = """\
# sys.path bootstrap so 'from src...' works no matter where the kernel was launched from
import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / "requirements.txt").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("project root:", ROOT)
"""

COMMON_IMPORTS = """\
import warnings, glob, subprocess, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

from src.analysis._shared import RESULTS_DIR, CRISIS_WINDOWS, MODEL_CATEGORIES
from src.plotting import plot_interval_timeline, compute_cell_metrics, render_cell_grid
"""


def _run_slice_cell(only_arg: str) -> str:
  return f"""\
# Idempotent: only invokes scripts/run_grid.py if the relevant CSVs are missing.
needed = {{
{_NEEDED[only_arg]}
}}
have = {{p.name for p in RESULTS_DIR.glob("*.csv")}} if RESULTS_DIR.exists() else set()
missing = sorted(n for n in needed if n not in have)
if missing:
    print(f"Running scripts/run_grid.py --only {only_arg}  ({{len(missing)}} cells missing)")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_grid.py"), "--only", "{only_arg}"], check=True)
else:
    print(f"All {only_arg} CSVs present in results/, skipping run.")
"""


def _figure_cell(category_name: str, model_filter: list[str]) -> str:
  return f"""\
# Load every interval CSV produced by this category, render one timeline figure
# per (model, wrapper, ticker, regime) cell, write to disk, return per-cell metrics.
out_dir = ROOT / "report" / "figures" / "timelines" / "{category_name}"
patterns = []
for m in {model_filter!r}:
    patterns.append(str(RESULTS_DIR / f"{{m}}_*.csv"))
files = []
for p in patterns:
    files.extend(glob.glob(p))
files = sorted(files)
if not files:
    raise RuntimeError(f"No CSVs found for category {category_name!r}; did the runner finish?")
raw = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
metrics = render_cell_grid(
    raw, model_filter={model_filter!r}, out_dir=out_dir,
    crisis_windows=CRISIS_WINDOWS, category_lookup=MODEL_CATEGORIES,
)
print(f"wrote {{len(metrics)}} timeline figures → {{out_dir.relative_to(ROOT)}}")
metrics_sorted = metrics.sort_values(["model", "wrapper", "regime", "ticker"]).reset_index(drop=True)
display(metrics_sorted.round(3))
"""


def _summary_cell() -> str:
  return """\
# Aggregate by (model, wrapper) — overall mean across all (ticker, regime) cells
summary = (
    metrics_sorted
    .groupby(["model", "wrapper"])[["cov", "winkler", "width"]]
    .mean()
    .sort_values("winkler")
    .round(3)
)
display(summary)

# Show a couple of representative figures inline
sample_files = sorted((ROOT / "report" / "figures" / "timelines").rglob("*.png"))
if sample_files:
    print(f"\\nShowing first 3 of {len(sample_files)} timeline figures:")
    for p in sample_files[:3]:
        display(Image(filename=str(p)))
"""


_NEEDED: dict[str, str] = {
  "baselines":     '\n'.join(
      f'    f"{m}_native_{{t}}_{{r}}.csv"'
      for m in ("fhs", "garch_roll")
      for r, ts in [
          ("2008", ("AIG","GNW","MGM")),
          ("2020", ("MCD","MGM","AZO")),
          ("2022", ("SNAP","NFLX","META")),
      ]
      for t in ts).replace("{m}", "<>m_BAD"),  # placeholder, fixed below
}
# regenerate _NEEDED cleanly
def _build_needed():
  pairs = [
      ("2008", ("AIG","GNW","MGM")),
      ("2020", ("MCD","MGM","AZO")),
      ("2022", ("SNAP","NFLX","META")),
  ]
  by_slice: dict[str, list[str]] = {}
  by_slice["baselines"] = [
      f'    "fhs_native_{t}_{r}.csv"' for r,ts in pairs for t in ts
  ] + [
      f'    "garch_roll_native_{t}_{r}.csv"' for r,ts in pairs for t in ts
  ]
  by_slice["simple_native"] = [
      f'    "{m}_native_{t}_{r}.csv"'
      for m in ("hist_binning","tree_based") for r,ts in pairs for t in ts
  ]
  by_slice["simple_cp"] = [
      f'    "linear_ar_{w}_{t}_{r}.csv"'
      for w in ("split","aci","agaci") for r,ts in pairs for t in ts
  ]
  by_slice["complex_cp"] = [
      f'    "{m}_{w}_{t}_{r}.csv"'
      for m in ("lstm","transformer")
      for w in ("split","aci","agaci")
      for r,ts in pairs for t in ts
  ]
  return {k: ",\n".join(v) for k, v in by_slice.items()}

_NEEDED = _build_needed()


# -----------------------------------------------------------------------
# Per-category notebook factory
# -----------------------------------------------------------------------

def make_category_notebook(
  out_path: Path,
  title: str,
  blurb: str,
  only_arg: str,
  category_dir_name: str,
  model_filter: list[str],
) -> None:
  nb = nbf.v4.new_notebook()
  nb.cells = [
    _md(f"# {title}\n\n{blurb}\n\n"
        "**Workflow.** This notebook (i) makes sure the relevant interval CSVs in "
        "`results/` exist (running `scripts/run_grid.py --only " + only_arg + "` if "
        "they don't), (ii) loads every CSV and writes a per-cell timeline figure "
        f"under `report/figures/timelines/{category_dir_name}/`, (iii) prints a "
        "per-cell metrics table and a summary aggregated across "
        "(ticker, regime).\n\n"
        "**Re-running.** Delete the CSVs in `results/` to force a re-run."),
    _code(BOOTSTRAP),
    _code(COMMON_IMPORTS),
    _md(f"## 1. Run the experiment slice `--only {only_arg}` (idempotent)"),
    _code(_run_slice_cell(only_arg)),
    _md(f"## 2. Render per-cell timeline figures and per-cell metrics"),
    _code(_figure_cell(category_dir_name, model_filter)),
    _md("## 3. Summary aggregated across (ticker, regime)"),
    _code(_summary_cell()),
  ]
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w") as f:
    nbf.write(nb, f)
  print(f"  wrote {out_path.relative_to(ROOT)}")


# -----------------------------------------------------------------------
# TCP-paper standalone notebook (the sanity-check we ran earlier)
# -----------------------------------------------------------------------

TCP_NOTEBOOK_BODY = """\
# TCP-paper sanity check
#
# Reproduces the garch_hardcoded vs garch_roll vs tcp_paper vs fhs
# comparison on the COVID-2020 crisis.  Documents WHY we dropped TCP
# from the main grid: with a properly fitted rolling GARCH (Student-t,
# parametric quantile) the TCP paper's apparent advantage disappears.
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.analysis._shared import ensure_data, iterate_experiments, walk_forward
from src.conformal.garch import GarchHardcoded, RollingGarch
from src.conformal.fhs import FHS
from src.conformal.tcp_paper import TCPPaperConformal
from src.metrics.metrics import Metrics
from src.metrics.prediction_cols import PredictionCols

CONFIDENCE_LEVEL = 0.90
NUM_LAGS = 5
REFIT_EVERY = 21

data = ensure_data()
TICKERS_2020 = {"MCD", "MGM", "AZO"}
splits_2020 = [s for s in iterate_experiments()
               if s.regime == "2020" and s.ticker.value in TICKERS_2020]

methods = [
    ("garch_hardcoded", lambda lc: GarchHardcoded(label_col_name=lc, confidence_level=CONFIDENCE_LEVEL)),
    ("garch_roll",      lambda lc: RollingGarch(label_col_name=lc, confidence_level=CONFIDENCE_LEVEL,
                                                 garch_window=504, refit_every=REFIT_EVERY)),
    ("fhs",             lambda lc: FHS(label_col_name=lc, confidence_level=CONFIDENCE_LEVEL)),
    ("tcp_paper",       lambda lc: TCPPaperConformal(label_col_name=lc, num_lags=NUM_LAGS,
                                                      confidence_level=CONFIDENCE_LEVEL,
                                                      refit_every=REFIT_EVERY)),
]

rows = []
for s in splits_2020:
    train = s.slice_train(data); calib = s.slice_calib(data); full = s.slice_full(data)
    for name, factory in methods:
        m = factory(s.label_col)
        if name == "fhs":
            m.fit_and_calibrate(pd.concat([train, calib]))
        else:
            m.fit_and_calibrate(train, calib)
        intervals = walk_forward(m, full, s.test_start, s.test_end)
        df = intervals.rename(columns={"r": PredictionCols.LOG_RETURNS,
                                        "l": PredictionCols.LOWER_BOUND,
                                        "u": PredictionCols.UPPER_BOUND})
        widths = df[PredictionCols.UPPER_BOUND] - df[PredictionCols.LOWER_BOUND]
        df = df[~np.isinf(widths)]
        met = Metrics(df, alpha=0.10)
        cov = met.get_coverage_metrics().get_marginal_coverage()
        wink, _, _ = met.get_winkler_metrics().get_mean_scores()
        width, _, _ = met.get_interval_metrics().get_mean_widths()
        rows.append({"method": name, "ticker": s.ticker.value,
                     "cov": cov, "winkler": wink, "width": width})

results = pd.DataFrame(rows)
display(results.round(3))

print("\\nAverage across MCD, MGM, AZO  (lower Winkler = better)")
summary = (results.groupby("method")[["cov", "winkler", "width"]]
           .mean().sort_values("winkler").round(3))
display(summary)
"""


def make_tcp_notebook(out_path: Path) -> None:
  nb = nbf.v4.new_notebook()
  nb.cells = [
    _md("# TCP-paper sanity check (standalone)\n\n"
        "Reproduces the comparison that justifies dropping TCP from the main grid:\n\n"
        "* `garch_hardcoded` — TCP paper's exact GARCH (ω=1e-6, α=0.05, β=0.9, normal, no MLE).\n"
        "* `garch_roll`     — properly fitted rolling GARCH(1,1) Student-t (parametric quantile).\n"
        "* `fhs`            — Filtered Historical Simulation.\n"
        "* `tcp_paper`      — paper-faithful TCP (gradient-boosted quantile + rolling split-conformal).\n\n"
        "Headline result: on COVID 2020 individual equities, `garch_roll` and `fhs` both beat "
        "`tcp_paper` on Winkler score; `garch_hardcoded` undercoverere as expected."),
    _code(BOOTSTRAP),
    _code(TCP_NOTEBOOK_BODY),
  ]
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w") as f:
    nbf.write(nb, f)
  print(f"  wrote {out_path.relative_to(ROOT)}")


# -----------------------------------------------------------------------
# Aggregate results notebook
# -----------------------------------------------------------------------

AGG_BODY = """\
# Aggregate across every CSV in results/
from src.analysis._shared import iterate_experiments

all_files = sorted(RESULTS_DIR.glob("*.csv"))
all_files = [f for f in all_files if f.name != "metrics_long.csv"]
print(f"Loaded {len(all_files)} interval CSVs")
raw = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in all_files], ignore_index=True)
raw["category"] = raw["model"].map(MODEL_CATEGORIES)
print(raw.groupby(["category", "model", "wrapper"]).size().to_string())
"""

AGG_METRICS = """\
# Per-cell metrics: overall, in-crisis, out-of-crisis
def metrics_for_subset(df, alpha=0.10):
    if len(df) == 0: return None
    realized = df['r'].astype(float).values
    lo = df['l'].astype(float).values
    up = df['u'].astype(float).values
    width_t = up - lo
    pen_lo = (2.0/alpha) * np.maximum(lo - realized, 0.0)
    pen_hi = (2.0/alpha) * np.maximum(realized - up, 0.0)
    in_iv = (realized >= lo) & (realized <= up)
    return {"cov": float(in_iv.mean()), "winkler": float(np.mean(width_t + pen_lo + pen_hi)),
            "width": float(np.mean(width_t)), "n": int(len(df))}

rows = []
for (model, wrapper, ticker, regime), grp in raw.groupby(["model", "wrapper", "ticker", "regime"]):
    cfg = CRISIS_WINDOWS[str(regime)]
    in_crisis  = grp[(grp["date"] >= cfg["onset"]) & (grp["date"] <= cfg["end"])]
    out_crisis = grp[(grp["date"] < cfg["onset"]) | (grp["date"] > cfg["end"])]
    overall = metrics_for_subset(grp)
    incr    = metrics_for_subset(in_crisis)
    outcr   = metrics_for_subset(out_crisis)
    if overall is None: continue
    rows.append({
        "category": MODEL_CATEGORIES.get(model, "unknown"),
        "model": model, "wrapper": wrapper, "ticker": ticker, "regime": regime,
        "cov_all": overall["cov"], "winkler_all": overall["winkler"], "width_all": overall["width"],
        "cov_in":  incr["cov"]    if incr  else np.nan, "winkler_in":  incr["winkler"]  if incr  else np.nan, "width_in":  incr["width"]  if incr  else np.nan,
        "cov_out": outcr["cov"]   if outcr else np.nan, "winkler_out": outcr["winkler"] if outcr else np.nan, "width_out": outcr["width"] if outcr else np.nan,
    })
metrics_long = pd.DataFrame(rows)
metrics_long.to_csv(RESULTS_DIR / "metrics_long.csv", index=False)
print(f"wrote {RESULTS_DIR/'metrics_long.csv'} ({len(metrics_long)} rows)")
"""

AGG_TABLE = """\
# Three summary tables: overall, in-crisis, out-of-crisis (sorted by Winkler)
def make_table(df, suffix):
    cols = [f"cov_{suffix}", f"winkler_{suffix}", f"width_{suffix}"]
    out = (df.groupby(["category", "model", "wrapper"])[cols]
             .mean()
             .sort_values(f"winkler_{suffix}")
             .round(3))
    return out

print("=" * 70)
print("Table 1 — Overall (all test-window days)")
print("=" * 70)
display(make_table(metrics_long, "all"))

print("\\n" + "=" * 70)
print("Table 2 — In-crisis only")
print("=" * 70)
display(make_table(metrics_long, "in"))

print("\\n" + "=" * 70)
print("Table 3 — Out-of-crisis (pre / post-crisis padding)")
print("=" * 70)
display(make_table(metrics_long, "out"))
"""

AGG_SCATTER = """\
# Cross-method scatter: |coverage gap| vs Winkler, coloured by category
fig, ax = plt.subplots(figsize=(9, 6))
agg = (metrics_long.groupby(["category", "model", "wrapper"])
       .agg(cov_all=("cov_all", "mean"), winkler=("winkler_all", "mean"))
       .reset_index())
agg["cov_gap"] = agg["cov_all"] - 0.90

CAT_COLOR = {"baseline":"#444444", "simple_native":"#9467bd", "simple_cp":"#2ca02c", "complex_cp":"#ff7f0e"}
for cat, sub in agg.groupby("category"):
    ax.scatter(sub["cov_gap"], sub["winkler"], s=110, alpha=0.85,
               edgecolor="black", linewidth=0.5,
               color=CAT_COLOR.get(cat, "C0"), label=cat)
    for _, row in sub.iterrows():
        ax.annotate(f"{row['model']}×{row['wrapper']}", (row["cov_gap"], row["winkler"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7, alpha=0.75)
ax.axvline(0, color="black", linestyle=":", alpha=0.5)
ax.set_xlabel("Coverage gap from nominal 0.90  (negative = undercoverage)")
ax.set_ylabel("Winkler score (lower = better)")
ax.set_title("Method comparison — coverage calibration vs interval quality")
ax.grid(True, alpha=0.25)
ax.legend(title="Category", loc="upper right")
out = ROOT / "report" / "figures" / "aggregate" / "scatter_cov_winkler.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"saved → {out.relative_to(ROOT)}")
plt.show()
"""


def make_results_notebook(out_path: Path) -> None:
  nb = nbf.v4.new_notebook()
  nb.cells = [
    _md("# Aggregate results — across-category comparison\n\n"
        "Reads every `results/<model>_<wrapper>_<ticker>_<regime>.csv`, computes\n"
        "(i) overall, (ii) in-crisis, (iii) out-of-crisis metrics, writes\n"
        "`results/metrics_long.csv`, and renders the cross-method scatter into\n"
        "`report/figures/aggregate/`."),
    _code(BOOTSTRAP),
    _code(COMMON_IMPORTS),
    _md("## 1. Load every interval CSV"),
    _code(AGG_BODY),
    _md("## 2. Compute per-cell metrics for overall / in-crisis / out-of-crisis"),
    _code(AGG_METRICS),
    _md("## 3. Summary tables"),
    _code(AGG_TABLE),
    _md("## 4. Cross-method scatter — coverage gap vs Winkler"),
    _code(AGG_SCATTER),
  ]
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w") as f:
    nbf.write(nb, f)
  print(f"  wrote {out_path.relative_to(ROOT)}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
  print(f"Building notebooks under {ROOT}/src/analysis/ ...")
  ana = ROOT / "src" / "analysis"

  make_category_notebook(
    ana / "Baselines" / "baselines.ipynb",
    "Baselines — FHS and rolling GARCH",
    "Industry-standard volatility-scaled prediction-interval methods, used as "
    "the comparison target for every other category in this paper.",
    only_arg="baselines",
    category_dir_name="Baselines",
    model_filter=["fhs", "garch_roll"],
  )
  make_category_notebook(
    ana / "Simple_Native" / "simple_native.ipynb",
    "Simple + Native intervals",
    "Methods that produce prediction intervals natively from data structure, "
    "without a conformal wrapper: histogram-binning quantiles and Quantile "
    "Regression Forests (Meinshausen 2006).",
    only_arg="simple_native",
    category_dir_name="Simple_Native",
    model_filter=["hist_binning", "tree_based"],
  )
  make_category_notebook(
    ana / "Simple_CP" / "simple_cp.ipynb",
    "Simple + CP",
    "Linear AR point predictor wrapped with split-conformal, ACI, and AgACI.",
    only_arg="simple_cp",
    category_dir_name="Simple_CP",
    model_filter=["linear_ar"],
  )
  make_category_notebook(
    ana / "Complex_CP" / "complex_cp.ipynb",
    "Complex + CP",
    "LSTM and Transformer point predictors wrapped with split-conformal, "
    "ACI, and AgACI.  Tests whether base-model complexity helps interval "
    "quality once the uncertainty layer is fixed.",
    only_arg="complex_cp",
    category_dir_name="Complex_CP",
    model_filter=["lstm", "transformer"],
  )
  make_tcp_notebook(ana / "TCP_Paper" / "tcp_paper.ipynb")
  make_results_notebook(ana / "results.ipynb")
  print("\nDone.")


if __name__ == "__main__":
  main()
