# Metrics Implementation Guide

## Overview

We evaluate all models on five metrics. The goal is to assess whether online conformal
prediction methods produce practically useful prediction intervals for US equity markets,
and whether performance degrades systematically during high-volatility regimes. Each
metric below includes its motivation, precise formula, implementation instructions, and
what to report.

All metrics are computed on the out-of-sample period only. Never use in-sample data
for any metric computation. Store daily interval bounds [l_t, u_t] and realized returns
r_t as arrays before computing any metric — all five metrics derive from these same arrays
plus the volatility day mask described in Metric 3.

Nominal coverage level: **1 - α = 0.90**, so **α = 0.10**.

---

## Shared Setup: Daily Coverage Indicator

Before computing any metric, compute the daily coverage indicator for each model and asset:

```python
covered_t = (r_t >= l_t) & (r_t <= u_t)  # Boolean array, shape (T,)
```

This array is reused by Metrics 2, 3, 4, and 5. Compute it once and store it.

---

## Shared Setup: Volatility Day Mask

Used by Metrics 3 and 4. Compute once per asset, shared across all models.

```python
import numpy as np
import pandas as pd

# Compute 20-day rolling realized variance
rolling_var = returns.rolling(20).var()

# Compute 75th percentile threshold over FULL sample
# This must be computed once and held fixed - do not recompute per window or per model
threshold = rolling_var.quantile(0.75)

# Boolean mask: True = high volatility day
high_vol_mask = rolling_var > threshold
low_vol_mask = ~high_vol_mask
```

**Critical:** The threshold is computed once over the full sample period for each asset
and is identical across all models. If you recompute it per model or per window the
cross-model comparison is invalid.

---

## Metric 1: Winkler Score (Primary Metric)

### Motivation

A model can achieve good marginal coverage simply by producing very wide intervals.
Reporting coverage and width separately allows this to go undetected — a model with
91% coverage and excessively wide intervals looks acceptable on both dimensions
individually. The Winkler score prevents this by jointly penalizing both miscoverage
and excessive width in a single number. It is the standard unified metric in
probabilistic forecasting and directly operationalizes whether intervals are practically
useful rather than just nominally calibrated. It is impossible to achieve a good Winkler
score through systematic over-widening because the width term penalizes that directly.

### Formula

For each time step t with prediction interval [l_t, u_t], realized return r_t, and
nominal miscoverage rate α = 0.10:

```
W_t = (u_t - l_t)
    + (2/α) * (l_t - r_t) * 1(r_t < l_t)
    + (2/α) * (r_t - u_t) * 1(r_t > u_t)
```

The first term penalizes interval width. The second term penalizes lower violations
weighted by miss severity. The third term penalizes upper violations weighted by miss
severity. At α = 0.10 the penalty multiplier is 20, meaning each unit of violation
incurs 20x the penalty of one unit of excess width. **Lower is better.**

### Implementation

```python
alpha = 0.10

width = u - l
lower_violation = np.maximum(l - r, 0)  # Positive when r_t < l_t
upper_violation = np.maximum(r - u, 0)  # Positive when r_t > u_t

winkler = width + (2 / alpha) * lower_violation + (2 / alpha) * upper_violation
# winkler is shape (T,) - daily scores
```

### What To Report

Report the following for each model and each asset:

| Quantity | Description |
|---|---|
| W_overall | Mean Winkler over full out-of-sample period |
| W_high | Mean Winkler on high volatility days (using volatility mask) |
| W_low | Mean Winkler on low volatility days |
| W_2008 | Mean Winkler within 2008 crisis window |
| W_2020 | Mean Winkler within 2020 crisis window |
| W_2022 | Mean Winkler within 2022 crisis window |

Report all six quantities in a single table per asset. Average across assets for the
summary table.

**Note on interpretation:** A single severe violation during a crisis will substantially
move the mean Winkler score because of the 20x multiplier. This is intentional — large
misses during crises are disproportionately costly in practice and the metric should
reflect that.

---

## Metric 2: Marginal Coverage

### Motivation

Table stakes for any prediction interval paper. Required for comparability with all
prior work in this area. Also serves as a sanity check — a model should achieve
approximately nominal coverage (0.90) over the full sample, and you should verify this
before interpreting the more nuanced metrics. Included to demonstrate that good Winkler
scores are not being achieved by systematic over-coverage.

### Formula

```
Coverage = (1/T) * Σ_t 1(r_t ∈ [l_t, u_t])
```

### Implementation

```python
marginal_coverage = covered_t.mean()  # Scalar per model per asset
```

### What To Report

Single number per model per asset. Include as the first column in your main results
table so readers can immediately verify nominal coverage before examining other metrics.
Flag any model achieving coverage below 0.85 or above 0.95 as noteworthy.

---

## Metric 3: Conditional Coverage Stratified by Volatility Regime

### Motivation

Marginal coverage allows good calm-period performance to mask crisis failures through
averaging. A model achieving 90% overall coverage but only 75% coverage during
high-volatility periods is failing precisely when a portfolio manager needs reliable
intervals most. This is your primary novel diagnostic contribution. It directly tests
whether CP methods are useful when it matters most rather than on average.

### Formula

```
Cov_high = mean(covered_t[high_vol_mask])
Cov_low  = mean(covered_t[low_vol_mask])
Gap      = Cov_low - Cov_high
```

A large positive Gap means the model masks crisis failures behind calm-period success.
A Gap near zero means coverage is consistent across regimes.

### Implementation

```python
cov_high = covered_t[high_vol_mask].mean()
cov_low  = covered_t[low_vol_mask].mean()
gap      = cov_low - cov_high

# Also compute within each crisis window using exogenous dates
# Define crisis masks using pandas DatetimeIndex
crisis_2008 = (dates >= '2008-09-15') & (dates <= '2009-03-31')
crisis_2020 = (dates >= '2020-02-20') & (dates <= '2020-04-30')
crisis_2022 = (dates >= '2022-03-16') & (dates <= '2022-12-31')

cov_crisis_2008 = covered_t[crisis_2008].mean()
cov_crisis_2020 = covered_t[crisis_2020].mean()
cov_crisis_2022 = covered_t[crisis_2022].mean()
```

### What To Report

| Quantity | Description |
|---|---|
| Cov_high | Coverage on high volatility days |
| Cov_low | Coverage on low volatility days |
| Gap | Cov_low - Cov_high |
| Cov_2008 | Coverage within 2008 crisis window |
| Cov_2020 | Coverage within 2020 crisis window |
| Cov_2022 | Coverage within 2022 crisis window |

**Required visualization:** Scatter plot with Cov_low on the x-axis and Cov_high on
the y-axis, one point per model, with the diagonal y = x as a reference line. Models
on the diagonal have consistent coverage across regimes. Models below the diagonal
under-cover during high volatility relative to calm periods. This is your main results
figure — produce it for each asset and an averaged version across all assets.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for model_name, (cov_l, cov_h) in results.items():
    ax.scatter(cov_l, cov_h, label=model_name)
# Reference diagonal
lims = [0.6, 1.0]
ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal coverage')
ax.set_xlabel('Coverage: Low Volatility Days')
ax.set_ylabel('Coverage: High Volatility Days')
ax.set_title('Conditional Coverage by Volatility Regime')
ax.legend()
```

---

## Metric 4: Interval Width Stratified by Volatility Regime

### Motivation

Necessary complement to conditional coverage. A model might achieve high conditional
coverage during volatile periods simply by producing very wide intervals always. The
Winkler score penalizes this globally but explicit width stratification makes the
mechanism transparent. It lets you distinguish whether a method achieves high-volatility
coverage through genuine calibration or through systematic over-widening, and allows
direct characterization of the sharpness-calibration tradeoff across regimes.

### Formula

```
Width_high = mean(u_t - l_t) for t in high_vol_mask
Width_low  = mean(u_t - l_t) for t in low_vol_mask
```

### Implementation

```python
width = u - l  # Shape (T,)

width_high = width[high_vol_mask].mean()
width_low  = width[low_vol_mask].mean()

# Also within crisis windows
width_2008 = width[crisis_2008].mean()
width_2020 = width[crisis_2020].mean()
width_2022 = width[crisis_2022].mean()
```

### What To Report

Report Width_high, Width_low, and width within each crisis window in the same table as
conditional coverage so the reader can immediately see the coverage-width relationship
side by side. If a model achieves high Cov_high but also has very high Width_high,
flag this explicitly in your discussion — that model is covering through over-widening
not calibration.

---

## Metric 5: Adaptation Speed

### Motivation

Practitioners managing risk through a regime transition need to know not just whether
a model eventually recovers valid coverage but how quickly it does so and how badly it
fails in the meantime. Conditional coverage tells you aggregate performance during a
crisis window but not the trajectory within it. Adaptation speed formalizes the recovery
trajectory into two interpretable numbers per crisis per model: how long recovery takes,
and how severe the worst failure is before recovery occurs.

### Exogenous Onset Dates

The following onset dates are fixed before any analysis. They are derived from
well-documented external events, not from your data. **Do not change these dates based
on your results.**

| Crisis | Onset Date | End Date | Event |
|---|---|---|---|
| 2008 | 2008-09-15 | 2009-03-31 | Lehman Brothers bankruptcy |
| 2020 | 2020-02-20 | 2020-04-30 | S&P 500 pre-crash peak |
| 2022 | 2022-03-16 | 2022-12-31 | First Fed rate hike |

Using exogenous dates is essential. If you derived onset dates from your return data
you would introduce circularity — the dates would reflect model-specific behavior and
comparisons across models would be confounded.

### Recovery Criterion

Rolling 20-day empirical coverage returns above **0.85** and remains above 0.85 for
at least **5 consecutive trading days** after the onset date.

Use 0.85 rather than nominal 0.90 because 20-day rolling coverage is noisy enough that
requiring exactly 0.90 would make recovery appear artificially delayed due to single-day
fluctuations. The 5-day persistence requirement ensures you are detecting genuine
recovery rather than a transient uptick.

### Quantities To Compute

- **T_recovery:** Number of trading days from onset until recovery criterion is met
- **Max_drawdown_cov:** Minimum rolling 20-day coverage observed between onset and
  T_recovery. Captures severity of worst failure independently of recovery speed.
- **Flag:** If recovery never occurs within the crisis window, record T_recovery as the
  full window length and mark the row with an explicit flag in your table.

### Implementation

```python
def compute_adaptation_speed(covered_t, dates, onset_date, end_date,
                              threshold=0.85, persistence=5, window=20):
    """
    covered_t: boolean array of daily coverage indicators
    dates: DatetimeIndex aligned with covered_t
    onset_date, end_date: strings 'YYYY-MM-DD'
    Returns T_recovery (int), max_drawdown_cov (float), recovered (bool)
    """
    # Compute rolling 20-day coverage over full series
    covered_series = pd.Series(covered_t.astype(float), index=dates)
    rolling_cov = covered_series.rolling(window).mean()

    # Extract crisis window starting from onset
    crisis_mask = (dates >= onset_date) & (dates <= end_date)
    crisis_rolling = rolling_cov[crisis_mask]
    crisis_dates = dates[crisis_mask]

    # Find first index where coverage exceeds threshold
    # for at least `persistence` consecutive days
    above = (crisis_rolling >= threshold).values
    recovered = False
    T_recovery = len(above)  # Default: full window if never recovers

    for i in range(len(above) - persistence + 1):
        if above[i:i + persistence].all():
            T_recovery = i
            recovered = True
            break

    # Max drawdown: minimum rolling coverage between onset and recovery
    max_drawdown_cov = crisis_rolling.iloc[:T_recovery].min()

    return T_recovery, max_drawdown_cov, recovered


# Example usage for one model and one crisis
T_rec, drawdown, rec_flag = compute_adaptation_speed(
    covered_t=covered_t,
    dates=dates,
    onset_date='2020-02-20',
    end_date='2020-04-30'
)
```

### What To Report

One table per crisis with a row per model:

| Model | T_recovery (days) | Max_drawdown_cov | Recovered? |
|---|---|---|---|
| FHS | ... | ... | Yes |
| GARCH | ... | ... | Yes |
| LSTM + Split CP | ... | ... | No* |
| ... | ... | ... | ... |

*Flag non-recovery explicitly.

Produce this table for each of the three crises separately. Do not average across
crises — the three crises are structurally different and the results may diverge.

### Robustness Check (Required)

In a short paragraph in your paper, also report T_recovery using:
- Threshold = 0.80 instead of 0.85
- No persistence requirement (just first crossing of threshold)

If your qualitative conclusions about which methods recover faster are unchanged across
these variations your results are robust to the threshold choice. Report this explicitly.

```python
# Robustness version
T_rec_robust, _, _ = compute_adaptation_speed(
    covered_t=covered_t,
    dates=dates,
    onset_date='2020-02-20',
    end_date='2020-04-30',
    threshold=0.80,
    persistence=1  # No persistence requirement
)
```

---

## Summary: What To Include In Each Table

### Main Results Table (one per asset, plus average across assets)

| Model | Marginal Cov | Cov_high | Cov_low | Gap | W_overall | W_high | W_low | Width_high | Width_low |
|---|---|---|---|---|---|---|---|---|---|

### Crisis-Specific Table (one per crisis)

| Model | Coverage | Winkler | Width | T_recovery | Max_drawdown_cov | Recovered |
|---|---|---|---|---|---|---|

### Key Figure

Scatter plot of Cov_low vs Cov_high for all models with diagonal reference line.
One plot per asset plus one averaged across all assets.

---

## Implementation Checklist

- [ ] Compute and store daily [l_t, u_t] and r_t for all models before any metric
- [ ] Compute covered_t boolean array from these
- [ ] Compute volatility threshold once per asset over full sample, store and reuse
- [ ] Compute high_vol_mask and low_vol_mask using fixed threshold
- [ ] Define crisis date masks using exogenous dates listed above
- [ ] Compute Winkler daily scores, then take means over relevant subsets
- [ ] Compute marginal coverage as scalar
- [ ] Compute Cov_high, Cov_low, Gap using masks
- [ ] Compute Width_high, Width_low using masks
- [ ] Run adaptation speed function for each model x each crisis
- [ ] Run robustness check on adaptation speed with threshold=0.80, persistence=1
- [ ] Produce scatter plot visualization
- [ ] Verify all models use identical volatility threshold and identical crisis date masks