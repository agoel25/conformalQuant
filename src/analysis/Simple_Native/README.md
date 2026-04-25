# Simple + Native intervals

Methods that produce prediction intervals **natively** from data
structure — no conformal wrapper.  Used to test whether a simple
non-conformal interval method can compete with the
volatility-baselines.

## `hist_binning` — Histogram-binning quantiles

* Source: classical non-parametric density-based interval idea; the
  same intuition as kernel-density-based VaR but with discrete bins.
  No specific paper "owns" the method; closest formal write-up is
  Wisniewski et al. (2020) "Application of conformal prediction
  interval estimations to market makers".
* Fit `KBinsDiscretizer(n_bins=10, strategy='quantile')` on the mean of
  the most recent `num_lags=5` log-returns.  For each bin, store every
  next-step return seen in training.
* Predict: find the bin that the current lag-window mean falls in, take
  the empirical α/2 and 1−α/2 quantiles of the stored returns from that
  bin.  Falls back to global quantiles when a bin is too sparse.
* Implementation: `src/conformal/hist_binning_native.py`.

## `tree_based` — Quantile Regression Forest (Meinshausen 2006)

* Source: **Nicolai Meinshausen, "Quantile Regression Forests", JMLR
  2006**.  This is the canonical leaf-based interval method.
* Fit a standard `RandomForestRegressor` on lag-window features →
  next-step return.  Record, per tree, which leaf each training point
  landed in.
* At test time:  drop the test point through every tree to get its
  leaf indices.  For each tree, training points in the same leaf get
  weight `1 / leaf_size`; sum across trees and normalise.  Read the
  α/2 and 1−α/2 quantiles off the **weighted** empirical CDF of the
  stored training labels.
* This is exactly the user's intuition ("which leaves does the test
  point fall into; aggregate training labels in those leaves; take
  percentiles") — formalised by Meinshausen.  It is the de facto
  standard tree-based interval method.
* Implementation: `src/conformal/tree_native.py`.

## Why these two?

Together they probe two complementary "simple" structures:
*histogram-binning* uses a **single 1-D summary** of the lag window
(its mean) and looks up a bucket;
*Quantile Regression Forests* use the full multi-dimensional lag
window and let trees discover the interaction structure.  Comparing
them tells us whether the lag structure beyond the mean is doing useful
work for interval estimation on equity returns.
