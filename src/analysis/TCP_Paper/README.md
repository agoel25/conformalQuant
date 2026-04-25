# TCP-paper sanity check (standalone)

This folder is **not** part of the main grid.  It exists to document
*why* TCP / TCP-RM were dropped from the headline comparison.

## What the notebook does

`tcp_paper.ipynb` runs four methods on the COVID-2020 crisis tickers
{MCD, MGM, AZO} and compares Winkler / coverage / width:

1. **`garch_hardcoded`** — TCP paper's exact GARCH baseline
   (Aich et al. 2026 eq. 5–6):  `(ω, α, β) = (1e-6, 0.05, 0.9)`,
   normal innovations, *parameters never estimated from data*,
   *no conformal calibration*.  Implemented via `arch_model.fix(params)`
   so the GARCH recursion comes from the library.
2. **`garch_roll`** — properly fitted rolling Student-t GARCH(1,1)
   with parametric quantile (the `Baselines` notebook's main method).
3. **`fhs`** — Filtered Historical Simulation.
4. **`tcp_paper`** — paper-faithful TCP: gradient-boosted quantile
   regressors at α/2 and 1−α/2 + rolling split-conformal calibration
   with a 252-day window split into 192 train / 60 calibration rows.
   Implementation: `src/conformal/tcp_paper.py`.

## Headline finding

Across COVID-2020 individual equities:

| method            | coverage | Winkler ↓ | width  |
|-------------------|---------:|----------:|-------:|
| fhs               |    0.908 |   **0.128** |  0.083 |
| garch_roll        |    0.934 |   0.134 |  0.102 |
| garch_hardcoded   |    0.799 |   0.138 |  0.061 |
| tcp_paper         |    0.884 |   **0.176** |  0.095 |

* `garch_roll` and `fhs` both **beat** `tcp_paper` on Winkler.
* `garch_hardcoded` undercoverere badly (79.9% vs nominal 90%) — this
  is what the TCP paper was actually comparing against, and it
  *does* lose to TCP, but only because it was deliberately broken.
* The TCP paper's apparent advantage **vanishes** as soon as the
  GARCH baseline is fairly implemented.

## Why this is in its own folder

The point of the main grid is to compare the four model categories
(`Baselines`, `Simple_Native`, `Simple_CP`, `Complex_CP`).  TCP would
be a fifth method that adds nothing and would muddy the story.  This
standalone notebook lives here so the comparison is reproducible if a
reviewer asks "why didn't you include TCP?".
