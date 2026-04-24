# Conformal Prediction for Financial Time Series

A systematic comparison of six conformal-prediction
methods (Split, ACI, EnbPI, AgACI, TCP, TCP-RM) against the industry baselines
(FHS, GARCH) on daily log-returns through three crisis regimes (2008, 2020,
2022). Final report at `report/report.pdf`.

## Layout

    src/
      conformal/      base + 6 conformal wrappers + FHS + GARCH baselines
      models/         5 base predictors (LinAR, HistBin, Trees, LSTM, Transformer)
      metrics/        Winkler, coverage, width, adaptation-speed metrics
      data/           yfinance loader and lag-window transform
      analysis/       per-model + per-baseline notebooks plus the aggregator
    scripts/
      run_grid.py         reproduce the full 180-CSV evaluation grid
      build_artifacts.py  aggregate metrics + regenerate paper figures
    report/           LaTeX source + figures + compiled PDF
    results/          per-(model, wrapper, ticker, regime) interval CSVs
                      (gitignored; produced by run_grid.py)

## Reproducing the results

```bash
pip install -r requirements.txt
python scripts/run_grid.py            # 180 CSVs, ~15 min
python scripts/build_artifacts.py     # metrics_long.csv + report figures
cd report && pdflatex report.tex && biber report && pdflatex report.tex && pdflatex report.tex
```

`scripts/run_grid.py --only {simple,deep,baselines,all}` runs subsets of the grid.

## Headline result

TCP and TCP-RM are the only methods that match nominal 0.90 coverage on
average (Marg.\ Cov $\geq 0.889$, low-vol/high-vol coverage gap $\leq 0.089$),
beat both industry baselines on Winkler score, and recover coverage in
$\sim 17$ trading days after the 2008 Lehman onset versus 109--137 days for
every other method. The advantage holds across both neural base predictors
(LSTM and Transformer give within 0.013 Winkler), supporting the claim that
TCP's rolling-window calibration drives the win rather than predictor sharpness.

See `report/report.pdf` for full results and discussion.
