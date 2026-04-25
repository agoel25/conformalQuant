# Complex + Conformal Prediction

Neural point predictors (LSTM, Transformer encoder) wrapped with the
same three conformal methods used in `Simple_CP`.  Tests whether
**base-model complexity** improves interval quality once the
uncertainty-quantification layer is fixed.

If the daily-return mean is approximately unpredictable (efficient
markets), a more expressive base model should not improve interval
quality — the CP layer is doing all the work.  This is the
"Fantazzini secondary question" the paper explores.

## Base predictors

### `lstm` — single-layer LSTM regressor

* `torch.nn.LSTM(input_size=1, hidden_size=50, num_layers=1)` followed
  by a linear head.  Inputs reshaped via `src.data.transform.build_deep_data`
  into `(N, num_lags=5, 1)` tensors.
* Mini-batch Adam, batch_size=32, lr=1e-3, max 50 epochs, gradient
  clipping at norm 1.0, last 15% of training rows held out
  chronologically as validation, early stopping on val MSE
  (patience=5).  NaN/inf rows in the label column (e.g. pre-IPO
  history for tickers like SNAP or GNW) are filtered out before
  fitting.
* Implementation: `src/models/lstm.py`.

### `transformer` — single-layer Transformer encoder

* `torch.nn.TransformerEncoder` with `d_model=32`, `nhead=4`,
  `num_layers=1`, `dim_feedforward=64`, `dropout=0.1`, followed by a
  linear head.  Same `(N, num_lags, 1)` input shape and training loop
  as the LSTM.
* Implementation: `src/models/transformer_encoder.py`.

## Conformal wrappers

`split`, `aci`, `agaci` — same three wrappers as `Simple_CP`.  See
`src/analysis/Simple_CP/README.md` for descriptions.

## Why only LSTM and Transformer?

These two cover the dominant deep-learning architectures used for
sequence forecasting:  recurrent (LSTM) and attention-based
(Transformer encoder).  We deliberately keep them small (50 hidden
LSTM units; `d_model=32` Transformer) because (a) the proposal calls
out the efficient-markets prior — bigger models should not help if
the mean is unpredictable, and (b) the headline test is "does
complexity help?", not "what is the best deep model?".

A hyperparameter sweep would not change the qualitative finding (in
both the COVID 2020 sanity check and the early TCP comparison, all
three base models — LinearAR, LSTM, Transformer — produced
nearly-identical Winkler scores when wrapped with the same CP
method).
