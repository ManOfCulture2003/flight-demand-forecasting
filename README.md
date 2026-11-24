# Flight Demand Forecasting

**Educational portfolio project — default data is synthetic, not real employer, airline, or machinery data.**

Fits ARIMA to monthly demand; checks held-out 12-month error against a seasonal-naive baseline before generating future forecasts.

## Setup (Python 3.11+)

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python flight_demand_forecasting.py
python -m unittest discover -v
```

The program writes generated artifacts under `outputs/` (ignored by Git). To analyze a local CSV instead, pass `--input your_file.csv`.

CSV fields: `Date,Passengers (consecutive months)`. Read the input validation code before substituting your data. No company or customer information should be uploaded without permission.

Input must be a plain, uncompressed CSV of at most 5 MiB. The program checks this limit before parsing or fitting a model. Extract compressed files yourself and check their size first. This keeps accidental large imports from using unbounded input memory; it is not a guarantee about model runtime.

## Technical approach

`outputs/backtest.csv` records each of the 12 held-out months, its actual demand, the ARIMA prediction, and the seasonal-naive prediction. Use these rows to inspect errors behind the aggregate MAE scores. These are holdout predictions, separate from the future values in `forecast.csv`.

Fits ARIMA to monthly demand; checks held-out 12-month error against a seasonal-naive baseline before generating future forecasts.

The automated tests cover reproducibility and core validation/metrics. GitHub Actions runs the tests on pushes and pull requests.

## Limits

The metrics JSON includes seasonal mean absolute scaled error (MASE) for ARIMA and the seasonal-naive forecast. Each holdout MAE is divided by the mean absolute difference between training observations 12 months apart. The denominator uses training data only, and `mase_scale` records its unrounded value. Unlike MAE, MASE has no passenger-count units. A value below 1 means lower error than that in-training seasonal reference; it does not by itself mean ARIMA beat the held-out baseline. Use `model_beats_baseline` for that comparison. If the training scale is zero, both MASE values are JSON `null` and `mase_status` explains why. See the [forecasting textbook's scaled-error definition](https://otexts.com/fpp3/accuracy.html#scaled-errors).

The generated figures and model scores demonstrate code behavior **only**. They are not real-world business improvements, production deployments, or evidence of predictive accuracy on actual organizations. More real-world validation, data quality checks, and monitoring are needed before deployment.

## Forecast uncertainty (new)

`outputs/forecast.csv` now includes `predicted_passengers`, `lower_95`, and `upper_95` for each future month. The chart shades the same 95% ARIMA model interval. These are **model-based intervals**, not guarantees: they rely on ARIMA assumptions and do not account for sudden schedule, policy, economic or route changes. Bounds may fall below zero because the fitted model is unconstrained; do not interpret negative demand as physically possible. The intervals are for future forecasts, not for the 12-month held-out backtest.

Run `python -m unittest discover -s tests -v` to check the interval shape, dates, and CSV/chart outputs.
