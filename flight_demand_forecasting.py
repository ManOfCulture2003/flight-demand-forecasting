"""Reproducible monthly ARIMA forecasting with holdout and seasonal-naive baseline."""
import argparse
import json
from pathlib import Path
from numbers import Integral
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def generate_data(seed=42, months=60):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=months, freq="MS")
    t = np.arange(months)
    values = 180 + 1.7*t + 22*np.sin(2*np.pi*t/12) + rng.normal(0, 5, months)
    return pd.Series(values, index=dates, name="passengers", dtype=float)

def validate(series):
    if len(series) < 30 or series.isna().any() or (series < 0).any():
        raise ValueError("At least 30 valid, nonnegative monthly observations required")
    if not np.isfinite(series).all():
        raise ValueError("Passenger counts must be finite")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Date index required")
    series = series.sort_index()
    if not series.index.equals(pd.date_range(series.index.min(), periods=len(series), freq="MS")):
        raise ValueError("Dates must be consecutive unique month starts")
    return series

def forecast_and_validate(series, horizon=6):
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or not 1 <= horizon <= 24:
        raise ValueError("horizon must be an integer between 1 and 24")
    series = validate(series)
    train, test = series.iloc[:-12], series.iloc[-12:]
    backtest = ARIMA(train, order=(1,1,1)).fit().forecast(steps=12)
    baseline = train.iloc[-12:].to_numpy()
    model_mae = float(np.mean(np.abs(test.to_numpy()-backtest.to_numpy())))
    baseline_mae = float(np.mean(np.abs(test.to_numpy()-baseline)))
    future = ARIMA(series, order=(1,1,1)).fit().forecast(steps=horizon)
    return future, {"holdout_months":12, "arima_mae":round(model_mae,3),
                    "seasonal_naive_mae":round(baseline_mae,3),
                    "model_beats_baseline":model_mae < baseline_mae}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional monthly CSV: Date,Passengers")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 1 <= args.horizon <= 24:
        parser.error("--horizon must be between 1 and 24")
    if args.input:
        df = pd.read_csv(args.input)
        if not {"Date","Passengers"}.issubset(df.columns):
            parser.error("CSV needs Date and Passengers")
        series = pd.Series(pd.to_numeric(df["Passengers"]).to_numpy(),
            index=pd.to_datetime(df["Date"]), name="passengers")
    else:
        series = generate_data(args.seed)
    future, metrics = forecast_and_validate(series, args.horizon)
    args.output.mkdir(parents=True, exist_ok=True)
    future.rename("predicted_passengers").to_csv(args.output/"forecast.csv", index_label="Date")
    (args.output/"backtest_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n",encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 4))
    series.plot(ax=ax, label="History")
    future.plot(ax=ax, label="ARIMA", style="--")
    ax.set(title="Monthly demand (synthetic example)", ylabel="Demand")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output/"flight_demand_forecast.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
