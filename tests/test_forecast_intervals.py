"""Regression checks for future ARIMA uncertainty bands (synthetic data only)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from flight_demand_forecasting import generate_data, forecast_and_validate, main


class ForecastIntervalTests(unittest.TestCase):
    def test_intervals_have_expected_dates_and_contain_point_forecast(self):
        series = generate_data(seed=42, months=48)
        forecast, metrics, bounds = forecast_and_validate(series, horizon=6, include_intervals=True)
        self.assertEqual(len(forecast), 6)
        self.assertEqual(list(bounds.columns), ["lower_95", "upper_95"])
        self.assertTrue(bounds.index.equals(forecast.index))
        self.assertEqual(forecast.index[0], series.index[-1] + pd.offsets.MonthBegin(1))
        self.assertTrue(np.isfinite(bounds.to_numpy()).all())
        self.assertTrue((bounds["lower_95"] <= forecast).all())
        self.assertTrue((forecast <= bounds["upper_95"]).all())
        self.assertEqual(metrics["holdout_months"], 12)

    def test_existing_two_result_api_still_works(self):
        future, metrics = forecast_and_validate(generate_data(months=36), horizon=1)
        self.assertEqual(len(future), 1)
        self.assertIn("arima_mae", metrics)

    def test_cli_writes_intervals_with_forecast_and_chart(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reports"
            main(["--output", str(output), "--horizon", "3"])
            table = pd.read_csv(output / "forecast.csv")
            self.assertEqual(len(table), 3)
            self.assertEqual(list(table.columns), ["Date", "predicted_passengers", "lower_95", "upper_95"])
            self.assertTrue((table["lower_95"] <= table["predicted_passengers"]).all())
            self.assertTrue((table["predicted_passengers"] <= table["upper_95"]).all())
            self.assertTrue((output / "flight_demand_forecast.png").is_file())


if __name__ == "__main__":
    unittest.main()
