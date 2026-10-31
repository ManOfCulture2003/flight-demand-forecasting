import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from flight_demand_forecasting import generate_data,validate,forecast_and_validate

class ForecastTests(unittest.TestCase):
    def test_infinite_demand_rejected_before_model_fit(self):
        series = generate_data()
        series.iloc[0] = np.inf
        with patch("flight_demand_forecasting.ARIMA") as arima:
            with self.assertRaisesRegex(ValueError, "finite"):
                forecast_and_validate(series)
            arima.assert_not_called()

    def test_invalid_horizon_rejected_before_model_fit(self):
        for horizon in (0, -1, 25, 1.5, True):
            with self.subTest(horizon=horizon):
                with patch("flight_demand_forecasting.ARIMA") as arima:
                    with self.assertRaisesRegex(ValueError, "horizon"):
                        forecast_and_validate(generate_data(), horizon)
                    arima.assert_not_called()
    def test_repeatability(self):
        pd.testing.assert_series_equal(generate_data(),generate_data())
    def test_missing_month(self):
        d=generate_data()
        with self.assertRaises(ValueError):validate(d.drop(d.index[10]))
    def test_baseline(self):
        future,report=forecast_and_validate(generate_data(),3)
        self.assertEqual(len(future),3)
        self.assertIn("seasonal_naive_mae",report)

if __name__=="__main__":unittest.main()
