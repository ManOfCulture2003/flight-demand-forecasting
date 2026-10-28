import unittest
import pandas as pd
from flight_demand_forecasting import generate_data,validate,forecast_and_validate

class ForecastTests(unittest.TestCase):
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
