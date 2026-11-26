import unittest
import numpy as np
from flight_demand_forecasting import forecast_and_validate, generate_data


class ScaledErrorTests(unittest.TestCase):
    def test_scale_uses_training_seasonal_differences(self):
        series = generate_data()
        _, report = forecast_and_validate(series)
        train = series.iloc[:-12].to_numpy()
        expected_scale = np.abs(train[12:] - train[:-12]).mean()
        rows = report['holdout_predictions']
        expected_mae = np.mean([abs(r['actual_passengers'] - r['arima_prediction']) for r in rows])
        self.assertAlmostEqual(report['mase_scale'], expected_scale)
        self.assertEqual(report['arima_mase'], round(expected_mae / expected_scale, 4))
        changed = series.copy()
        changed.iloc[-12:] += 1000
        _, updated = forecast_and_validate(changed)
        self.assertEqual(report['mase_scale'], updated['mase_scale'])
        self.assertGreater(updated['arima_mase'], report['arima_mase'])

    def test_zero_scale_is_unavailable_not_zero_error(self):
        series = generate_data()
        series.iloc[:] = np.tile(np.arange(1., 13.), 5)
        _, report = forecast_and_validate(series)
        self.assertEqual(report['mase_scale'], 0)
        self.assertIsNone(report['arima_mase'])
        self.assertIsNone(report['seasonal_naive_mase'])
        self.assertEqual(report['mase_status'], 'undefined_zero_training_scale')
