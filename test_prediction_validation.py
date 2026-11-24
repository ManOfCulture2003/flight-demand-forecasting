import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import flight_demand_forecasting as app


class PredictionValidationTests(unittest.TestCase):
    def test_finite_predictions_with_overflowing_error_are_rejected(self):
        series = app.generate_data()
        series.iloc[:] = 1e308
        with patch.object(app, 'ARIMA') as arima, np.errstate(over='ignore'):
            arima.return_value.fit.return_value.forecast.return_value = pd.Series([-1e308] * 12)
            with self.assertRaisesRegex(ValueError, 'non-finite errors'):
                app.forecast_and_validate(series)

    def test_nonfinite_predictions_rejected(self):
        for stage in ('holdout', 'future'):
            for invalid in (np.nan, np.inf, -np.inf):
                with self.subTest(stage=stage, invalid=invalid):
                    backtest = pd.Series([200.0] * 12)
                    future = pd.Series([200.0] * 6)
                    if stage == 'holdout':
                        backtest.iloc[0] = invalid
                    else:
                        future.iloc[0] = invalid
                    fitted = Mock()
                    fitted.forecast.return_value = backtest
                    fitted.get_forecast.return_value.predicted_mean = future
                    fitted.get_forecast.return_value.conf_int.return_value = pd.DataFrame({
                        'lower': [100.] * 6, 'upper': [300.] * 6,
                    })
                    with patch.object(app, 'ARIMA') as arima:
                        arima.return_value.fit.return_value = fitted
                        with self.assertRaisesRegex(ValueError, 'non-finite'):
                            app.forecast_and_validate(app.generate_data())

    def test_model_failure_is_cli_error_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'out'
            output.mkdir()
            saved = output / 'forecast.csv'
            saved.write_text('previous forecast')
            with patch.object(app, 'forecast_and_validate', side_effect=ValueError('Model returned non-finite predictions')):
                with contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as raised:
                        app.main(['--output', str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('non-finite', error.getvalue())
            self.assertNotIn('Traceback', error.getvalue())
            self.assertEqual(saved.read_text(), 'previous forecast')
            self.assertEqual(list(output.iterdir()), [saved])


if __name__ == '__main__':
    unittest.main()
