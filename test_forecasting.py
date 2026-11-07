import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from flight_demand_forecasting import generate_data,validate,forecast_and_validate,main,plt

class ForecastTests(unittest.TestCase):
    def test_bad_csv_reports_cli_error_before_fitting(self):
        cases = ("Date,Passengers\n2026-01-01,bad\n", "Date,Passengers\nnot-a-date,1\n", None)
        for content in cases:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / "input.csv"
                if content is not None:
                    source.write_text(content)
                error = io.StringIO()
                with contextlib.redirect_stderr(error), patch("flight_demand_forecasting.ARIMA") as arima:
                    with self.assertRaises(SystemExit) as raised:
                        main(["--input", str(source), "--output", str(root / "out")])
                    self.assertEqual(raised.exception.code, 2)
                    arima.assert_not_called()
                self.assertIn("error:", error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())
                self.assertFalse((root / "out").exists())

    def test_holdout_values_cannot_leak_into_backtest_predictions(self):
        original = generate_data()
        changed = original.copy()
        changed.iloc[-12:] += 1000
        before = changed.copy()
        _, first = forecast_and_validate(original)
        _, second = forecast_and_validate(changed)
        first_rows = pd.DataFrame(first['holdout_predictions'])
        second_rows = pd.DataFrame(second['holdout_predictions'])
        for prediction in ('arima_prediction', 'seasonal_naive_prediction'):
            np.testing.assert_allclose(first_rows[prediction], second_rows[prediction])
        np.testing.assert_allclose(second_rows['actual_passengers'] - first_rows['actual_passengers'], 1000)
        self.assertGreater(second['arima_mae'], first['arima_mae'])
        pd.testing.assert_series_equal(changed, before)

    def test_exports_cannot_overwrite_source_csv(self):
        for name in ('forecast.csv', 'backtest.csv'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / name
                generate_data().rename('Passengers').to_csv(source, index_label='Date')
                original = source.read_bytes()
                with patch('flight_demand_forecasting.ARIMA') as arima:
                    with self.assertRaisesRegex(ValueError, 'overwrite'):
                        main(['--input', str(source), '--output', str(root)])
                    arima.assert_not_called()
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(list(root.iterdir()), [source])

    def test_export_identifies_input_source(self):
        for source in ("synthetic", "user-supplied CSV"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                args = ["--output", str(root / "export"), "--horizon", "1"]
                if source == "user-supplied CSV":
                    data = generate_data()
                    data.rename("Passengers").to_csv(root / "input.csv", index_label="Date")
                    args += ["--input", str(root / "input.csv")]
                fig, ax = plt.subplots()
                with contextlib.redirect_stdout(io.StringIO()), patch(
                        "flight_demand_forecasting.plt.subplots", return_value=(fig, ax)):
                    main(args)
                report = json.loads((root / "export" / "backtest_metrics.json").read_text())
                self.assertEqual(report["data_source"], source)
                self.assertEqual(ax.get_title(), f"Monthly demand ({source})")
                self.assertTrue((root / "export" / "flight_demand_forecast.png").is_file())
                backtest = pd.read_csv(root / "export" / "backtest.csv")
                self.assertEqual(len(backtest), 12)
                self.assertEqual(list(backtest.columns), ["Date", "actual_passengers", "arima_prediction", "seasonal_naive_prediction"])
                expected = generate_data()
                np.testing.assert_allclose(backtest["actual_passengers"], expected.iloc[-12:])
                np.testing.assert_allclose(backtest["seasonal_naive_prediction"], expected.iloc[-24:-12])
                self.assertEqual(backtest["Date"].tolist(), expected.index[-12:].strftime("%Y-%m-%d").tolist())
                mae = np.abs(backtest["actual_passengers"] - backtest["arima_prediction"]).mean()
                self.assertAlmostEqual(mae, report["arima_mae"], places=3)
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
