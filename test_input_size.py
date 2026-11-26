import contextlib
import gzip
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import flight_demand_forecasting as app


class InputSizeTests(unittest.TestCase):
    def test_csv_at_size_limit_still_exports_forecast(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'data.csv'
            csv = app.generate_data().rename('Passengers').to_csv(index_label='Date').encode()
            source.write_bytes(csv + b'\n' * (5 * 1024 * 1024 - len(csv)))
            output = Path(folder) / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                app.main(['--input', str(source), '--output', str(output), '--horizon', '1'])
            self.assertEqual(len(app.pd.read_csv(output / 'forecast.csv')), 1)

    def test_oversized_csv_rejected_before_parsing_or_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'large.csv'
            source.write_bytes(b'x' * (5 * 1024 * 1024 + 1))
            output = Path(folder) / 'out'
            with contextlib.redirect_stderr(io.StringIO()) as error, patch.object(
                app.pd, 'read_csv'
            ) as read, patch.object(app, 'ARIMA') as fit:
                with self.assertRaises(SystemExit) as raised:
                    app.main(['--input', str(source), '--output', str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('5 MiB', error.getvalue())
            read.assert_not_called()
            fit.assert_not_called()
            self.assertFalse(output.exists())

    def test_compressed_input_is_not_implicitly_expanded(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'data.csv.gz'
            csv = app.generate_data().rename('Passengers').to_csv(index_label='Date')
            source.write_bytes(gzip.compress(csv.encode()))
            with contextlib.redirect_stderr(io.StringIO()), patch.object(app, 'ARIMA') as fit:
                with self.assertRaises(SystemExit) as raised:
                    app.main(['--input', str(source), '--output', str(Path(folder) / 'out')])
            self.assertEqual(raised.exception.code, 2)
            fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
