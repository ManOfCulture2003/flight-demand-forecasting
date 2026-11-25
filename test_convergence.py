import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import flight_demand_forecasting as app


class ConvergenceTests(unittest.TestCase):
    def test_separate_fit_statuses_survive_json_export(self):
        for holdout, final in [(True, False), (False, True), (None, None)]:
            with self.subTest(holdout=holdout, final=final):
                first, second = Mock(), Mock()
                first.mle_retvals = {} if holdout is None else {'converged': np.bool_(holdout)}
                second.mle_retvals = {} if final is None else {'converged': np.bool_(final)}
                first.forecast.return_value = pd.Series([200.] * 12)
                second.get_forecast.return_value = SimpleNamespace(
                    predicted_mean=pd.Series([200.] * 6),
                    conf_int=lambda alpha: pd.DataFrame({'low': [100.] * 6, 'high': [300.] * 6}),
                )
                with patch.object(app, 'ARIMA') as arima:
                    arima.return_value.fit.side_effect = [first, second]
                    _, metrics = app.forecast_and_validate(app.generate_data())
                exported = json.loads(json.dumps(metrics))
                self.assertIs(exported['fit_convergence']['holdout'], holdout)
                self.assertIs(exported['fit_convergence']['full_history'], final)

    def test_missing_optimizer_metadata_is_unknown(self):
        self.assertIsNone(app.fit_converged(SimpleNamespace()))
        self.assertIsNone(app.fit_converged(SimpleNamespace(mle_retvals={'converged': 'False'})))


if __name__ == '__main__':
    unittest.main()
