import unittest

from pyod.models.iforest import IForest
from unquad.data.load import load_shuttle
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.bootstrap import Bootstrap
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseBootstrapConformal(unittest.TestCase):
    def test_bootstrap_conformal_compute_n_bootstraps(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.995, n_calib=1_000),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(len(ce.calibration_set), 1_000)
        self.assertEqual(fdr, 0.075)
        self.assertEqual(power, 0.98)

    def test_bootstrap_conformal_compute_n_calib(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.99, n_bootstraps=15),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(len(ce.calibration_set), 3419)
        self.assertEqual(fdr, 0.261)
        self.assertEqual(power, 0.99)

    def test_bootstrap_conformal_compute_resampling_ratio(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_calib=1_000, n_bootstraps=25),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(len(ce.calibration_set), 1000)
        self.assertEqual(fdr, 0.175)
        self.assertEqual(power, 0.99)


if __name__ == "__main__":
    unittest.main()
