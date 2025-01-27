import unittest

from pyod.models.iforest import IForest

from unquad.data.load import load_shuttle
from unquad.estimation.weighted_conformal import WeightedConformalDetector
from unquad.strategy.bootstrap import Bootstrap
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseBootstrapConformalWeighted(unittest.TestCase):
    def test_bootstrap_conformal_compute_n_bootstraps_weighted(self):

        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.995, n_calib=1_000),
        )

        wce.fit(x_train)
        est = wce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(len(wce.calibration_set), 1_000)
        self.assertEqual(fdr, 0.067)
        self.assertEqual(power, 0.933)

    def test_bootstrap_conformal_compute_n_calib_weighted(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.99, n_bootstraps=15),
        )

        wce.fit(x_train)
        est = wce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(len(wce.calibration_set), 3419)
        self.assertEqual(fdr, 0.214)
        self.assertEqual(power, 0.786)

    def test_bootstrap_conformal_compute_resampling_ratio_weighted(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_calib=1_000, n_bootstraps=25),
        )

        wce.fit(x_train)
        est = wce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(len(wce.calibration_set), 1000)
        self.assertEqual(fdr, 0.067)
        self.assertEqual(power, 0.933)


if __name__ == "__main__":
    unittest.main()
