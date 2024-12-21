import unittest

from pyod.models.iforest import IForest

from unquad.strategy.bootstrap import BootstrapConformal
from unquad.utils.enums.dataset import Dataset
from unquad.data.loader import DataLoader
from unquad.estimator.estimator import ConformalDetector
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseBootstrapConformal(unittest.TestCase):
    def test_bootstrap_conformal_compute_n_bootstraps(self):

        dl = DataLoader(dataset=Dataset.SHUTTLE)
        x_train, x_test, y_test = dl.get_example_setup(random_state=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=BootstrapConformal(resampling_ratio=0.995, n_calib=1_000),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(len(ce.calibration_set), 1_000)
        self.assertEqual(fdr, 0.075)
        self.assertEqual(power, 0.925)

    def test_bootstrap_conformal_compute_n_calib(self):
        dl = DataLoader(dataset=Dataset.SHUTTLE)
        x_train, x_test, y_test = dl.get_example_setup(random_state=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=BootstrapConformal(resampling_ratio=0.99, n_bootstraps=15),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(len(ce.calibration_set), 3419)
        self.assertEqual(fdr, 0.261)
        self.assertEqual(power, 0.739)

    def test_bootstrap_conformal_compute_resampling_ratio(self):
        dl = DataLoader(dataset=Dataset.SHUTTLE)
        x_train, x_test, y_test = dl.get_example_setup(random_state=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=BootstrapConformal(n_calib=1_000, n_bootstraps=25),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(len(ce.calibration_set), 1000)
        self.assertEqual(fdr, 0.175)
        self.assertEqual(power, 0.825)


if __name__ == "__main__":
    unittest.main()
