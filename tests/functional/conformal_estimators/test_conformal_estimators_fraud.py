import unittest
from pyod.models.iforest import IForest

from unquad.enums.method import Method
from unquad.enums.dataset import Dataset
from unquad.datasets.loader import DataLoader
from unquad.enums.adjustment import Adjustment
from unquad.estimator.conformal_estimator import ConformalEstimator
from unquad.estimator.split_configuration import SplitConfiguration
from unquad.evaluation.metrics import false_discovery_rate, statistical_power


class TestConformalEstimatorsIonosphere(unittest.TestCase):

    dl = DataLoader(dataset=Dataset.FRAUD)
    x_train, x_test, y_test = dl.get_example_setup()

    def test_split_conformal(self):
        sc = SplitConfiguration(n_split=2_000)
        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.SPLIT_CONFORMAL,
            split=sc,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.1,
            seed=1,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.09)
        self.assertEqual(power, 0.91)

    def test_cv(self):
        sc = SplitConfiguration(n_split=10)
        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.CV,
            split=sc,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.1,
            seed=1,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.031)
        self.assertEqual(power, 0.969)

    def test_jackknife_after_bootstrap(self):
        sc = SplitConfiguration(n_bootstraps=30, n_calib=1_000)
        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.JACKKNIFE_AFTER_BOOTSTRAP,
            split=sc,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.1,
            seed=1,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.047)
        self.assertEqual(power, 0.953)

    def test_jackknife_plus_after_bootstrap(self):
        sc = SplitConfiguration(n_bootstraps=40, n_split=0.975)
        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
            split=sc,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.1,
            seed=1,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.026)
        self.assertEqual(power, 0.974)


if __name__ == "__main__":
    unittest.main()
