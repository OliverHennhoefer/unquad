import unittest

from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control
from unquad.utils.load import load_fraud
from unquad.estimation.weighted_conformal import WeightedConformalDetector
from unquad.strategy.cross_val import CrossValidation
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseSplitConformalWeighted(unittest.TestCase):
    def test_cross_val_conformal_weighted(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=CrossValidation(k=5)
        )

        ce.fit(x_train)
        est = ce.predict(x_test)
        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.0)
        self.assertEqual(power, 0.25)

    def test_cross_val_conformal_plus_weighted(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
        )

        wce.fit(x_train)
        est = wce.predict(x_test)
        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.0)
        self.assertEqual(power, 0.25)


if __name__ == "__main__":
    unittest.main()
