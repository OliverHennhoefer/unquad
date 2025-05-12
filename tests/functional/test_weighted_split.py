import unittest

from pyod.models.iforest import IForest

from unquad.data.load import load_fraud, load_shuttle
from unquad.estimation.weighted_conformal import WeightedConformalDetector
from unquad.strategy.split import Split
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseSplitConformalWeighted(unittest.TestCase):
    def test_split_conformal_weighted_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(calib_size=2_000)
        )

        wce.fit(x_train)
        est = wce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.12)
        self.assertEqual(power, 0.73)

    def test_split_conformal_weighted_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(calib_size=1_000)
        )

        wce.fit(x_train)
        est = wce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.19)
        self.assertEqual(power, 0.98)


if __name__ == "__main__":
    unittest.main()
