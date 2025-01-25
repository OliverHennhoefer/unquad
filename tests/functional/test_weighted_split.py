import unittest

from pyod.models.iforest import IForest

from unquad.data.load import load_fraud
from unquad.estimation.weighted_conformal import WeightedConformalDetector
from unquad.strategy.split import Split
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseSplitConformal(unittest.TestCase):
    def test_split_conformal(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(calib_size=2_000)
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.107)
        self.assertEqual(power, 0.893)


if __name__ == "__main__":
    unittest.main()
