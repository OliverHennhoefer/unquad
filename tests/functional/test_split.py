import unittest

from pyod.models.iforest import IForest

from unquad.data.load import load_fraud, load_shuttle
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.split import Split
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseSplitConformal(unittest.TestCase):
    def test_split_conformal_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(calib_size=2_000)
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.134)
        self.assertEqual(power, 0.866)

    def test_split_conformal_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(calib_size=1_000)
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.25)
        self.assertEqual(power, 0.75)


if __name__ == "__main__":
    unittest.main()
