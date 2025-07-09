import unittest

from scipy.stats import false_discovery_control

from pyod.models.iforest import IForest
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.split import Split
from unquad.utils.data.load import load_fraud, load_shuttle
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power


class TestCaseSplitConformal(unittest.TestCase):
    def test_split_conformal_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(calib_size=2_000)
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.134)
        self.assertEqual(power, 0.84)

    def test_split_conformal_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(calib_size=1_000)
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.25)
        self.assertEqual(power, 0.99)


if __name__ == "__main__":
    unittest.main()
