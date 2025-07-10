import unittest

from scipy.stats import false_discovery_control

from pyod.models.iforest import IForest
from unquad.estimation.standard_conformal import StandardConformalDetector
from unquad.strategy.jackknife import Jackknife
from unquad.utils.data.load import load_breast
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power


class TestCaseJackknifeConformal(unittest.TestCase):
    def test_jackknife_conformal(self):
        x_train, x_test, y_test = load_breast(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"), strategy=Jackknife()
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.222)
        self.assertEqual(power, 1.0)


if __name__ == "__main__":
    unittest.main()
