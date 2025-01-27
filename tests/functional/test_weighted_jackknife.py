import unittest

from pyod.models.iforest import IForest

from unquad.data.load import load_breast
from unquad.estimation.weighted_conformal import WeightedConformalDetector
from unquad.strategy.jackknife import Jackknife
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseJackknifeConformalWeighted(unittest.TestCase):
    def test_jackknife_conformal_weighted(self):

        x_train, x_test, y_test = load_breast(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=Jackknife()
        )

        wce.fit(x_train)
        est = wce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.143)
        self.assertEqual(power, 0.857)


if __name__ == "__main__":
    unittest.main()
