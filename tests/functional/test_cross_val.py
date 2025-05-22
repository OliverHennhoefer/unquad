import unittest

from pyod.models.iforest import IForest
from unquad.data.load import load_fraud
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.cross_val import CrossValidation
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseSplitConformal(unittest.TestCase):
    def test_cross_val_conformal(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"), strategy=CrossValidation(k=5)
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.115)
        self.assertEqual(power, 0.77)

    def test_cross_val_conformal_plus(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.141)
        self.assertEqual(power, 0.79)


if __name__ == "__main__":
    unittest.main()
