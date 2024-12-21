import unittest

from pyod.models.iforest import IForest

from unquad.strategy.cross_val import CrossValidationConformal
from unquad.utils.enums.dataset import Dataset
from tests.datasets.loader import DataLoader
from unquad.estimator.estimator import ConformalDetector
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseSplitConformal(unittest.TestCase):
    def test_cross_val_conformal(self):

        dl = DataLoader(dataset=Dataset.FRAUD)
        x_train, x_test, y_test = dl.get_example_setup(random_state=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"), strategy=CrossValidationConformal(k=5)
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.115)
        self.assertEqual(power, 0.885)

    def test_cross_val_conformal_plus(self):

        dl = DataLoader(dataset=Dataset.FRAUD)
        x_train, x_test, y_test = dl.get_example_setup(random_state=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidationConformal(k=5, plus=True),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.141)
        self.assertEqual(power, 0.859)


if __name__ == "__main__":
    unittest.main()
