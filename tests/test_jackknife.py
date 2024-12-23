import unittest

from pyod.models.iforest import IForest

from unquad.strategy.jackknife import JackknifeConformal
from unquad.utils.enums.dataset import Dataset
from unquad.utils.data.loader import DataLoader
from unquad.estimator.detector import ConformalDetector
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseJackknifeConformal(unittest.TestCase):
    def test_jackknife_conformal(self):

        dl = DataLoader(dataset=Dataset.BREAST)
        x_train, x_test, y_test = dl.get_example_setup(random_state=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"), strategy=JackknifeConformal()
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.222)
        self.assertEqual(power, 0.778)


if __name__ == "__main__":
    unittest.main()
