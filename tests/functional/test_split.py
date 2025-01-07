import unittest

from pyod.models.iforest import IForest

from unquad.utils.enums import Dataset
from unquad.data.loader import DataLoader
from unquad.strategy.split import SplitConformal
from unquad.estimator.detector import ConformalDetector
from unquad.estimator.configuration import DetectorConfig
from unquad.utils.metrics import false_discovery_rate, statistical_power


class TestCaseSplitConformal(unittest.TestCase):
    def test_split_conformal(self):

        dl = DataLoader(dataset=Dataset.FRAUD)
        x_train, x_test, y_test = dl.get_example_setup(random_state=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"), strategy=SplitConformal(calib_size=1_000)
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.096)
        self.assertEqual(power, 0.904)

    def test_split_conformal_force_anomaly(self):

        dl = DataLoader(dataset=Dataset.FRAUD)
        x_train, x_test, y_test = dl.get_example_setup(random_state=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=SplitConformal(calib_size=500),
            config=DetectorConfig(force_anomaly=True),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        fdr = false_discovery_rate(y=y_test, y_hat=est)
        power = statistical_power(y=y_test, y_hat=est)

        self.assertEqual(fdr, 0.122)
        self.assertEqual(power, 0.878)


if __name__ == "__main__":
    unittest.main()
