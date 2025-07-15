import unittest

from pyod.models.deep_svdd import DeepSVDD
from nonconform.estimation.standard_conformal import StandardConformalDetector
from nonconform.strategy.split import Split


class TestUnsupportedDetector(unittest.TestCase):
    def test_unsupported_detector(self):
        with self.assertRaises(ValueError) as _:
            StandardConformalDetector(detector=DeepSVDD(n_features=1), strategy=Split())


if __name__ == "__main__":
    unittest.main()
