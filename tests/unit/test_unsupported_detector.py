import unittest

from pyod.models.deep_svdd import DeepSVDD

from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.split import Split


class TestUnsupportedDetector(unittest.TestCase):
    def test_unsupported_detector(self):

        with self.assertRaises(ValueError) as _:
            ConformalDetector(detector=DeepSVDD(n_features=1), strategy=Split())


if __name__ == "__main__":
    unittest.main()
