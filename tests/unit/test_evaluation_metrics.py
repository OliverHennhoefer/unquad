import unittest
import numpy as np

from noncon.evaluation.metrics import false_discovery_rate, statistical_power


class TestEvaluationMetrics(unittest.TestCase):

    # Anomaly Estimates
    EST1 = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1])
    EST2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Ground Truth
    ACT1 = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1])
    ACT2 = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1])
    ACT3 = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0])

    # FDR Calculation
    def test_metric_false_discovery_rate1(self):
        """
        All estimates correct.
        """
        fdr = false_discovery_rate(self.ACT1, self.EST1)
        self.assertEqual(fdr, 0.0)

    def test_metric_false_discovery_rate2(self):
        """
        Some estimates correct.
        """
        fdr = false_discovery_rate(self.ACT2, self.EST1)
        self.assertEqual(fdr, 0.444)

    def test_metric_false_discovery_rate3(self):
        """
        No estimates correct.
        """
        fdr = false_discovery_rate(self.ACT3, self.EST1)
        self.assertEqual(fdr, 1.0)

    def test_metric_false_discovery_rate4(self):
        """
        No outlier found.
        """
        fdr = false_discovery_rate(self.ACT3, self.EST2)
        self.assertEqual(fdr, 0.0)

    # Power Calculation
    def test_metric_statistical_power1(self):
        """
        All estimates correct.
        """
        power = statistical_power(self.ACT1, self.EST1)
        self.assertEqual(power, 1.0)

    def test_metric_statistical_power2(self):
        """
        Some estimates correct.
        """
        power = statistical_power(self.ACT2, self.EST1)
        self.assertEqual(power, 0.556)

    def test_metric_statistical_power3(self):
        """
        No estimates correct.
        """
        power = statistical_power(self.ACT3, self.EST1)
        self.assertEqual(power, 0.0)


if __name__ == "__main__":
    unittest.main()
