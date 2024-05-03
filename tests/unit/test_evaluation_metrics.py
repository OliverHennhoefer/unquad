import unittest
import numpy as np

from unquad.evaluation.metrics import false_discovery_rate, statistical_power


class TestEvaluationMetrics(unittest.TestCase):

    estimates_variant1 = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1])
    estimates_variant2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    label_variant_1 = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1])
    label_variant_2 = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1])
    label_variant_3 = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0])

    def test_metric_false_discovery_rate_all_correct(self):
        fdr = false_discovery_rate(self.label_variant_1, self.estimates_variant1)
        self.assertEqual(fdr, 0.0)

    def test_metric_false_discovery_rate_some_correct(self):
        fdr = false_discovery_rate(self.label_variant_2, self.estimates_variant1)
        self.assertEqual(fdr, 0.444)

    def test_metric_false_discovery_no_correct(self):
        fdr = false_discovery_rate(self.label_variant_3, self.estimates_variant1)
        self.assertEqual(fdr, 1.0)

    def test_metric_false_discovery_no_outlier_found(self):
        fdr = false_discovery_rate(self.label_variant_3, self.estimates_variant2)
        self.assertEqual(fdr, 0.0)

    def test_metric_statistical_power_all_correct(self):
        power = statistical_power(self.label_variant_1, self.estimates_variant1)
        self.assertEqual(power, 1.0)

    def test_metric_statistical_power_some_correct(self):
        power = statistical_power(self.label_variant_2, self.estimates_variant1)
        self.assertEqual(power, 0.556)

    def test_metric_statistical_power_no_correct(self):
        power = statistical_power(self.label_variant_3, self.estimates_variant1)
        self.assertEqual(power, 0.0)


if __name__ == "__main__":
    unittest.main()
