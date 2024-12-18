import unittest

from unquad.estimator.config.split import SplitConfiguration


class TestSplitConfiguration(unittest.TestCase):
    def test_split_configuration_calculate_calibration_set_size(self):
        split_config = SplitConfiguration(n_split=0.05, n_bootstraps=20)
        split_config.configure(n_train=1_000)
        self.assertEqual(split_config.n_calib, 1_000)

    def test_bootstrap_configuration_calculate_calibration_n_split(self):
        split_config = SplitConfiguration(n_bootstraps=20, n_calib=10_000)
        split_config.configure(n_train=1_000)
        self.assertEqual(split_config.n_split, 0.5)

    def test_bootstrap_configuration_calculate_calibration_n_bootstraps(self):
        split_config = SplitConfiguration(n_split=0.95, n_calib=10_000)
        split_config.configure(n_train=1_000)
        self.assertEqual(split_config.n_bootstraps, 200)

    def test_bootstrap_configuration_n_split_larger_n_train(self):
        with self.assertRaises(ValueError):
            split_config = SplitConfiguration(n_bootstraps=20, n_split=3_000)
            split_config.configure(n_train=1_000)

    def test_bootstrap_configuration_with_all(self):
        with self.assertRaises(ValueError):
            split_config = SplitConfiguration(
                n_bootstraps=20, n_split=0.05, n_calib=10_000
            )
            split_config.configure(n_train=1_000)


if __name__ == "__main__":
    unittest.main()
