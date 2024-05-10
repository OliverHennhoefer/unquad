import unittest

from unquad.estimator.split_config.bootstrap_config import BootstrapConfiguration


class TestEvaluationMetrics(unittest.TestCase):

    def test_bootstrap_configuration_with_n_b_m(self):
        bc_nbm = BootstrapConfiguration(n=1_000, b=20, m=0.5)
        self.assertEqual(bc_nbm._c, 10_000)

    def test_bootstrap_configuration_with_n_b_c(self):
        bc_nbc = BootstrapConfiguration(n=1_000, b=20, c=10_000)
        self.assertEqual(bc_nbc._m, 0.5)

    def test_bootstrap_configuration_with_n_m_c(self):
        bc_nmc = BootstrapConfiguration(n=1_000, m=0.5, c=10_000)
        self.assertEqual(bc_nmc.b, 20)

    def test_bootstrap_configuration_with_all(self):
        with self.assertRaises(ValueError):
            BootstrapConfiguration(n=1_000, b=20, m=0.5, c=10_000)


if __name__ == "__main__":
    unittest.main()
