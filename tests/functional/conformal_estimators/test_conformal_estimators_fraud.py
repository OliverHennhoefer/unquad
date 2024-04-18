import unittest
import pandas as pd
from pyod.models.iforest import IForest

from unquad.enums.adjustment import Adjustment
from unquad.estimator.conformal import ConformalEstimator
from unquad.enums.method import Method
from unquad.evaluation.metrics import false_discovery_rate, statistical_power


class TestConformalEstimatorsIonosphere(unittest.TestCase):

    df = pd.read_csv("./test_data/fraud.zip", compression="zip")
    outliers = df.loc[df.Class == 1]
    inliers = df.loc[df.Class == 0]

    n_inlier = len(inliers)
    n_train = n_inlier // 2

    x_train = inliers.head(n_train)
    x_train = x_train.drop(["Class"], axis=1)

    x_test = pd.concat(
        [
            inliers.tail((n_inlier - n_train)).sample(frac=0.05, random_state=1),
            outliers,
        ],
        axis=0,
    )
    y_test = x_test["Class"]
    x_test = x_test.drop(["Class"], axis=1)

    def test_split_conformal(self):
        """
        Split-Conformal Estimator.
        """

        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.SPLIT_CONFORMAL,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.1,
            random_state=1,
            split=2_000,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.125)
        self.assertEqual(power, 0.875)

    def test_cv(self):
        """
        CV Estimator.
        """

        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.CV,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.1,
            random_state=1,
            split=10,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.131)
        self.assertEqual(power, 0.869)

    def test_jackknife_after_bootstrap(self):
        """
        Jackknife-after-Bootstrap Estimator.
        """

        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.JACKKNIFE_AFTER_BOOTSTRAP,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.1,
            random_state=1,
            split=30,
            bootstrap=0.975,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.129)
        self.assertEqual(power, 0.871)

    def test_jackknife_plus_after_bootstrap(self):
        """
        Jackknife+-after-Bootstrap Estimator.
        """

        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.10,
            random_state=1,
            split=30,
            bootstrap=0.975,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.154)
        self.assertEqual(power, 0.846)
