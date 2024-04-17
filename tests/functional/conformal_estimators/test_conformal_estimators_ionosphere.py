import unittest
import pandas as pd
from pyod.models.iforest import IForest

from noncon.enums.adjustment import Adjustment
from noncon.estimator.conformal import ConformalEstimator
from noncon.enums.method import Method
from noncon.evaluation.metrics import false_discovery_rate, statistical_power


class TestConformalEstimatorsIonosphere(unittest.TestCase):

    df = pd.read_csv("./test_data/ionosphere.csv")
    outliers = df.loc[df.Class == 1]
    inliers = df.loc[df.Class == 0]

    n_inlier = len(inliers)
    n_train = n_inlier // 2

    x_train = inliers.head(n_train)
    x_train = x_train.drop(["Class"], axis=1)

    x_test = pd.concat([inliers.tail((n_inlier - n_train)), outliers], axis=0)
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
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.191)
        self.assertEqual(power, 0.809)

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
            split=20,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.042)
        self.assertEqual(power, 0.958)

    def test_cv_plus(self):
        """
        CV+ Estimator.
        """

        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.CV_PLUS,
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

        self.assertEqual(fdr, 0.0)
        self.assertEqual(power, 1.0)

    def test_jackknife(self):
        """
        Jackknife Estimator.
        """

        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.JACKKNIFE,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.2,
            random_state=1,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.128)
        self.assertEqual(power, 0.872)

    def test_jackknife_plus(self):
        """
        Jackknife+ Estimator.
        """

        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.JACKKNIFE_PLUS,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.2,
            random_state=1,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.165)
        self.assertEqual(power, 0.835)

    def test_jackknife_after_bootstrap(self):
        """
        Jackknife-after-Bootstrap Estimator.
        """

        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.JACKKNIFE_AFTER_BOOTSTRAP,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.2,
            random_state=1,
            split=50,
            bootstrap=0.75,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.053)
        self.assertEqual(power, 0.947)

    def test_jackknife_plus_after_bootstrap(self):
        """
        Jackknife+-after-Bootstrap Estimator.
        """

        ce = ConformalEstimator(
            detector=IForest(behaviour="new"),
            method=Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
            adjustment=Adjustment.BENJAMINI_HOCHBERG,
            alpha=0.2,
            random_state=1,
            split=50,
            bootstrap=0.75,
            silent=True,
        )

        ce.fit(self.x_train)
        estimates = ce.predict(self.x_test, raw=False)

        fdr = false_discovery_rate(y=self.y_test, y_hat=estimates)
        power = statistical_power(y=self.y_test, y_hat=estimates)

        self.assertEqual(fdr, 0.03)
        self.assertEqual(power, 0.97)
