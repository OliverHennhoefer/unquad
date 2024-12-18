from pyod.models.abod import ABOD

from unquad.datasets.loader import DataLoader
from unquad.enums.adjustment import Adjustment
from unquad.enums.dataset import Dataset
from unquad.estimator.conformal_estimator import ConformalEstimator
from unquad.enums.method import Method
from unquad.estimator.config.split import SplitConfiguration
from unquad.evaluation.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    dl = DataLoader(dataset=Dataset.MUSK)
    x_train, x_test, y_test = dl.get_example_setup()

    ce = ConformalEstimator(
        detector=ABOD(),
        method=Method.CV_PLUS,
        split=SplitConfiguration(n_split=10),
        adjustment=Adjustment.BENJAMINI_HOCHBERG,
        alpha=0.2,
        seed=1,
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test, raw=False)

    print(false_discovery_rate(y=y_test, y_hat=estimates))
    print(statistical_power(y=y_test, y_hat=estimates))
