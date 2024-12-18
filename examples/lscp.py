from pyod.models.lscp import LSCP
from pyod.models.pca import PCA

from unquad.datasets.loader import DataLoader
from unquad.enums.adjustment import Adjustment
from unquad.enums.dataset import Dataset
from unquad.estimator.conformal_estimator import ConformalEstimator
from unquad.enums.method import Method
from unquad.estimator.config.split import SplitConfiguration
from unquad.evaluation.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    dl = DataLoader(dataset=Dataset.FRAUD)
    x_train, x_test, y_test = dl.get_example_setup()

    detector_list = [
        PCA(n_components=1),
        PCA(n_components=3),
        PCA(n_components=5),
        PCA(n_components=10),
    ]

    ce = ConformalEstimator(
        detector=LSCP(detector_list),
        method=Method.CV,
        split=SplitConfiguration(n_split=5),
        adjustment=Adjustment.BENJAMINI_HOCHBERG,
        alpha=0.1,
        seed=1,
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test, raw=False)

    print(false_discovery_rate(y=y_test, y_hat=estimates))
    print(statistical_power(y=y_test, y_hat=estimates))
