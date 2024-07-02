from pyod.models.iforest import IForest

from unquad.datasets.loader import DataLoader
from unquad.enums.adjustment import Adjustment
from unquad.enums.aggregation import Aggregation
from unquad.enums.dataset import Dataset
from unquad.estimator.conformal_estimator import ConformalEstimator
from unquad.enums.method import Method
from unquad.estimator.split_configuration import SplitConfiguration
from unquad.evaluation.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    dl = DataLoader(dataset=Dataset.SHUTTLE)
    x_train, x_test, y_test = dl.get_example_setup()

    ce = ConformalEstimator(
        detector=IForest(behaviour="new"),
        method=Method.SPLIT_CONFORMAL,
        split=SplitConfiguration(n_split=550),
        adjustment=Adjustment.BENJAMINI_HOCHBERG,
        aggregation=Aggregation.MINIMUM,
        alpha=0.1,
        seed=1,
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test, raw=False)

    print(false_discovery_rate(y=y_test, y_hat=estimates))
    print(statistical_power(y=y_test, y_hat=estimates))
