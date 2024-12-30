from pyod.models.iforest import IForest

from unquad.utils.enums import Dataset
from unquad.data.loader import DataLoader
from unquad.estimator.configuration import DetectorConfig
from unquad.estimator.detector import ConformalDetector
from unquad.strategy.bootstrap import BootstrapConformal
from unquad.utils.enums import Aggregation
from unquad.utils.enums import Adjustment
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    dl = DataLoader(dataset=Dataset.SHUTTLE)
    x_train, x_test, y_test = dl.get_example_setup(random_state=1)

    ce = ConformalDetector(
        detector=IForest(behaviour="new"),
        strategy=BootstrapConformal(resampling_ratio=0.99, n_bootstraps=20, plus=True),
        config=DetectorConfig(
            alpha=0.1, adjustment=Adjustment.BY, aggregation=Aggregation.MEAN
        ),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
