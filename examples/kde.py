from pyod.models.kde import KDE

from unquad.data.loader import DataLoader
from unquad.estimator.configuration import EstimatorConfig
from unquad.estimator.estimator import ConformalDetector
from unquad.strategy.split import SplitConformal
from unquad.utils.enums.dataset import Dataset
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    dl = DataLoader(dataset=Dataset.MAMMOGRAPHY)
    x_train, x_test, y_test = dl.get_example_setup(random_state=1)

    ce = ConformalDetector(
        detector=KDE(),
        strategy=SplitConformal(calib_size=2_000),
        config=EstimatorConfig(alpha=0.1),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
