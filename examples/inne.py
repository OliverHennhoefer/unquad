from pyod.models.inne import INNE

from tests.datasets.loader import DataLoader
from unquad.estimator.estimator import ConformalDetector
from unquad.strategy.bootstrap import BootstrapConformal
from unquad.utils.enums.dataset import Dataset
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    dl = DataLoader(dataset=Dataset.SHUTTLE)
    x_train, x_test, y_test = dl.get_example_setup(random_state=1)

    ce = ConformalDetector(
        detector=INNE(),
        strategy=BootstrapConformal(resampling_ratio=0.99, n_calib=2_000),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
