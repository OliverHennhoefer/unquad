from pyod.models.pca import PCA

from tests.datasets.loader import DataLoader
from unquad.estimator.configuration import EstimatorConfig
from unquad.estimator.estimator import ConformalDetector
from unquad.strategy.jackknife import JackknifeConformal
from unquad.utils.enums.dataset import Dataset
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    dl = DataLoader(dataset=Dataset.THYROID)
    x_train, x_test, y_test = dl.get_example_setup(random_state=1)

    ce = ConformalDetector(
        detector=PCA(n_components=5),
        strategy=JackknifeConformal(plus=True),
        config=EstimatorConfig(alpha=0.125),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
