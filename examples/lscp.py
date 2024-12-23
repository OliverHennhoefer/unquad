from pyod.models.lscp import LSCP
from pyod.models.pca import PCA

from unquad.utils.data.loader import DataLoader
from unquad.estimator.detector import ConformalDetector
from unquad.strategy.cross_val import CrossValidationConformal
from unquad.utils.enums.dataset import Dataset
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    dl = DataLoader(dataset=Dataset.FRAUD)
    x_train, x_test, y_test = dl.get_example_setup(random_state=1)

    detector_list = [
        PCA(n_components=1),
        PCA(n_components=3),
        PCA(n_components=5),
        PCA(n_components=10),
    ]

    ce = ConformalDetector(
        detector=LSCP(detector_list), strategy=CrossValidationConformal(k=20)
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
