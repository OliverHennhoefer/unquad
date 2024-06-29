from pyod.models.auto_encoder_torch import AutoEncoder

from unquad.datasets.loader import DataLoader
from unquad.enums.adjustment import Adjustment
from unquad.enums.dataset import Dataset
from unquad.estimator.conformal_estimator import ConformalEstimator
from unquad.enums.method import Method
from unquad.estimator.split_configuration import SplitConfiguration
from unquad.evaluation.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    dl = DataLoader(dataset=Dataset.FRAUD)
    x_train, x_test, y_test = dl.get_experiment_setup()

    ce = ConformalEstimator(
        detector=AutoEncoder(epochs=10, batch_size=256),
        method=Method.SPLIT_CONFORMAL,
        split=SplitConfiguration(n_split=3_000),
        adjustment=Adjustment.BENJAMINI_HOCHBERG,
        alpha=0.1,
        seed=1,
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test, raw=False)

    print(false_discovery_rate(y=y_test, y_hat=estimates))
    print(statistical_power(y=y_test, y_hat=estimates))
