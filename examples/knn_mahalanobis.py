import numpy as np

from pyod.models.knn import KNN
from pyod.utils import generate_data

from unquad.enums.adjustment import Adjustment
from unquad.estimator.conformal_estimator import ConformalEstimator
from unquad.enums.method import Method
from unquad.estimator.split_configuration import SplitConfiguration
from unquad.evaluation.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generate_data(
        n_train=1_000,
        n_test=1_000,
        n_features=10,
        contamination=0.1,
        random_state=1,
    )

    x_train = x_train[y_train == 0]
    X_train_cov = np.cov(x_train, rowvar=False)

    sc = SplitConfiguration(n_split=0.99, n_bootstraps=30)
    ce = ConformalEstimator(
        detector=KNN(
            algorithm="auto", metric="mahalanobis", metric_params={"V": X_train_cov}
        ),
        method=Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
        split=sc,
        adjustment=Adjustment.BENJAMINI_HOCHBERG,
        alpha=0.1,
        seed=1,
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test, raw=False)

    print(false_discovery_rate(y=y_test, y_hat=estimates))
    print(statistical_power(y=y_test, y_hat=estimates))
