import numpy as np
from scipy.stats import false_discovery_control

from pyod.models.knn import KNN
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.bootstrap import Bootstrap
from unquad.utils.data.load import load_shuttle
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_shuttle(setup=True)

    ce = ConformalDetector(
        detector=KNN(
            algorithm="auto",
            metric="mahalanobis",
            metric_params={"V": np.cov(x_train, rowvar=False)},
        ),
        strategy=Bootstrap(resampling_ratio=0.95, n_bootstraps=10, plus=True),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)
    # Apply FDR control
    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
