from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control
from unquad.data.load import load_shuttle
from unquad.estimation.conformal import ConformalDetector
from unquad.estimation.configuration import DetectorConfig
from unquad.strategy.bootstrap import Bootstrap
from unquad.utils.enums import Adjustment, Aggregation
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_shuttle(setup=True)

    ce = ConformalDetector(
        detector=IForest(behaviour="new"),
        strategy=Bootstrap(resampling_ratio=0.99, n_bootstraps=20, plus=True),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
