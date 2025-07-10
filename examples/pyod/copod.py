from scipy.stats import false_discovery_control

from pyod.models.copod import COPOD
from unquad.estimation.standard_conformal import StandardConformalDetector
from unquad.strategy.jackknife import Jackknife
from unquad.utils.data.load import load_breast
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_breast(setup=True)

    ce = StandardConformalDetector(
        detector=COPOD(),
        strategy=Jackknife(),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
