from pyod.models.lof import LOF
from scipy.stats import false_discovery_control
from unquad.utils.load import load_musk
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.jackknife import Jackknife
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_musk(setup=True)

    ce = ConformalDetector(detector=LOF(), strategy=Jackknife(plus=True))

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
