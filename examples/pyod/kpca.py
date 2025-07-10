from scipy.stats import false_discovery_control

from pyod.models.kpca import KPCA
from unquad.estimation.standard_conformal import StandardConformalDetector
from unquad.strategy.cross_val import CrossValidation
from unquad.utils.data.load import load_ionosphere
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_ionosphere(setup=True)

    ce = StandardConformalDetector(detector=KPCA(), strategy=CrossValidation(k=25))

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
