from pyod.models.mad import MAD
from scipy.stats import false_discovery_control
from unquad.utils.data.load import load_fraud
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.cross_val import CrossValidation
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_fraud(setup=True)

    ce = ConformalDetector(detector=MAD(), strategy=CrossValidation(k=10))

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
