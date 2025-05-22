from pyod.models.loda import LODA
from unquad.data.load import load_mammography
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.cross_val import CrossValidation
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_mammography(setup=True)

    ce = ConformalDetector(detector=LODA(), strategy=CrossValidation(k=10))

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
