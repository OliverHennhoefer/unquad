from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from unquad.data.load import load_shuttle
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.split import Split
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":

    # Example Setup
    x_train, x_test, y_test = load_shuttle(setup=True)

    # One-Class Classification
    model = IForest(behaviour="new")

    # Conformal Strategy
    strategy = Split(calib_size=1_000)

    # Conformal Anomaly Detector
    ce = ConformalDetector(detector=model, strategy=strategy)
    ce.fit(x_train)
    estimates = ce.predict(x_test)

    # Apply FDR control
    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
