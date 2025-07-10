from scipy.stats import false_discovery_control

from pyod.models.iforest import IForest
from unquad.estimation import WeightedConformalDetector
from unquad.strategy import Split
from unquad.utils.data import load_shuttle
from unquad.utils.stat import false_discovery_rate, statistical_power

if __name__ == "__main__":
    # Example Setup
    x_train, x_test, y_test = load_shuttle(setup=True)

    # One-Class Classification
    model = IForest(behaviour="new")

    # Conformal Strategy
    strategy = Split(calib_size=1_000)

    # Weighted Conformal Anomaly Detector
    ce = WeightedConformalDetector(detector=model, strategy=strategy)
    ce.fit(x_train)
    estimates = ce.predict(x_test)

    # Apply FDR control
    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
