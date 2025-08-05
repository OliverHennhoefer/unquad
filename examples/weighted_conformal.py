from scipy.stats import false_discovery_control

from nonconform.estimation import WeightedConformalDetector
from nonconform.strategy import Split
from nonconform.utils.data import load_shuttle
from nonconform.utils.stat import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest

if __name__ == "__main__":
    # Example Setup
    x_train, x_test, y_test = load_shuttle(setup=True)

    # One-Class Classification
    model = IForest(behaviour="new")

    # Conformal Strategy
    strategy = Split(n_calib=1_000)

    # Weighted Conformal Anomaly Detector
    ce = WeightedConformalDetector(detector=model, strategy=strategy)
    ce.fit(x_train)
    estimates = ce.predict(x_test)

    # Apply FDR control
    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
