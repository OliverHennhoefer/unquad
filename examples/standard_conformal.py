from scipy.stats import false_discovery_control

from pyod.models.iforest import IForest
from unquad.estimation import StandardConformalDetector
from unquad.strategy import Split, Jackknife, Bootstrap
from unquad.utils.data import load_wbc
from unquad.utils.stat import false_discovery_rate, statistical_power

if __name__ == "__main__":

    # Example Setup
    x_train, x_test, y_test = load_wbc(setup=True)

    # One-Class Classification
    model = IForest(behaviour="new")

    # Conformal Strategy
    strategy = Bootstrap(n_calib=1_000, resampling_ratio=0.95)

    # Conformal Anomaly Detector
    ce = StandardConformalDetector(detector=model, strategy=strategy)
    ce.fit(x_train)
    estimates = ce.predict(x_test)

    # Apply FDR control
    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
