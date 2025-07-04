from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from unquad.utils.data.load import load_shuttle
from unquad.estimation.evt_conformal import EVTConformalDetector
from unquad.strategy.split import Split
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":

    # Example Setup
    x_train, x_test, y_test = load_shuttle(setup=True)

    # One-Class Classification
    model = IForest(behaviour="new")

    # Conformal Strategy
    strategy = Split(calib_size=5_000)

    # EVT-Enhanced Conformal Anomaly Detector
    ce = EVTConformalDetector(
        detector=model,
        strategy=strategy,
        evt_threshold_method="percentile",
        evt_threshold_value=0.95,
        evt_min_tail_size=10,
    )
    ce.fit(x_train)
    estimates = ce.predict(x_test)

    # Apply FDR control
    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
