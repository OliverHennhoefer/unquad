from pyod.models.loci import LOCI
from scipy.stats import false_discovery_control
from unquad.utils.load import load_thyroid
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.split import Split
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_thyroid(setup=True)

    ce = ConformalDetector(detector=LOCI(k=1), strategy=Split(calib_size=1_000))

    ce.fit(x_train)
    estimates = ce.predict(x_test)
    # Apply FDR control
    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
