from pyod.models.auto_encoder import AutoEncoder
from scipy.stats import false_discovery_control
from unquad.utils.data.load import load_fraud
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.split import Split
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_fraud(setup=True)

    ce = ConformalDetector(
        detector=AutoEncoder(epoch_num=10, batch_size=256),
        strategy=Split(calib_size=2_000),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    decisions = false_discovery_control(estimates, method="bh") <= 0.125

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
