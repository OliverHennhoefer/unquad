from pyod.models.ocsvm import OCSVM
from unquad.data.load import load_ionosphere
from unquad.estimation.conformal import ConformalDetector
from unquad.estimation.properties.configuration import DetectorConfig
from unquad.strategy.jackknife import Jackknife
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_ionosphere(setup=True)

    ce = ConformalDetector(
        detector=OCSVM(),
        strategy=Jackknife(),
        config=DetectorConfig(alpha=0.125),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
