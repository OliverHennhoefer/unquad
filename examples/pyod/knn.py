from pyod.models.knn import KNN
from unquad.data.load import load_shuttle
from unquad.estimation.conformal import ConformalDetector
from unquad.estimation.properties.configuration import DetectorConfig
from unquad.strategy.split import Split
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_shuttle(setup=True)

    ce = ConformalDetector(
        detector=KNN(),
        strategy=Split(calib_size=2_000),
        config=DetectorConfig(alpha=0.1),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
