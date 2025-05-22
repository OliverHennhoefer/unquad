from pyod.models.iforest import IForest
from unquad.data.load import load_shuttle
from unquad.estimation.conformal import ConformalDetector
from unquad.estimation.properties.configuration import DetectorConfig
from unquad.strategy.split import Split
from unquad.utils.enums import Adjustment, Aggregation
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    # Example Setup
    x_train, x_test, y_test = load_shuttle(setup=True)

    # One-Class Classification
    model = IForest(behaviour="new")

    # Conformal Strategy
    strategy = Split(calib_size=1_000)

    # Settings
    config = DetectorConfig(
        alpha=0.1, adjustment=Adjustment.BH, aggregation=Aggregation.MEAN
    )

    # Conformal Anomaly Detector
    ce = ConformalDetector(detector=model, strategy=strategy, config=config)
    ce.fit(x_train)
    estimates = ce.predict(x_test)

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
