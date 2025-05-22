from pyod.models.mad import MAD
from unquad.data.load import load_fraud
from unquad.estimation.conformal import ConformalDetector
from unquad.estimation.properties.configuration import DetectorConfig
from unquad.strategy.cross_val import CrossValidation
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    x_train, x_test, y_test = load_fraud(setup=True)

    ce = ConformalDetector(
        detector=MAD(),
        strategy=CrossValidation(k=10),
        config=DetectorConfig(alpha=0.125),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
