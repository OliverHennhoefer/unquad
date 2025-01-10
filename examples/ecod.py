from pyod.models.ecod import ECOD

from unquad.utils.enums import Dataset
from unquad.data.loader import DataLoader
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.jackknife import Jackknife
from unquad.utils.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":
    dl = DataLoader(dataset=Dataset.BREAST)
    x_train, x_test, y_test = dl.get_example_setup(random_state=1)

    ce = ConformalDetector(detector=ECOD(), strategy=Jackknife(plus=True))

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
