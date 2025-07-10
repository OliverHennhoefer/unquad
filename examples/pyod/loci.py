from pyod.models.loci import LOCI
from scipy.stats import false_discovery_control

from unquad.estimation import StandardConformalDetector
from unquad.strategy import Split
from unquad.utils.data import load_thyroid
from unquad.utils.stat import false_discovery_rate, statistical_power

x_train, x_test, y_test = load_thyroid(setup=True)

ce = StandardConformalDetector(detector=LOCI(k=1), strategy=Split(calib_size=1_000))

ce.fit(x_train)
estimates = ce.predict(x_test)
# Apply FDR control
decisions = false_discovery_control(estimates, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
