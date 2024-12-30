# *unquad*: Uncertainty-Quantified Anomaly Detection

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unquad)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**unquad** is a wrapper applicable for most [*PyOD*](https://pyod.readthedocs.io/en/latest/) detectors (see [Supported Estimators](#supported-estimators)) enabling
**uncertainty-quantified anomaly detection** based on one-class classification and the principles of **conformal inference**.

```sh
pip install unquad
```

Mind the **optional dependencies** for using deep learning models or the built-in datasets (see. [pyproject.toml](https://github.com/OliverHennhoefer/unquad/blob/main/pyproject.toml)).

## What is *Conformal Anomaly Detection*?

[![start with why](https://img.shields.io/badge/start%20with-why%3F-brightgreen.svg?style=flat)](https://www.diva-portal.org/smash/get/diva2:690997/FULLTEXT02.pdf)

[*Conformal Anomaly Detection*](https://www.diva-portal.org/smash/get/diva2:690997/FULLTEXT02.pdf) applies the principles of conformal inference ([*conformal prediction*](https://en.wikipedia.org/wiki/Conformal_prediction#:~:text=Conformal%20prediction%20(CP)%20is%20a,assuming%20exchangeability%20of%20the%20data.)) to anomaly detection.
*Conformal Anomaly Detection* focuses on controlling error metrics like the [*false discovery rate*](https://en.wikipedia.org/wiki/False_discovery_rate), while maintaining [*statistical power*](https://en.wikipedia.org/wiki/Power_of_a_test).

CAD converts anomaly scores to _p_-values by comparing test data scores against calibration scores from normal training data.
The resulting _p_-value of the test score(s) is computed as the normalized rank among the calibration scores.
These **statistically valid** _p_-values enable error control through methods like *Benjamini-Hochberg*, replacing traditional anomaly estimates that lack any kind of statistical guarantee.

### Usage: Split-Conformal (Inductive Approach)

Using the default behavior of `ConformalDetector()` with default `DetectorConfig()`.

```python
from pyod.models.gmm import GMM

from unquad.utils.enums import Dataset
from unquad.data.loader import DataLoader
from unquad.estimator.detector import ConformalDetector
from unquad.strategy.split import SplitConformal
from unquad.utils.metrics import false_discovery_rate, statistical_power

dl = DataLoader(dataset=Dataset.SHUTTLE)
x_train, x_test, y_test = dl.get_example_setup(random_state=1)

ce = ConformalDetector(
    detector=GMM(),
    strategy=SplitConformal(calib_size=1_000)
)

ce.fit(x_train)
estimates = ce.predict(x_test)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
```

Output:
```text
Empirical FDR: 0.108
Empirical Power: 0.892
```

The behavior can be customized by changing the `DetectorConfig()`:

```python
@dataclass
class DetectorConfig:
    alpha: float = 0.2  # Nominal FDR value
    adjustment: Adjustment = Adjustment.BH  # Multiple Testing Procedure
    aggregation: Aggregation = Aggregation.MEDIAN  # Score Aggregation (if necessary)
    seed: int = 1
    silent: bool = True
```

### Usage: Bootstrap-after-Jackknife+ (JaB+)

Using `ConformalDetector()` with customized `DetectorConfig()`.
The `BootstrapConformal()` strategy allows to set 2 of the 3 parameters `resampling_ratio`, `n_boostraps` and `n_calib`.
For either combination, the remaining parameter will be filled automatically. This allows exact control of the
calibration procedure when using a bootstrap strategy.

```python
from pyod.models.iforest import IForest

from unquad.data.loader import DataLoader
from unquad.estimator.configuration import DetectorConfig
from unquad.estimator.detector import ConformalDetector
from unquad.strategy.bootstrap import BootstrapConformal
from unquad.utils.enums import Aggregation, Adjustment, Dataset
from unquad.utils.metrics import false_discovery_rate, statistical_power

dl = DataLoader(dataset=Dataset.SHUTTLE)
x_train, x_test, y_test = dl.get_example_setup(random_state=1)

ce = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=BootstrapConformal(resampling_ratio=0.99, n_bootstraps=20, plus=True),
    config=DetectorConfig(alpha=0.1, adjustment=Adjustment.BY, aggregation=Aggregation.MEAN),
)

ce.fit(x_train)
estimates = ce.predict(x_test)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
```

Output:
```text
Empirical FDR: 0.0
Empirical Power: 1.0
```

### Supported Estimators

The package only supports anomaly estimators that are suitable for unsupervised one-class classification. As respective
detectors are therefore exclusively fitted on *normal* (or *non-anomalous*) data, parameters like *threshold* are internally
set to the smallest possible values.

Models that are **currently supported** include:

* Angle-Based Outlier Detection (**ABOD**)
* Autoencoder (**AE**)
* Cook's Distance (**CD**)
* Copula-based Outlier Detector (**COPOD**)
* Deep Isolation Forest (**DIF**)
* Empirical-Cumulative-distribution-based Outlier Detection (**ECOD**)
* Gaussian Mixture Model (**GMM**)
* Histogram-based Outlier Detection (**HBOS**)
* Isolation-based Anomaly Detection using Nearest-Neighbor Ensembles (**INNE**)
* Isolation Forest (**IForest**)
* Kernel Density Estimation (**KDE**)
* *k*-Nearest Neighbor (***k*NN**)
* Kernel Principal Component Analysis (**KPCA**)
* Linear Model Deviation-base Outlier Detection (**LMDD**)
* Local Outlier Factor (**LOF**)
* Local Correlation Integral (**LOCI**)
* Lightweight Online Detector of Anomalies (**LODA**)
* Locally Selective Combination of Parallel Outlier Ensembles (**LSCP**)
* GNN-based Anomaly Detection Method (**LUNAR**)
* Median Absolute Deviation (**MAD**)
* Minimum Covariance Determinant (**MCD**)
* One-Class SVM (**OCSVM**)
* Principal Component Analysis (**PCA**)
* Quasi-Monte Carlo Discrepancy Outlier Detection (**QMCD**)
* Rotation-based Outlier Detection (**ROD**)
* Subspace Outlier Detection (**SOD**)
* Scalable Unsupervised Outlier Detection (**SUOD**)

## Contact
**Bug reporting:** [https://github.com/OliverHennhoefer/unquad/issues](https://github.com/OliverHennhoefer/unquad/issues)
