# *unquad*: Uncertainty-Quantified Anomaly Detection

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unquad)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**unquad** is a wrapper applicable for most [*PyOD*](https://pyod.readthedocs.io/en/latest/) detectors (see [Supported Estimators](#supported-estimators)) for
**uncertainty-quantified anomaly detection** based on one-class classification and the principles of **conformal inference**.

```sh
pip install unquad
```

## What is *Conformal Anomaly Detection*?

[*Conformal Anomaly Detection*](https://www.diva-portal.org/smash/get/diva2:690997/FULLTEXT02.pdf) (CAD) is based on the
model-agnostic and non-parametric framework of [*conformal prediction*](https://en.wikipedia.org/wiki/Conformal_prediction#:~:text=Conformal%20prediction%20(CP)%20is%20a,assuming%20exchangeability%20of%20the%20data.) (CP).
While CP aims to produce statistically valid prediction regions (*prediction intervals* or *prediction sets*) for any
given point predictor or classifier, CAD aims to control statistical metrics, like the [*false discovery rate*](https://en.wikipedia.org/wiki/False_discovery_rate),
for a given anomaly detector suitable for one-class classification – without overly compromising on its
[*statistical power*](https://en.wikipedia.org/wiki/Power_of_a_test).

In essence, CAD translates anomaly scores into statistical p-values by comparing anomaly scores observed on test data to a retained set of calibration
scores as previously obtained for normal data during the model training stage.
The larger the discrepancy between *normal* scores and observed test scores, the lower the obtained (and **statistically valid**) p-value.
The p-values, instead of the usual anomaly estimates, allow, e.g., for FDR control by statistical procedures like *Benjamini-Hochberg*.


### Usage: Split-Conformal (Inductive Approach)

```python
from pyod.models.gmm import GMM

from unquad.utils.data.loader import DataLoader
from unquad.utils.enums.dataset import Dataset
from unquad.estimator.configuration import DetectorConfig
from unquad.estimator.detector import ConformalDetector
from unquad.strategy.split import SplitConformal
from unquad.utils.metrics import false_discovery_rate, statistical_power

dl = DataLoader(dataset=Dataset.SHUTTLE)
x_train, x_test, y_test = dl.get_example_setup(random_state=1)

ce = ConformalDetector(
    detector=GMM(),
    strategy=SplitConformal(calib_size=1_000),
    config=DetectorConfig(alpha=0.05),
)

ce.fit(x_train)
estimates = ce.predict(x_test)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
```

Output:
```python
Empirical FDR: 0.03
Empirical Power: 0.97
```

### Usage: Bootstrap-after-Jackknife+ (JaB+)

```python
from pyod.models.iforest import IForest

from unquad.utils.data.loader import DataLoader
from unquad.utils.enums.dataset import Dataset
from unquad.estimator.configuration import DetectorConfig
from unquad.estimator.detector import ConformalDetector
from unquad.strategy.bootstrap import BootstrapConformal
from unquad.utils.enums.aggregation import Aggregation
from unquad.utils.enums.adjustment import Adjustment
from unquad.utils.metrics import false_discovery_rate, statistical_power

dl = DataLoader(dataset=Dataset.SHUTTLE)
x_train, x_test, y_test = dl.get_example_setup(random_state=1)

ce = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=BootstrapConformal(resampling_ratio=0.99, n_bootstraps=20, plus=True),
    config=DetectorConfig(alpha=0.1,
                          adjustment=Adjustment.BENJAMINI_HOCHBERG,
                          aggregation=Aggregation.MEAN),
)

ce.fit(x_train)
estimates = ce.predict(x_test)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
```

Output:
```python
Empirical FDR: 0.067
Empirical Power: 0.933
```

### Supported Estimators

The package currently supports anomaly estimators that are suitable for unsupervised one-class classification. As respective
detectors are therefore exclusively fitted on *normal* (or *non-anomalous*) data, parameters like *threshold* are therefore internally
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
