# *unquad*

A Python library for uncertainty-quantified anomaly detection.

**unquad** is a wrapper applicable for most [*PyOD*](https://pyod.readthedocs.io/en/latest/) detectors (see [Supported Estimators](#supported-estimators)) for
**uncertainty-quantified anomaly detection** based on one-class classification and the principles of **conformal inference**.

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![start with why](https://img.shields.io/badge/start%20with-why%3F-brightgreen.svg?style=flat)](https://arxiv.org/abs/2107.07511)

## What is *Conformal Anomaly Detection*?

[*Conformal anomaly detection*](https://www.diva-portal.org/smash/get/diva2:690997/FULLTEXT02.pdf) (CAD) is based on the
model-agnostic and non-parametric framework of [*conformal prediction*](https://en.wikipedia.org/wiki/Conformal_prediction#:~:text=Conformal%20prediction%20(CP)%20is%20a,assuming%20exchangeability%20of%20the%20data.) (CP).
While CP aims to produce statistically valid prediction regions (*prediction intervals* or *prediction sets*) for any
given point predictor or classifier, CAD aims to control the [*false discovery rate*](https://en.wikipedia.org/wiki/False_discovery_rate)
(FDR) for any given anomaly detector, suitable for one-class classification, without compromising its
[*statistical power*](https://en.wikipedia.org/wiki/Power_of_a_test).

***CAD translates anomaly scores into statistical p-values by comparing anomaly scores observed on test data to a retained set of calibration
scores as previously on normal data during model training*** (see [*One-Class Classification*](https://en.wikipedia.org/wiki/One-class_classification#:~:text=In%20machine%20learning%2C%20one%2Dclass,of%20one%2Dclass%20classifiers%20where)).
The larger the discrepancy between *normal* scores and observed test scores, the lower the obtained (**statistically valid**) p-value.
The p-values, instead of the usual anomaly estimates, allow for FDR control by statistical procedures like *Benjamini-Hochberg*.

## Getting started

```sh
pip install unquad
```

### Usage: CV+

```python
from pyod.models.iforest import IForest

from unquad.estimator.conformal_estimator import ConformalEstimator
from unquad.estimator.config.split import SplitConfiguration
from unquad.datasets.loader import DataLoader
from unquad.enums.adjustment import Adjustment
from unquad.enums.dataset import Dataset
from unquad.enums.method import Method
from unquad.evaluation.metrics import false_discovery_rate, statistical_power

dl = DataLoader(dataset=Dataset.THYROID)
x_train, x_test, y_test = dl.get_example_setup()

ce = ConformalEstimator(
    detector=IForest(behaviour="new"),
    method=Method.CV_PLUS,
    split=SplitConfiguration(n_split=10),
    adjustment=Adjustment.BENJAMINI_HOCHBERG,
    alpha=0.2,  # nominal FDR level
    seed=1
)

ce.fit(x_train)  # model fit and calibration
estimates = ce.predict(x_test, raw=False)

print(false_discovery_rate(y=y_test, y_hat=estimates))
print(statistical_power(y=y_test, y_hat=estimates))
```

Output:
```python
0.174  # empirical FDR
0.826  # empirical Power
```

### Usage: Jackknife+-after-Bootstrap

```python
from pyod.models.iforest import IForest

from unquad.estimator.conformal_estimator import ConformalEstimator
from unquad.estimator.config.split import SplitConfiguration
from unquad.datasets.loader import DataLoader
from unquad.enums.adjustment import Adjustment
from unquad.enums.dataset import Dataset
from unquad.enums.method import Method
from unquad.evaluation.metrics import false_discovery_rate, statistical_power

dl = DataLoader(dataset=Dataset.THYROID)
x_train, x_test, y_test = dl.get_example_setup()

ce = ConformalEstimator(
    detector=IForest(behaviour="new"),
    method=Method.JACKKNIFE_PLUS_AFTER_BOOTSTRAP,
    split=SplitConfiguration(n_split=0.95, n_bootstraps=40),
    adjustment=Adjustment.BENJAMINI_HOCHBERG,
    alpha=0.1,  # nominal FDR level
    seed=1,
)

ce.fit(x_train)  # model fit and calibration
estimates = ce.predict(x_test, raw=False)

print(false_discovery_rate(y=y_test, y_hat=estimates))
print(statistical_power(y=y_test, y_hat=estimates))
```

Output:
```python
0.041 # empirical FDR
0.959 # empirical Power
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
