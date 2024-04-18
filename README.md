# *unquad*

Tired of *alarm fatique*?

**unquad** is a wrapper applicable for most [*PyOD*](https://pyod.readthedocs.io/en/latest/) detectors for **uncertainty quantified anomaly detection**
based on one-class classification and the principles of **conformal inference**.

* ***unquad*** wraps almost any '[*PyOD*](https://pyod.readthedocs.io/en/latest/)' anomaly estimator (see [Supported Estimators](#supported-estimators)).
* ***unquad*** fits and calibrates given estimator to control the (marginal) **False Discovery Rate** (FDR).

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
[![HitCount](https://hits.dwyl.com/OliverHennhoefer/unquad.svg?style=flat-square&show=unique)](http://hits.dwyl.com/OliverHennhoefer/unquad)
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
The larger the discrepancy between *normal* scores and observed test scores, the lower the obtained (and statistically valid) p-value.
The p-values, instead of the usual anomaly estimates, allow for FDR-control by statistical procedures like *Benjamini-Hochberg*.

### Assumption
CAD assumes ***exchangability*** of training and future test data. *Exchangability* is closely related to the statistical
term of *independent and identically distributed random variables* (*IID*). IID implies both, independence <ins>and</ins> 
exchangability. Exchangability defines a joint probability distribution that remains the same under permutations
of the variables. With that, exchangability is a very practicable as it is a weaker assumption than IID.

### Limitations
Since CAD controls the FDR by adjustment procedures in context of **multiple testing**, trained conformal detectors currently
only work for *batch-wise* anomaly detection (on static data).\
Generally, CAD also offers a range of methods for the online setting when working with dynamic time-series data under potential
co-variate shift. Currently, this kind of online detector is not implemented. It is planned to add respective methods in future releases.

## Getting started

***(Not yet available. Will briefly be published.)***

```sh
pip install unquad
```

### Usage

```python
from pyod.models.iforest import IForest  # Isolation Forest (sklearn-based)
from pyod.utils import generate_data  # Example Data (PyOD built-in)

from unquad.estimator.conformal import ConformalEstimator  # Model Wrapper
from unquad.enums.adjustment import Adjustment  # Multiple Testing Adjustments
from unquad.enums.method import Method  # Conformal Methods
from unquad.evaluation.metrics import false_discovery_rate, statistical_power  # Evaluation Metrics

x_train, x_test, y_train, y_test = generate_data(
        n_train=1_000,
        n_test=1_000,
        n_features=10,
        contamination=0.1,
        random_state=1,
    )

x_train = x_train[y_train == 0]  # Normal Instances (One-Class Classification)

ce = ConformalEstimator(
        detector=IForest(behaviour="old"),
        method=Method.CV,
        adjustment=Adjustment.BENJAMINI_HOCHBERG,
        alpha=0.1,  # FDR
        random_state=2,
        split=10,
    )

ce.fit(x_train)  # Model Fit/Calibration
estimates = ce.predict(x_test, raw=False)

print(false_discovery_rate(y=y_test, y_hat=estimates))  # Empirical FDR
print(statistical_power(y=y_test, y_hat=estimates))  # Empirical Power
```

Output:
```
0.099 (False Discovery Rate; FDR)
0.901 (Statistical Power)
```

### Supported Estimators

The package currently supports anomaly estimators that are suitable for unsupervised one-class classification. As respective
detectors are therefore exclusively fitted on *normal* (or *non-anomalous*) data, parameters like *threshold* are set to the
smallest possible values.

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
