![Logo](./docs/img/banner_dark.png#gh-dark-mode-only)
![Logo](./docs/img/banner_light.png#gh-light-mode-only)

[![PyPI Downloads](https://static.pepy.tech/badge/unquad)](https://pepy.tech/projects/unquad) [![PyPI Downloads](https://static.pepy.tech/badge/unquad/month)](https://pepy.tech/projects/unquad) [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unquad) [![start with why](https://img.shields.io/badge/start%20with-why%3F-brightgreen.svg?style=flat)](https://www.diva-portal.org/smash/get/diva2:690997/FULLTEXT02.pdf)

**unquad** is a Python library that enhances anomaly detection by providing uncertainty quantification. It acts as a wrapper around most detectors from the popular [*PyOD*](https://pyod.readthedocs.io/en/latest/) library (see [Supported Estimators](#supported-estimators)). By leveraging one-class classification principles and **conformal inference**, **unquad** enables **statistically rigorous anomaly detection**.

# Key Features

*   **Uncertainty Quantification:** Go beyond simple anomaly scores; get statistically valid _p_-values.
*   **Error Control:** Reliably control metrics like the False Discovery Rate (FDR).
*   **Broad PyOD Compatibility:** Works with a wide range of PyOD estimators (see [Supported Estimators](#supported-estimators)).
*   **Flexible Strategies:** Implements various conformal strategies like Split-Conformal and Bootstrap-after-Jackknife+ (JaB+).

# :hatching_chick: Getting Started

```sh
pip install unquad
```

_For advanced features (e.g. deep learning models) you might need optional dependencies. Please refer to the [pyproject.toml](https://github.com/OliverHennhoefer/unquad/blob/main/pyproject.toml) for details._

## Split-Conformal (also _Inductive_) Approach

Using a _Gaussian Mixture Model_ on the _Shuttle_ dataset with standard configuration (no `DetectorConfig()` set).

```python
from pyod.models.gmm import GMM

from unquad.strategy.split import Split
from unquad.estimation.conformal import ConformalDetector

from unquad.data.load import load_shuttle
from unquad.utils.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load_shuttle(setup=True)

ce = ConformalDetector(
    detector=GMM(),
    strategy=Split(calib_size=1_000)
)

ce.fit(x_train)
estimates = ce.predict(x_test)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
```

Output:
```text
Empirical FDR: 0.108
Empirical Power: 0.99
```

The behavior can be customized by customizing the `DetectorConfig()`:

```python
@dataclass
class DetectorConfig:
    alpha: float = 0.2                              # Nominal FDR value
    adjustment: Adjustment = Adjustment.BH          # Multiple testing procedure
    aggregation: Aggregation = Aggregation.MEDIAN   # Score aggregation (if applicable)
    seed: int = 1                                   # Reproducibility
    silent: bool = True                             # Verbosity
```

# :hatched_chick: Advanced Usage

## Bootstrap-after-Jackknife+ (JaB+)

The `BootstrapConformal()` strategy allows to set 2 of the 3 parameters `resampling_ratio`, `n_boostraps` and `n_calib`.
For either combination, the remaining parameter will be filled automatically. This allows exact control of the
calibration procedure when using a bootstrap strategy.

```python
from pyod.models.iforest import IForest

from unquad.estimation.properties.configuration import DetectorConfig
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.bootstrap import Bootstrap
from unquad.utils.enums import Aggregation, Adjustment

from unquad.data.load import load_shuttle
from unquad.utils.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load_shuttle(setup=True)

ce = ConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Bootstrap(resampling_ratio=0.99, n_bootstraps=20, plus=True),
    config=DetectorConfig(alpha=0.1, adjustment=Adjustment.BH, aggregation=Aggregation.MEAN),
)

ce.fit(x_train)
estimates = ce.predict(x_test)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
```

Output:
```text
Empirical FDR: 0.067
Empirical Power: 0.98
```

## Weighted Conformal Anomaly Detection

The statistical validity of conformal anomaly detection depends on data *exchangability* (weakher than i.i.d.). This assumption can be slightly relaxed with weighted conformal _p_-values.

```python
from pyod.models.iforest import IForest

from unquad.data.load import load_shuttle
from unquad.estimation.properties.configuration import DetectorConfig
from unquad.estimation.weighted_conformal import WeightedConformalDetector
from unquad.strategy.split import Split
from unquad.utils.enums import Aggregation
from unquad.utils.enums import Adjustment
from unquad.utils.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load_shuttle(setup=True)

model = IForest(behaviour="new")
strategy = Split(calib_size=1_000)
config = DetectorConfig(
    alpha=0.1, adjustment=Adjustment.BH, aggregation=Aggregation.MEAN
)

ce = WeightedConformalDetector(detector=model, strategy=strategy, config=config)
ce.fit(x_train)
estimates = ce.predict(x_test)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=estimates)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=estimates)}")
```

Output:
```text
Empirical FDR: 0.077
Empirical Power: 0.96
```

# Citation

If you find this repository useful for your research, please cite following papers:

##### Leave-One-Out-, Bootstrap- and Cross-Conformal Anomaly Detectors
```text
@inproceedings{Hennhofer2024,
	title        = {{ Leave-One-Out-, Bootstrap- and Cross-Conformal Anomaly Detectors }},
	author       = {Hennhofer, Oliver and Preisach, Christine},
	year         = 2024,
	month        = {Dec},
	booktitle    = {2024 IEEE International Conference on Knowledge Graph (ICKG)},
	publisher    = {IEEE Computer Society},
	address      = {Los Alamitos, CA, USA},
	pages        = {110--119},
	doi          = {10.1109/ICKG63256.2024.00022},
	url          = {https://doi.ieeecomputersociety.org/10.1109/ICKG63256.2024.00022}
}
```

##### Testing for outliers with conformal p-values
```text
@article{Bates2023,
	title        = {Testing for outliers with conformal p-values},
	author       = {Bates,  Stephen and Candès,  Emmanuel and Lei,  Lihua and Romano,  Yaniv and Sesia,  Matteo},
	year         = 2023,
	month        = feb,
	journal      = {The Annals of Statistics},
	publisher    = {Institute of Mathematical Statistics},
	volume       = 51,
	number       = 1,
	doi          = {10.1214/22-aos2244},
	issn         = {0090-5364},
	url          = {http://dx.doi.org/10.1214/22-AOS2244}
}
```
##### Model-free selective inference under covariate shift via weighted conformal p-values
```text
@inproceedings{Jin2023,
	title        = {Model-free selective inference under covariate shift via weighted conformal p-values},
	author       = {Ying Jin and Emmanuel J. Cand{\`e}s},
	year         = 2023,
	url          = {https://api.semanticscholar.org/CorpusID:259950903}
}
```

# Supported Estimators

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

# Contact
**Bug reporting:** [https://github.com/OliverHennhoefer/unquad/issues](https://github.com/OliverHennhoefer/unquad/issues)
