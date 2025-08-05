**nonconform** is a Python library that enhances anomaly detection by providing uncertainty quantification. It acts as a wrapper around most detectors from the popular [*PyOD*](https://pyod.readthedocs.io/en/latest/) library (see [Supported Estimators](#supported-estimators)). By leveraging one-class classification principles and **conformal inference**, **nonconform** enables **statistically rigorous anomaly detection**.

# Key Features

*   **Uncertainty Quantification:** Go beyond simple anomaly scores; get statistically valid _p_-values.
*   **Error Control:** Reliably control metrics like the False Discovery Rate (FDR).
*   **Broad PyOD Compatibility:** Works with a wide range of PyOD estimators (see [Supported Estimators](#supported-estimators)).
*   **Flexible Strategies:** Implements various conformal strategies like Split-Conformal and Bootstrap-after-Jackknife+ (JaB+).

# Getting Started

```sh
pip install nonconform
```

_For additional features, you might need optional dependencies:_
- `pip install nonconform[data]` - Includes pyarrow for loading example data (via remote download)
- `pip install nonconform[deep]` - Includes deep learning dependencies (PyTorch)
- `pip install nonconform[fdr]` - Includes advanced FDR control methods (online-fdr)
- `pip install nonconform[dev]` - Includes development tools (black, ruff, pre-commit)
- `pip install nonconform[docs]` - Includes documentation building tools (sphinx, furo, etc.)
- `pip install nonconform[all]` - Includes all optional dependencies

_Please refer to the [pyproject.toml](https://github.com/OliverHennhoefer/nonconform/blob/main/pyproject.toml) for details._

## Split-Conformal (also _Inductive_) Approach

Using a _Gaussian Mixture Model_ on the _Shuttle_ dataset:

> **Note:** The examples below use the built-in datasets. Install with `pip install nonconform[data]` to run these examples.

```python
from pyod.models.gmm import GMM
from scipy.stats import false_discovery_control

from nonconform.strategy import Split
from nonconform.estimation import StandardConformalDetector
from nonconform.utils.data import load_shuttle
from nonconform.utils.stat import false_discovery_rate, statistical_power

x_train, x_test, y_test = load_shuttle(setup=True)

ce = StandardConformalDetector(
    detector=GMM(),
    strategy=Split(n_calib=1_000)
)

ce.fit(x_train)
estimates = ce.predict(x_test)

decisions = false_discovery_control(estimates, method='bh') <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
```

Output:
```text
Empirical FDR: 0.108
Empirical Power: 0.99
```

# Advanced Usage

## Bootstrap-after-Jackknife+ (JaB+)

The `BootstrapConformal()` strategy allows to set 2 of the 3 parameters `resampling_ratio`, `n_boostraps` and `n_calib`.
For either combination, the remaining parameter will be filled automatically. This allows exact control of the
calibration procedure when using a bootstrap strategy.

```python
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from nonconform.estimation import StandardConformalDetector
from nonconform.strategy import Bootstrap
from nonconform.utils.data import load_shuttle
from nonconform.utils.stat import false_discovery_rate, statistical_power

x_train, x_test, y_test = load_shuttle(setup=True)

ce = StandardConformalDetector(
    detector=IForest(behaviour="new"),
    strategy=Bootstrap(resampling_ratio=0.99, n_bootstraps=20, plus=True)
)

ce.fit(x_train)
estimates = ce.predict(x_test)

decisions = false_discovery_control(estimates, method='bh') <= 0.1

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
```

Output:
```text
Empirical FDR: 0.067
Empirical Power: 0.98
```

## Weighted Conformal Anomaly Detection

The statistical validity of conformal anomaly detection depends on data *exchangability* (weaker than i.i.d.). This assumption can be slightly relaxed by computing weighted conformal _p_-values.

```python
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from nonconform.utils.data import load_shuttle
from nonconform.estimation import WeightedConformalDetector
from nonconform.strategy import Split
from nonconform.utils.stat import false_discovery_rate, statistical_power

x_train, x_test, y_test = load_shuttle(setup=True)

model = IForest(behaviour="new")
strategy = Split(n_calib=1_000)

ce = WeightedConformalDetector(detector=model, strategy=strategy)
ce.fit(x_train)
estimates = ce.predict(x_test)

decisions = false_discovery_control(estimates, method='bh') <= 0.1

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
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
**Bug reporting:** [https://github.com/OliverHennhoefer/nonconform/issues](https://github.com/OliverHennhoefer/nonconform/issues)
