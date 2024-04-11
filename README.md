# *noncon*

Tired of *alarm fatique*? :rotating_light:

**noncon** is a wrapper for [*PyOD*](https://pyod.readthedocs.io/en/latest/) detectors (*scikit-learn*-based) for **uncertainty quantified anomaly detection**
based on one-class classification.

:envelope: Wraps an untrained anomaly detector\
:telescope: Fits and calibrates given anomaly detectors to provide marginal FDR-control

## What is *Conformal Anomaly Detection*?

Conformal anomaly detection (CAD) is based on the model-agnostic and non-parametric framework of conformal prediction (CP).
While CP aims to produce statistically valid prediction regions for any given point predictor, CAD aims to control the
**false discovery rate** (FDR) for any given anomaly detector, suitable for one-class classification, without compromising
it's **statistical power**.

### Assumption
CAD assumes ***exchangability*** of training and future test data. *Exchangability* is closely related to the statistical
term of *independent and identically distributed random variables* (*IID*). IID implies both, independence <ins>and</ins> 
exchangability. Exchangability defines a joint probability distribution that remains the same under permutations
of the variables.

### Limitations
Since CAD controls the FDR by adjustment procedures in context of multiple testing, trained conformal detectors currently
only work for batch-wise anomaly detection.\
CAD also offers methods for CAD on dynamic time-series data suitable also under co-variate shift in online settings.
Currently, this kind of online detector is not implemented but is planned to be added in future releases.

## Getting started

```sh
pip install noncon
```

## Usage

```python
from pyod.models.iforest import IForest  # Isolation Forest (sklearn-based)
from pyod.utils import generate_data  # Example Data (PyOD built-in)

from noncon.estimator.conformal import ConformalEstimator  # Model Wrapper
from noncon.enums.adjustment import Adjustment  # Multiple Testing Adjustment Procedures
from noncon.enums.method import Method  # (Cross-)Conformal Method
from noncon.evaluation.metrics import false_discovery_rate, statistical_power  # Evaluation Metrics

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
