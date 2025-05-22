# Welcome to unquad documentation!

**unquad** (*uncertainty-quantified anomaly detection*) is a Python library that provides uncertainty quantification in anomaly detection via conformal inference.

## What is unquad?

The `unquad` library enables conformal inference for one-class classification, providing statistically rigorous p-values for anomaly detection. Instead of relying on arbitrary thresholds, unquad converts anomaly scores to statistically valid p-values that can be used to control False Discovery Rates (FDR).

### Key Features

- **Conformal Anomaly Detection**: Convert raw anomaly scores into statistically valid p-values
- **Multiple Conformalization Strategies**: Support for Leave-One-Out, Cross-Conformal, and Bootstrap methods
- **Weighted Conformal p-values**: Handle covariate shift scenarios
- **FDR Control**: Built-in Benjamini-Hochberg correction for multiple testing
- **PyOD Integration**: Compatible with most PyOD anomaly detection models
- **Low-Data Regimes**: Optimized for scenarios with limited calibration data

### Statistical Guarantees

unquad provides statistical guarantees for:
- **Type I Error Control**: Control false positive rates at specified levels
- **False Discovery Rate Control**: Manage the proportion of false alarms among discoveries
- **Marginal Validity**: Ensure p-values are statistically valid under exchangeability assumptions

## Quick Start

```python
import unquad
from unquad.conformal import ClassicalCAD
from sklearn.ensemble import IsolationForest

# Initialize your anomaly detector
detector = IsolationForest(random_state=42)

# Create conformal anomaly detector
cad = ClassicalCAD(detector)

# Fit on normal data
cad.fit(X_normal)

# Get p-values for test data
p_values = cad.predict_proba(X_test)

# Control FDR at 5% level
from unquad.multiple_testing import benjamini_hochberg
discoveries = benjamini_hochberg(p_values, alpha=0.05)
```

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/conformal_inference
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/unquad/index
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`