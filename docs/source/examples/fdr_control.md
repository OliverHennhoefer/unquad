# FDR Control for Multiple Testing

This example demonstrates how to use False Discovery Rate (FDR) control in anomaly detection.

## Setup

```python
import numpy as np
from pyod.models import LOF
from unquad.estimation import ConformalDetector
from unquad.strategy import Split
from unquad.estimation.properties.configuration import DetectorConfig
from unquad.data import load_breast

# Load example data
x, y = load_breast()
```

## Basic Usage

```python
# Initialize detector with FDR control
detector = ConformalDetector(
    detector=LOF(),
    strategy=Split(calib_size=0.2),
    config=DetectorConfig(
        alpha=0.1,  # Target FDR level
        adjustment="bh"  # Benjamini-Hochberg procedure
    )
)

# Fit and predict
detector.fit(x)
predictions = detector.predict(x)
```

## Different FDR Control Methods

```python
# Benjamini-Yekutieli procedure
detector_by = ConformalDetector(
    detector=LOF(),
    strategy=Split(calib_size=0.2),
    config=DetectorConfig(
        alpha=0.1,
        adjustment="by"
    )
)

# No adjustment
detector_none = ConformalDetector(
    detector=LOF(),
    strategy=Split(calib_size=0.2),
    config=DetectorConfig(
        alpha=0.1,
        adjustment=None
    )
)
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [weighted conformal detection](weighted_conformal.md) for handling distribution shift
- Explore [bootstrap-based detection](bootstrap_conformal.md) for uncertainty estimation 