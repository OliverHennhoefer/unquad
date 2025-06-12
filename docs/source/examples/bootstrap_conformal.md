# Bootstrap-based Conformal Detection

This example demonstrates how to use bootstrap resampling for conformal anomaly detection.

## Setup

```python
import numpy as np
from pyod.models import LOF
from unquad.estimation import ConformalDetector
from unquad.strategy import Bootstrap
from unquad.estimation.properties.configuration import DetectorConfig
from unquad.data import load_breast

# Load example data
x, y = load_breast()
```

## Basic Usage

```python
# Initialize detector with bootstrap strategy
detector = ConformalDetector(
    detector=LOF(),
    strategy=Bootstrap(n_bootstraps=100, resampling_ratio=0.8),
    config=DetectorConfig(alpha=0.1)
)

# Fit and predict
detector.fit(x)
predictions = detector.predict(x)
```

## Plus Mode

```python
# Use plus mode to retain all bootstrap models
detector_plus = ConformalDetector(
    detector=LOF(),
    strategy=Bootstrap(
        n_bootstraps=100,
        resampling_ratio=0.8,
        plus=True
    ),
    config=DetectorConfig(alpha=0.1)
)

# Fit and predict with ensemble
detector_plus.fit(x)
predictions_plus = detector_plus.predict(x)
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [weighted conformal detection](weighted_conformal.md) for handling distribution shift
- Explore [cross-validation detection](cross_val_conformal.md) for robust calibration 