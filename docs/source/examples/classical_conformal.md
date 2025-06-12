# Classical Conformal Anomaly Detection

This example demonstrates how to use classical conformal prediction for anomaly detection.

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
# Initialize detector
detector = ConformalDetector(
    detector=LOF(),
    strategy=Split(calib_size=0.2),
    config=DetectorConfig(alpha=0.1)
)

# Fit and predict
detector.fit(x)
predictions = detector.predict(x)

# Get p-values
p_values = detector.predict(x, output="p-value")

# Get raw scores
scores = detector.predict(x, output="score")
```

## Advanced Usage

```python
# Use cross-validation instead of split
from unquad.strategy import CrossValidation

detector = ConformalDetector(
    detector=LOF(),
    strategy=CrossValidation(k=5),
    config=DetectorConfig(alpha=0.1)
)

# Fit and predict with cross-validation
detector.fit(x)
predictions = detector.predict(x)
```

## Next Steps

- Try [weighted conformal detection](weighted_conformal.md) for handling distribution shift
- Learn about [FDR control](fdr_control.md) for multiple testing
- Explore [bootstrap-based detection](bootstrap_conformal.md) for uncertainty estimation 