# Weighted Conformal Anomaly Detection

This example demonstrates how to use weighted conformal prediction for handling distribution shift in anomaly detection.

## Setup

```python
import numpy as np
from pyod.models import LOF
from unquad.estimation import WeightedConformalDetector
from unquad.strategy import Split
from unquad.estimation.properties.configuration import DetectorConfig
from unquad.data import load_breast

# Load example data
x, y = load_breast()
```

## Basic Usage

```python
# Initialize detector
detector = WeightedConformalDetector(
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

## Handling Distribution Shift

```python
# Simulate distribution shift
x_shifted = x + np.random.normal(0, 0.1, x.shape)

# Predict on shifted data
predictions_shifted = detector.predict(x_shifted)
```

## Next Steps

- Try [classical conformal detection](classical_conformal.md) for standard scenarios
- Learn about [FDR control](fdr_control.md) for multiple testing
- Explore [bootstrap-based detection](bootstrap_conformal.md) for uncertainty estimation 