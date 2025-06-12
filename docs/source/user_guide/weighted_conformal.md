# Weighted Conformal P-values

This guide explains how to use weighted conformal p-values in `unquad` for handling distribution shift.

## Overview

Weighted conformal p-values extend classical conformal prediction to handle cases where the test data distribution differs from the calibration data distribution. This is particularly useful in scenarios with covariate shift.

## Usage

```python
from unquad.estimation import WeightedConformalDetector
from unquad.strategy import Split
from pyod.models import LOF

# Initialize detector
detector = WeightedConformalDetector(
    detector=LOF(),
    strategy=Split(calib_size=0.2),
    config=DetectorConfig(alpha=0.1)
)

# Fit and predict
detector.fit(x_train)
predictions = detector.predict(x_test)
```

## How It Works

1. The detector computes importance weights for calibration and test instances
2. These weights are used to adjust the p-value calculation
3. The resulting p-values account for the distribution shift

## When to Use

- When test data distribution differs from training data
- In scenarios with known covariate shift
- When you need to account for sample importance

## Next Steps

- Learn about [FDR control](fdr_control.md) for multiple testing
- Explore [different conformalization strategies](conformalization_strategies.md) 