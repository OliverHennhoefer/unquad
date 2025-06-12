# False Discovery Rate Control

This guide explains how to use False Discovery Rate (FDR) control in `unquad` for multiple testing scenarios.

## Overview

FDR control is a statistical method for handling multiple hypothesis testing. In anomaly detection, it helps control the proportion of false positives among all detected anomalies.

## Usage

```python
from unquad.estimation import ConformalDetector
from unquad.strategy import Split
from unquad.estimation.properties.configuration import DetectorConfig
from pyod.models import LOF

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
detector.fit(x_train)
predictions = detector.predict(x_test)
```

## Available Methods

- **BH**: Benjamini-Hochberg procedure (default)
- **BY**: Benjamini-Yekutieli procedure
- **None**: No adjustment

## When to Use

- When testing multiple hypotheses simultaneously
- When you need to control the proportion of false positives
- In high-dimensional anomaly detection

## Next Steps

- Learn about [weighted conformal p-values](weighted_conformal.md)
- Explore [different conformalization strategies](conformalization_strategies.md) 