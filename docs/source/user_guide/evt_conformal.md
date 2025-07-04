# EVT Conformal Detection

Extreme Value Theory (EVT) enhanced conformal prediction for superior detection of extreme anomalies.

## Overview

`EVTConformalDetector` extends standard conformal prediction by modeling extreme calibration scores using a Generalized Pareto Distribution (GPD). This hybrid approach provides better calibration for detecting extreme anomalies while maintaining empirical methods for normal values.

## Key Features

- **Hybrid p-value calculation**: Combines empirical and parametric approaches
- **Automatic threshold selection**: Multiple methods for identifying extreme values
- **Robust fallback**: Graceful degradation to standard conformal when EVT fitting fails
- **Configurable sensitivity**: Adjustable parameters for different data characteristics

## Basic Usage

```python
from unquad.estimation.evt_conformal import EVTConformalDetector
from unquad.strategy.split import Split
from pyod.models.lof import LOF

# Initialize detector
detector = EVTConformalDetector(
    detector=LOF(),
    strategy=Split(calibration_size=0.3),
    evt_threshold_method="percentile",
    evt_threshold_value=0.95,
    evt_min_tail_size=10
)

# Fit and predict
detector.fit(X_train)
p_values = detector.predict(X_test)
```

## Threshold Selection Methods

### Percentile Method
```python
detector = EVTConformalDetector(
    detector=LOF(),
    strategy=Split(),
    evt_threshold_method="percentile",
    evt_threshold_value=0.95  # Use 95th percentile
)
```

### Top-K Method
```python
detector = EVTConformalDetector(
    detector=LOF(),
    strategy=Split(),
    evt_threshold_method="top_k",
    evt_threshold_value=50  # Use top 50 scores
)
```

### Mean Excess Method
```python
detector = EVTConformalDetector(
    detector=LOF(),
    strategy=Split(),
    evt_threshold_method="mean_excess",
    evt_threshold_value=0.1  # Threshold tolerance
)
```

### Custom Method
```python
def custom_threshold(scores):
    return np.quantile(scores, 0.98)

detector = EVTConformalDetector(
    detector=LOF(),
    strategy=Split(),
    evt_threshold_method="custom",
    evt_threshold_value=custom_threshold
)
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `evt_threshold_method` | Method for selecting EVT threshold | `"percentile"` |
| `evt_threshold_value` | Parameter for threshold method | `0.95` |
| `evt_min_tail_size` | Minimum exceedances for GPD fitting | `10` |

## When to Use EVT Conformal

**Recommended for:**
- Datasets with extreme outliers
- High-dimensional data with tail dependence
- Applications requiring precise extreme value detection
- Scenarios where standard conformal underperforms on outliers

**Not recommended for:**
- Small datasets (< 100 samples)
- Uniform score distributions
- Real-time applications with strict latency requirements

## Performance Considerations

- **Computational overhead**: Additional GPD fitting during training
- **Memory usage**: Stores calibration scores and EVT parameters
- **Robustness**: Automatically falls back to standard conformal if EVT fitting fails

## Integration with Strategies

EVT conformal works with all conformal strategies:

```python
from unquad.strategy.cross_val import CrossValidation
from unquad.strategy.bootstrap import Bootstrap

# Cross-validation strategy
detector = EVTConformalDetector(
    detector=LOF(),
    strategy=CrossValidation(n_folds=5),
    evt_threshold_method="percentile",
    evt_threshold_value=0.9
)

# Bootstrap strategy
detector = EVTConformalDetector(
    detector=LOF(),
    strategy=Bootstrap(n_bootstraps=100),
    evt_threshold_method="top_k",
    evt_threshold_value=25
)
```

## Diagnostic Information

Access fitted EVT parameters for analysis:

```python
detector.fit(X_train)

# Check if EVT fitting succeeded
if detector.gpd_params is not None:
    shape, loc, scale = detector.gpd_params
    print(f"GPD parameters - Shape: {shape:.3f}, Scale: {scale:.3f}")
    print(f"EVT threshold: {detector.evt_threshold:.3f}")
else:
    print("EVT fitting failed, using standard conformal")
```