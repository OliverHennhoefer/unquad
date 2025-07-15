# Extreme Conformal Detection

Extreme Value Theory (EVT) enhanced conformal prediction for superior detection of extreme anomalies.

## Overview

`EVTConformalDetector` extends standard conformal prediction by modeling extreme calibration scores using a Generalized Pareto Distribution (GPD). This hybrid approach provides better calibration for detecting extreme anomalies while maintaining empirical methods for normal values.

## Key Features

- **Hybrid p-value calculation**: Combines empirical and parametric approaches
- **Automatic threshold selection**: Multiple methods for identifying extreme values
- **Robust fallback**: Graceful degradation to standard conformal when EVT fitting fails
- **Configurable sensitivity**: Adjustable parameters for different data characteristics

## How the Hybrid P-Value Computation Works

The extreme conformal detector uses a sophisticated hybrid approach that combines two complementary methods:

### Standard Conformal for Observed Range
For test scores within the calibration range, the detector uses the standard empirical conformal p-value:
```
p_value = (1 + count(calibration_scores >= test_score)) / (1 + N_calibration)
```
This preserves exact conformal prediction guarantees for typical values.

### GPD-Based Extrapolation for Extreme Values
For test scores beyond the maximum calibration score (truly extreme values), the detector uses Generalized Pareto Distribution (GPD) modeling:

1. **Threshold Selection**: A threshold (e.g., 95th percentile) separates bulk from tail distribution
2. **GPD Fitting**: The tail exceedances are modeled using Maximum Likelihood Estimation
3. **Boundary Probability**: The minimum empirical p-value is `1/(N_calibration + 1)`
4. **Tail Extrapolation**: GPD provides principled estimates for the small probability space below this boundary

### The Key Insight
If calibration scores range from -0.9 to -0.1 with 1000 samples, then:
- Scores ≤ -0.1: Use empirical p-values (range: `1/1001` to `1.0`)
- Scores > -0.1: Use GPD to model the tiny tail probability below `1/1001`

The result is monotonic p-values that decrease appropriately with increasing anomaly scores.

### Theoretical Justification

The hybrid EVT-conformal approach maintains statistical validity under the standard EVT assumption that the GPD accurately models the extreme tail distribution. This approach is actually more conservative than alternatives:

**Without EVT**: Practitioners are limited to p-values ≥ 1/(n+1), requiring massive calibration sets for detecting extreme anomalies that survive conservative multiple testing corrections.

**With EVT**: The GPD provides principled tail probability estimates while respecting statistical uncertainty, offering a more conservative alternative to arbitrarily large calibration sets that would push extreme p-values to artificially small values.

**Key assumption**: The validity relies on the GPD accurately modeling the tail of the calibration score distribution - a standard assumption in extreme value theory applications.

## Basic Usage

```python
from nonconform.estimation.extreme_conformal import ExtremeConformalDetector
from nonconform.strategy.split import Split
from pyod.models.lof import LOF

# Initialize detector
detector = ExtremeConformalDetector(
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
| `extreme_threshold_method` | Method for selecting extreme threshold | `"percentile"` |
| `extreme_threshold_value` | Parameter for threshold method | `0.95` |
| `extreme_min_tail_size` | Minimum exceedances for GPD fitting | `10` |

## When to Use Extreme Conformal

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

Extreme conformal works with all conformal strategies:

```python
from nonconform.strategy.cross_val import CrossValidation
from nonconform.strategy.bootstrap import Bootstrap

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

Access fitted extreme parameters for analysis:

```python
detector.fit(X_train)

# Check if extreme fitting succeeded
if detector.gpd_params is not None:
    shape, loc, scale = detector.gpd_params
    print(f"GPD parameters - Shape: {shape:.3f}, Scale: {scale:.3f}")
    print(f"Extreme threshold: {detector.extreme_threshold:.3f}")
else:
    print("Extreme fitting failed, using standard conformal")
```