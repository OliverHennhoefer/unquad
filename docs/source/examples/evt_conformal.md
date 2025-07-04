# EVT Conformal Detection Example

Complete example demonstrating Extreme Value Theory enhanced conformal anomaly detection for superior extreme value detection.

## Dataset and Setup

```python
import numpy as np
import pandas as pd
from pyod.models.lof import LOF
from pyod.models.isolation_forest import IForest
from pyod.models.ocsvm import OCSVM
from unquad.estimation.evt_conformal import EVTConformalDetector
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.split import Split
from unquad.utils.data.load import load_shuttle
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power
from scipy.stats import false_discovery_control

# Load dataset
x_train, x_test, y_test = load_shuttle(setup=True)
print(f"Training set: {x_train.shape}")
print(f"Test set: {x_test.shape}")
print(f"Anomaly rate: {y_test.mean():.3f}")
```

## Basic EVT Conformal Usage

```python
# Create EVT conformal detector
evt_detector = EVTConformalDetector(
    detector=LOF(n_neighbors=20),
    strategy=Split(calib_size=0.3),
    evt_threshold_method="percentile",
    evt_threshold_value=0.95,
    evt_min_tail_size=15,
    seed=42
)

# Fit detector
evt_detector.fit(x_train)

# Check if EVT fitting succeeded
if evt_detector.gpd_params is not None:
    shape, loc, scale = evt_detector.gpd_params
    print(f"EVT fitting successful:")
    print(f"  GPD Shape: {shape:.3f}")
    print(f"  GPD Scale: {scale:.3f}")
    print(f"  EVT Threshold: {evt_detector.evt_threshold:.3f}")
else:
    print("EVT fitting failed, using standard conformal")

# Generate predictions
p_values = evt_detector.predict(x_test)
print(f"P-values range: [{p_values.min():.4f}, {p_values.max():.4f}]")
```

## Comparison with Standard Conformal

```python
# Standard conformal detector for comparison
standard_detector = ConformalDetector(
    detector=LOF(n_neighbors=20),
    strategy=Split(calib_size=0.3),
    seed=42
)

# Fit and predict
standard_detector.fit(x_train)
p_values_standard = standard_detector.predict(x_test)

# Compare p-value distributions
print("P-value Statistics:")
print(f"EVT Conformal - Mean: {p_values.mean():.3f}, Std: {p_values.std():.3f}")
print(f"Standard Conformal - Mean: {p_values_standard.mean():.3f}, Std: {p_values_standard.std():.3f}")

# Focus on extreme values (lowest p-values)
extreme_mask = p_values < 0.01
print(f"\nExtreme values (p < 0.01):")
print(f"EVT Conformal: {extreme_mask.sum()} instances")
print(f"Standard Conformal: {(p_values_standard < 0.01).sum()} instances")
```

## Performance Evaluation

```python
# Test different significance levels
significance_levels = [0.01, 0.05, 0.1, 0.2]
results = []

for alpha in significance_levels:
    # EVT conformal decisions
    evt_decisions = p_values < alpha
    evt_fdr = false_discovery_rate(y_test, evt_decisions)
    evt_power = statistical_power(y_test, evt_decisions)
    
    # Standard conformal decisions
    std_decisions = p_values_standard < alpha
    std_fdr = false_discovery_rate(y_test, std_decisions)
    std_power = statistical_power(y_test, std_decisions)
    
    results.append({
        'alpha': alpha,
        'evt_fdr': evt_fdr,
        'evt_power': evt_power,
        'std_fdr': std_fdr,
        'std_power': std_power
    })

# Display results
print("\nPerformance Comparison:")
print("Alpha\tEVT FDR\tEVT Power\tStd FDR\tStd Power")
for r in results:
    print(f"{r['alpha']:.2f}\t{r['evt_fdr']:.3f}\t{r['evt_power']:.3f}\t\t{r['std_fdr']:.3f}\t{r['std_power']:.3f}")
```

## Threshold Selection Methods

```python
# Test different threshold selection methods
threshold_methods = [
    ("percentile", 0.90),
    ("percentile", 0.95),
    ("percentile", 0.99),
    ("top_k", 50),
    ("top_k", 100),
    ("mean_excess", 0.1)
]

threshold_results = []

for method, value in threshold_methods:
    detector = EVTConformalDetector(
        detector=LOF(n_neighbors=20),
        strategy=Split(calib_size=0.3),
        evt_threshold_method=method,
        evt_threshold_value=value,
        evt_min_tail_size=10,
        seed=42
    )
    
    detector.fit(x_train)
    
    # Check if EVT fitting succeeded
    if detector.gpd_params is not None:
        p_vals = detector.predict(x_test)
        decisions = p_vals < 0.05
        fdr = false_discovery_rate(y_test, decisions)
        power = statistical_power(y_test, decisions)
        
        threshold_results.append({
            'method': f"{method}({value})",
            'evt_threshold': detector.evt_threshold,
            'gpd_shape': detector.gpd_params[0],
            'fdr': fdr,
            'power': power
        })

# Display threshold comparison
print("\nThreshold Method Comparison:")
print("Method\t\tThreshold\tGPD Shape\tFDR\tPower")
for r in threshold_results:
    print(f"{r['method']:<15}\t{r['evt_threshold']:.3f}\t\t{r['gpd_shape']:.3f}\t\t{r['fdr']:.3f}\t{r['power']:.3f}")
```

## FDR Control with EVT

```python
# Apply FDR control to EVT conformal p-values
fdr_controlled_pvals = false_discovery_control(p_values, method='bh')
fdr_decisions = fdr_controlled_pvals < 0.05

# Compare with raw p-values
raw_decisions = p_values < 0.05

print("FDR Control Results:")
print(f"Raw p-values < 0.05: {raw_decisions.sum()} detections")
print(f"FDR controlled: {fdr_decisions.sum()} detections")
print(f"Raw FDR: {false_discovery_rate(y_test, raw_decisions):.3f}")
print(f"FDR controlled: {false_discovery_rate(y_test, fdr_decisions):.3f}")
```

## Multiple Detectors with EVT

```python
# Test EVT conformal with different base detectors
detectors = [
    ("LOF", LOF(n_neighbors=20)),
    ("IForest", IForest(n_estimators=100)),
    ("OCSVM", OCSVM(gamma='scale'))
]

detector_results = []

for name, base_detector in detectors:
    evt_det = EVTConformalDetector(
        detector=base_detector,
        strategy=Split(calib_size=0.3),
        evt_threshold_method="percentile",
        evt_threshold_value=0.95,
        seed=42
    )
    
    evt_det.fit(x_train)
    
    if evt_det.gpd_params is not None:
        p_vals = evt_det.predict(x_test)
        decisions = p_vals < 0.05
        fdr = false_discovery_rate(y_test, decisions)
        power = statistical_power(y_test, decisions)
        
        detector_results.append({
            'detector': name,
            'evt_success': True,
            'fdr': fdr,
            'power': power
        })
    else:
        detector_results.append({
            'detector': name,
            'evt_success': False,
            'fdr': np.nan,
            'power': np.nan
        })

# Display results
print("\nDetector Comparison with EVT:")
print("Detector\tEVT Success\tFDR\tPower")
for r in detector_results:
    if r['evt_success']:
        print(f"{r['detector']}\t\tYes\t\t{r['fdr']:.3f}\t{r['power']:.3f}")
    else:
        print(f"{r['detector']}\t\tNo\t\t-\t-")
```

## Advanced Configuration

```python
# Custom threshold function
def custom_threshold(scores):
    """Custom threshold based on interquartile range."""
    q75 = np.percentile(scores, 75)
    q25 = np.percentile(scores, 25)
    iqr = q75 - q25
    return q75 + 1.5 * iqr

# EVT detector with custom threshold
custom_detector = EVTConformalDetector(
    detector=LOF(n_neighbors=20),
    strategy=Split(calib_size=0.3),
    evt_threshold_method="custom",
    evt_threshold_value=custom_threshold,
    evt_min_tail_size=5,  # Lower minimum for custom threshold
    seed=42
)

custom_detector.fit(x_train)

if custom_detector.gpd_params is not None:
    print(f"Custom threshold: {custom_detector.evt_threshold:.3f}")
    p_vals_custom = custom_detector.predict(x_test)
    decisions_custom = p_vals_custom < 0.05
    fdr_custom = false_discovery_rate(y_test, decisions_custom)
    power_custom = statistical_power(y_test, decisions_custom)
    
    print(f"Custom EVT - FDR: {fdr_custom:.3f}, Power: {power_custom:.3f}")
```

## Visualization and Analysis

```python
import matplotlib.pyplot as plt

# Compare p-value distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# P-value histograms
ax1.hist(p_values, bins=50, alpha=0.7, label='EVT Conformal', density=True)
ax1.hist(p_values_standard, bins=50, alpha=0.7, label='Standard Conformal', density=True)
ax1.set_xlabel('P-values')
ax1.set_ylabel('Density')
ax1.set_title('P-value Distribution Comparison')
ax1.legend()

# Focus on extreme values
extreme_evt = p_values[p_values < 0.1]
extreme_std = p_values_standard[p_values_standard < 0.1]

ax2.hist(extreme_evt, bins=20, alpha=0.7, label='EVT Conformal', density=True)
ax2.hist(extreme_std, bins=20, alpha=0.7, label='Standard Conformal', density=True)
ax2.set_xlabel('P-values')
ax2.set_ylabel('Density')
ax2.set_title('Extreme P-values (p < 0.1)')
ax2.legend()

plt.tight_layout()
plt.savefig('evt_conformal_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC-style analysis
from sklearn.metrics import roc_curve, auc

# Calculate ROC curves
fpr_evt, tpr_evt, _ = roc_curve(y_test, 1 - p_values)
fpr_std, tpr_std, _ = roc_curve(y_test, 1 - p_values_standard)

auc_evt = auc(fpr_evt, tpr_evt)
auc_std = auc(fpr_std, tpr_std)

print(f"\nROC AUC Comparison:")
print(f"EVT Conformal: {auc_evt:.3f}")
print(f"Standard Conformal: {auc_std:.3f}")
```

## Key Takeaways

1. **EVT Enhancement**: EVT conformal detection provides better modeling of extreme anomalies
2. **Threshold Selection**: Choice of threshold method significantly impacts performance
3. **Robustness**: Automatic fallback to standard conformal when EVT fitting fails
4. **Extreme Value Focus**: Most beneficial for detecting truly extreme anomalies
5. **Computational Trade-off**: Slightly higher training cost for improved extreme value detection