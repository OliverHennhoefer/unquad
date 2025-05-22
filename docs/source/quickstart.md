# Quickstart Guide

This guide will get you started with `unquad` in just a few minutes.

## Basic Usage

### 1. Classical Conformal Anomaly Detection

The most straightforward way to use unquad is with classical conformal anomaly detection:

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
from unquad.conformal import ClassicalCAD

# Generate some example data
X_normal, _ = make_blobs(n_samples=1000, centers=1, random_state=42)
X_test, _ = make_blobs(n_samples=100, centers=1, random_state=123)

# Add some anomalies to test set
X_anomalies = np.random.uniform(-10, 10, (20, X_test.shape[1]))
X_test = np.vstack([X_test, X_anomalies])

# Initialize detector
detector = IsolationForest(contamination=0.1, random_state=42)

# Create conformal anomaly detector
cad = ClassicalCAD(detector)

# Fit on normal data
cad.fit(X_normal)

# Get p-values for test instances
p_values = cad.predict_proba(X_test)

print(f"P-values range: {p_values.min():.4f} - {p_values.max():.4f}")
print(f"Number of potential anomalies (p < 0.05): {(p_values < 0.05).sum()}")
```

### 2. False Discovery Rate Control

Control the False Discovery Rate using the Benjamini-Hochberg procedure:

```python
from unquad.multiple_testing import benjamini_hochberg

# Control FDR at 5% level
discoveries, adjusted_p_values = benjamini_hochberg(p_values, alpha=0.05)

print(f"Number of discoveries: {discoveries.sum()}")
print(f"Adjusted p-values range: {adjusted_p_values.min():.4f} - {adjusted_p_values.max():.4f}")

# Get indices of discovered anomalies
anomaly_indices = np.where(discoveries)[0]
print(f"Discovered anomaly indices: {anomaly_indices}")
```

### 3. Resampling-based Strategies

For better performance in low-data regimes, use resampling-based strategies:

```python
from unquad.conformal import LOOCAD, CrossConformalCAD

# Leave-One-Out Conformal Anomaly Detection
loo_cad = LOOCAD(detector)
loo_cad.fit(X_normal)
loo_p_values = loo_cad.predict_proba(X_test)

# Cross-Conformal Anomaly Detection
cv_cad = CrossConformalCAD(detector, cv_folds=5)
cv_cad.fit(X_normal)
cv_p_values = cv_cad.predict_proba(X_test)

print("Comparison of strategies:")
print(f"Classical: {(p_values < 0.05).sum()} detections")
print(f"LOO: {(loo_p_values < 0.05).sum()} detections")
print(f"Cross-Conformal: {(cv_p_values < 0.05).sum()} detections")
```

## Weighted Conformal p-values

When dealing with covariate shift, use weighted conformal p-values:

```python
from unquad.conformal import WeightedCAD

# Assume we have importance weights for test instances
# In practice, these would be estimated from domain knowledge or methods
weights = np.ones(len(X_test))  # Placeholder weights

# Create weighted conformal anomaly detector
weighted_cad = WeightedCAD(detector)
weighted_cad.fit(X_normal)

# Get weighted p-values
weighted_p_values = weighted_cad.predict_proba(X_test, weights=weights)

print(f"Weighted p-values range: {weighted_p_values.min():.4f} - {weighted_p_values.max():.4f}")
```

## Integration with PyOD

unquad integrates seamlessly with PyOD detectors:

```python
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

# Try different PyOD detectors
detectors = {
    'KNN': KNN(contamination=0.1),
    'LOF': LOF(contamination=0.1),
    'OCSVM': OCSVM(contamination=0.1)
}

results = {}
for name, detector in detectors.items():
    cad = ClassicalCAD(detector)
    cad.fit(X_normal)
    p_vals = cad.predict_proba(X_test)
    detections = (p_vals < 0.05).sum()
    results[name] = detections
    print(f"{name}: {detections} detections")
```

## Complete Example

Here's a complete example that ties everything together:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
from unquad.conformal import ClassicalCAD
from unquad.multiple_testing import benjamini_hochberg

# Generate data
np.random.seed(42)
X_normal, _ = make_blobs(n_samples=500, centers=1, cluster_std=1.0, random_state=42)
X_test_normal, _ = make_blobs(n_samples=80, centers=1, cluster_std=1.0, random_state=123)
X_test_anomalies = np.random.uniform(-6, 6, (20, 2))
X_test = np.vstack([X_test_normal, X_test_anomalies])

# True labels (0 = normal, 1 = anomaly)
y_true = np.hstack([np.zeros(80), np.ones(20)])

# Setup and fit detector
detector = IsolationForest(contamination=0.1, random_state=42)
cad = ClassicalCAD(detector)
cad.fit(X_normal)

# Get p-values and control FDR
p_values = cad.predict_proba(X_test)
discoveries, _ = benjamini_hochberg(p_values, alpha=0.05)

# Evaluate results
true_positives = np.sum(discoveries & (y_true == 1))
false_positives = np.sum(discoveries & (y_true == 0))
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / np.sum(y_true == 1)

print(f"Results with FDR control at 5%:")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Empirical FDR: {false_positives / max(1, discoveries.sum()):.3f}")
```

## Next Steps

- Read the [User Guide](user_guide/conformal_inference.md) for detailed explanations
- Check out the [Examples](examples/index.md) for more complex use cases
- Explore the [API Reference](api/unquad/index.rst) for detailed documentation