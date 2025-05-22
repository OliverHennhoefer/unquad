# Understanding Conformal Inference

This guide explains the theoretical foundations and practical applications of conformal inference in anomaly detection.

## What is Conformal Inference?

Conformal inference is a framework for creating prediction intervals or hypothesis tests with finite-sample validity guarantees. In the context of anomaly detection, it transforms raw anomaly scores into statistically valid p-values.

### The Problem with Traditional Anomaly Detection

Traditional anomaly detection methods output scores and require setting arbitrary thresholds:

```python
# Traditional approach - arbitrary threshold
scores = detector.decision_function(X_test)
anomalies = scores < -0.5  # Why -0.5? No statistical justification!
```

This approach has several issues:
- No statistical guarantees about error rates
- Threshold selection is often arbitrary
- No control over false positive rates

### The Conformal Solution

Conformal inference provides a principled way to convert scores to p-values:

```python
# Conformal approach - statistically valid p-values
from unquad.conformal import ClassicalCAD

cad = ClassicalCAD(detector)
cad.fit(X_calibration)  # Fit on clean calibration data
p_values = cad.predict_proba(X_test)  # Get valid p-values

# Now we can control error rates!
anomalies = p_values < 0.05  # 5% significance level
```

## Mathematical Foundation

### Classical Conformal p-values

Given a scoring function $s(X)$ where higher scores indicate normalcy, and a calibration set $D_{calib} = \{X_1, \ldots, X_n\}$, the classical conformal p-value for a test instance $X_{test}$ is:

$$p_{classical}(X_{test}) = \frac{1 + \sum_{i=1}^{n} \mathbf{1}\{s(X_i) \leq s(X_{test})\}}{n+1}$$

where $\mathbf{1}\{\cdot\}$ is the indicator function.

### Statistical Validity

**Key Property**: If $X_{test}$ is exchangeable with the calibration data (i.e., drawn from the same distribution), then:

$$\mathbb{P}(p_{classical}(X_{test}) \leq \alpha) \leq \alpha$$

for any $\alpha \in (0,1)$.

This means that if we declare $X_{test}$ anomalous when $p_{classical}(X_{test}) \leq 0.05$, we'll have at most a 5% false positive rate.

## Exchangeability Assumption

### What is Exchangeability?

Exchangeability is weaker than the i.i.d. assumption. A sequence of random variables is exchangeable if their joint distribution is invariant to permutations.

**Practical interpretation**: If your test instance is truly normal, then the set $\{X_1, \ldots, X_n, X_{test}\}$ should behave as if all elements were drawn in random order.

### When Exchangeability Holds

```python
# Good: Test data from same distribution as calibration
X_train = load_normal_data(source='production', time='2023-01')
X_calib = load_normal_data(source='production', time='2023-02') 
X_test = load_test_data(source='production', time='2023-03')
```

### When Exchangeability is Violated

```python
# Problem: Different sources or time periods
X_calib = load_normal_data(source='lab', time='2023-01')
X_test = load_test_data(source='production', time='2023-06')
# Solution: Use weighted conformal p-values (see next section)
```

## Practical Implementation

### Basic Setup

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from unquad.conformal import ClassicalCAD

# 1. Prepare your data
X_normal = load_normal_training_data()
X_calibration = load_normal_calibration_data()  # Hold-out normal data
X_test = load_test_data()  # Data to be tested

# 2. Train your base detector
detector = IsolationForest(contamination=0.1, random_state=42)

# 3. Create conformal wrapper
cad = ClassicalCAD(detector)

# 4. Fit on calibration data
cad.fit(X_calibration)

# 5. Get p-values for test data
p_values = cad.predict_proba(X_test)
```

### Understanding the Output

```python
# p-values are between 0 and 1
print(f"P-values range: [{p_values.min():.4f}, {p_values.max():.4f}]")

# Small p-values indicate anomalies
suspicious_indices = np.where(p_values < 0.05)[0]
print(f"Suspicious instances: {len(suspicious_indices)}")

# Very small p-values are strong evidence
very_suspicious = np.where(p_values < 0.01)[0]
print(f"Very suspicious instances: {len(very_suspicious)}")
```

## Scoring Function Requirements

### Higher Scores = More Normal

The conformal framework assumes that higher scores indicate greater conformity to the normal distribution:

```python
# Good: Higher scores for normal instances
scores_normal = [0.8, 0.9, 0.85, 0.92]    # Normal instances
scores_anomalous = [0.1, 0.2, 0.05, 0.15] # Anomalous instances

# Bad: Lower scores for normal instances  
scores_normal = [-0.1, -0.2, -0.05]       # This breaks the assumption!
scores_anomalous = [-0.8, -0.9, -0.95]
```

### Handling Different Detector Types

```python
# Some detectors output "anomaly scores" (higher = more anomalous)
# Need to flip these for conformal inference

class FlippedDetector:
    def __init__(self, base_detector):
        self.base_detector = base_detector
    
    def fit(self, X):
        return self.base_detector.fit(X)
    
    def decision_function(self, X):
        # Flip sign so higher = more normal
        return -self.base_detector.decision_function(X)

# Usage
detector = FlippedDetector(IsolationForest())
cad = ClassicalCAD(detector)
```

## Common Pitfalls and Solutions

### 1. Data Leakage

**Problem**: Using contaminated calibration data
```python
# Wrong: Calibration data contains anomalies
X_mixed = load_mixed_data()  # Contains both normal and anomalous
cad.fit(X_mixed)  # This invalidates the guarantees!
```

**Solution**: Ensure calibration data is clean
```python
# Correct: Use only verified normal data for calibration
X_clean_calibration = load_verified_normal_data()
cad.fit(X_clean_calibration)
```

### 2. Insufficient Calibration Data

**Problem**: Too few calibration samples
```python
# Problematic: Only 10 calibration samples
X_calib_small = X_normal[:10]  # p-values will be coarse (0.09, 0.18, 0.27, ...)
```

**Solution**: Use resampling-based methods
```python
from unquad.conformal import LOOCAD

# Better: Leave-One-Out increases effective calibration size
loo_cad = LOOCAD(detector)
loo_cad.fit(X_normal)  # Uses all data efficiently
```

### 3. Distribution Shift

**Problem**: Test distribution differs from calibration
```python
# Problematic: Different seasons, locations, etc.
X_calib = load_data(season='summer')
X_test = load_data(season='winter')
```

**Solution**: Use weighted conformal p-values
```python
from unquad.conformal import WeightedCAD

weighted_cad = WeightedCAD(detector)
weights = estimate_importance_weights(X_test, X_calib)
p_values = weighted_cad.predict_proba(X_test, weights=weights)
```

## Next Steps

- Learn about [different conformalization strategies](conformalization_strategies.md)
- Understand [weighted conformal p-values](weighted_conformal.md) for handling distribution shift
- Explore [FDR control](fdr_control.md) for multiple testing scenarios