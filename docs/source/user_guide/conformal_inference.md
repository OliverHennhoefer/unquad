# Understanding Conformal Inference

This guide explains the theoretical foundations and practical applications of conformal inference in anomaly detection using the new unquad API.

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
- Results are not interpretable in probabilistic terms

### The Conformal Solution

Conformal inference provides a principled way to convert scores to p-values:

```python
# Conformal approach - statistically valid p-values
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.split import SplitStrategy
from unquad.utils.func.enums import Aggregation

# Create conformal detector
strategy = SplitStrategy(calibration_size=0.2)
detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# Fit on training data (includes automatic calibration)
detector.fit(X_train)

# Get valid p-values
p_values = detector.predict(X_test, raw=False)

# Now we can control error rates!
anomalies = p_values < 0.05  # 5% significance level
```

## Mathematical Foundation

### Classical Conformal p-values

Given a scoring function $s(X)$ where higher scores indicate more anomalous behavior, and a calibration set $D_{calib} = \{X_1, \ldots, X_n\}$, the classical conformal p-value for a test instance $X_{test}$ is:

$$p_{classical}(X_{test}) = \frac{1 + \sum_{i=1}^{n} \mathbf{1}\{s(X_i) \geq s(X_{test})\}}{n+1}$$

where $\mathbf{1}\{\cdot\}$ is the indicator function.

### Statistical Validity

**Key Property**: If $X_{test}$ is exchangeable with the calibration data (i.e., drawn from the same distribution), then:

$$\mathbb{P}(p_{classical}(X_{test}) \leq \alpha) \leq \alpha$$

for any $\alpha \in (0,1)$.

This means that if we declare $X_{test}$ anomalous when $p_{classical}(X_{test}) \leq 0.05$, we'll have at most a 5% false positive rate.

### Intuitive Understanding

The p-value answers the question: "If this test instance were actually normal, what's the probability of observing an anomaly score at least as extreme as what we observed?"

- **High p-value (e.g., 0.8)**: The test instance looks very similar to calibration data
- **Medium p-value (e.g., 0.3)**: The test instance is somewhat unusual but not clearly anomalous  
- **Low p-value (e.g., 0.02)**: The test instance is very different from calibration data

## Exchangeability Assumption

### What is Exchangeability?

Exchangeability is weaker than the i.i.d. assumption. A sequence of random variables is exchangeable if their joint distribution is invariant to permutations.

**Practical interpretation**: If your test instance is truly normal, then the set $\{X_1, \ldots, X_n, X_{test}\}$ should behave as if all elements were drawn in random order from the same distribution.

### When Exchangeability Holds

```python
# Good: Test data from same distribution as training
X_train = load_normal_data(source='production', time='2023-01')
X_test = load_test_data(source='production', time='2023-03')

# All data comes from the same underlying process
detector.fit(X_train)
p_values = detector.predict(X_test, raw=False)
```

### When Exchangeability is Violated

```python
# Problem: Different sources, time periods, or conditions
X_train = load_normal_data(source='lab', time='2023-01')
X_test = load_test_data(source='production', time='2023-06')

# Solution: Use weighted conformal p-values
from unquad.estimation.weighted_conformal import WeightedConformalDetector

weighted_detector = WeightedConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
weighted_detector.fit(X_train)
p_values = weighted_detector.predict(X_test, raw=False)  # Automatically handles shift
```

## Practical Implementation

### Basic Setup

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from unquad.estimation.conformal import ConformalDetector
from unquad.strategy.split import SplitStrategy
from unquad.utils.func.enums import Aggregation

# 1. Prepare your data
X_train = load_normal_training_data()  # Normal data for training and calibration
X_test = load_test_data()  # Data to be tested

# 2. Create base detector
base_detector = IsolationForest(contamination=0.1, random_state=42)

# 3. Create conformal detector with strategy
strategy = SplitStrategy(calibration_size=0.2)  # 20% for calibration
detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)

# 4. Fit detector (automatically handles train/calibration split)
detector.fit(X_train)

# 5. Get p-values for test data
p_values = detector.predict(X_test, raw=False)
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

# P-value interpretation
for i, p_val in enumerate(p_values[:5]):
    if p_val < 0.01:
        print(f"Instance {i}: p={p_val:.4f} - Strong evidence of anomaly")
    elif p_val < 0.05:
        print(f"Instance {i}: p={p_val:.4f} - Moderate evidence of anomaly") 
    elif p_val < 0.1:
        print(f"Instance {i}: p={p_val:.4f} - Weak evidence of anomaly")
    else:
        print(f"Instance {i}: p={p_val:.4f} - Consistent with normal behavior")
```

## Strategies for Different Scenarios

### 1. Split Strategy

Best for large datasets where you can afford to hold out calibration data:

```python
from unquad.strategy.split import SplitStrategy

# Use 20% of data for calibration
strategy = SplitStrategy(calibration_size=0.2)

# Or use absolute number for very large datasets
strategy = SplitStrategy(calibration_size=1000)
```

### 2. Cross-Validation Strategy

Better utilization of data by using all samples for both training and calibration:

```python
from unquad.strategy.cross_val import CrossValidationStrategy

# 5-fold cross-validation
strategy = CrossValidationStrategy(n_splits=5)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
```

### 3. Bootstrap Strategy

Provides robust estimates through resampling:

```python
from unquad.strategy.bootstrap import BootstrapStrategy

# 100 bootstrap samples with 80% sampling ratio
strategy = BootstrapStrategy(n_bootstraps=100, sample_ratio=0.8)

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
```

### 4. Jackknife Strategy (Leave-One-Out)

Maximum use of small datasets:

```python
from unquad.strategy.jackknife import JackknifeStrategy

# Leave-one-out cross-validation
strategy = JackknifeStrategy()

detector = ConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
```

## Common Pitfalls and Solutions

### 1. Data Leakage

**Problem**: Using contaminated calibration data

```python
# Wrong: Training data contains anomalies
X_mixed = load_mixed_data()  # Contains both normal and anomalous
detector.fit(X_mixed)  # This invalidates the guarantees!
```

**Solution**: Ensure training data is clean

```python
# Correct: Use only verified normal data for training
X_clean = load_verified_normal_data()
detector.fit(X_clean)

# Or clean the data first
def clean_training_data(X, contamination_rate=0.05):
    temp_detector = IsolationForest(contamination=contamination_rate)
    temp_detector.fit(X)
    scores = temp_detector.decision_function(X)
    # Keep only the most normal samples
    threshold = np.percentile(scores, contamination_rate * 100)
    return X[scores >= threshold]

X_cleaned = clean_training_data(X_train)
detector.fit(X_cleaned)
```

### 2. Insufficient Calibration Data

**Problem**: Too few calibration samples lead to coarse p-values

```python
# Problematic: Only 10 calibration samples
# P-values will be coarse (0.09, 0.18, 0.27, ...)
X_small = X_normal[:50]
strategy = SplitStrategy(calibration_size=0.2)  # Only 10 samples for calibration
```

**Solution**: Use strategies that make better use of data

```python
# Better: Use jackknife for small datasets
strategy = JackknifeStrategy()  # Uses all data efficiently

# Or use bootstrap for medium datasets
strategy = BootstrapStrategy(n_bootstraps=100, sample_ratio=0.8)

# Or increase calibration size for split strategy
strategy = SplitStrategy(calibration_size=0.5)  # Use more data for calibration
```

### 3. Distribution Shift

**Problem**: Test distribution differs from training distribution

```python
# Problematic: Different conditions
X_train = load_data(season='summer', location='lab')
X_test = load_data(season='winter', location='production')
```

**Solution**: Use weighted conformal p-values

```python
from unquad.estimation.weighted_conformal import WeightedConformalDetector

# Weighted detector automatically estimates importance weights
weighted_detector = WeightedConformalDetector(
    detector=base_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
weighted_detector.fit(X_train)
p_values = weighted_detector.predict(X_test, raw=False)
```

### 4. Multiple Testing

**Problem**: Testing many instances increases false positive rate

```python
# Problem: With 1000 tests at α=0.05, expect ~50 false positives
p_values = detector.predict(X_test_large, raw=False)
raw_discoveries = (p_values < 0.05).sum()
print(f"Raw discoveries: {raw_discoveries}")  # Likely inflated
```

**Solution**: Use FDR control

```python
from scipy.stats import false_discovery_control

# Control False Discovery Rate at 5%
adjusted_p_values = false_discovery_control(p_values, method='bh', alpha=0.05)
controlled_discoveries = (adjusted_p_values < 0.05).sum()

print(f"Raw discoveries: {(p_values < 0.05).sum()}")
print(f"FDR-controlled discoveries: {controlled_discoveries}")
```

## Advanced Topics

### Raw Scores vs P-values

You can get both raw anomaly scores and p-values:

```python
# Get raw aggregated anomaly scores
raw_scores = detector.predict(X_test, raw=True)

# Get p-values
p_values = detector.predict(X_test, raw=False)

# Understand the relationship
import matplotlib.pyplot as plt
plt.scatter(raw_scores, p_values)
plt.xlabel('Raw Anomaly Score')
plt.ylabel('P-value')
plt.title('Score vs P-value Relationship')
plt.show()
```

### Aggregation Methods

When using ensemble strategies, you can control how multiple model outputs are combined:

```python
# Different aggregation methods
aggregation_methods = [Aggregation.MEAN, Aggregation.MEDIAN, Aggregation.MAX]

for agg_method in aggregation_methods:
    detector = ConformalDetector(
        detector=base_detector,
        strategy=CrossValidationStrategy(n_splits=5),
        aggregation=agg_method,
        seed=42
    )
    detector.fit(X_train)
    p_values = detector.predict(X_test, raw=False)
    
    print(f"{agg_method.value}: {(p_values < 0.05).sum()} detections")
```

### Custom Scoring Functions

For advanced users, you can create custom detectors:

```python
from pyod.models.base import BaseDetector

class CustomDetector(BaseDetector):
    """Custom anomaly detector following PyOD interface."""
    
    def __init__(self, contamination=0.1):
        super().__init__(contamination=contamination)
    
    def fit(self, X, y=None):
        # Your custom fitting logic here
        self.decision_scores_ = self._compute_scores(X)
        self._process_decision_scores()
        return self
    
    def decision_function(self, X):
        # Your custom scoring logic here
        return self._compute_scores(X)
    
    def _compute_scores(self, X):
        # Higher scores should indicate more anomalous behavior
        # This is a dummy implementation
        return np.random.random(len(X))

# Use with conformal detection
custom_detector = CustomDetector()
detector = ConformalDetector(
    detector=custom_detector,
    strategy=strategy,
    aggregation=Aggregation.MEDIAN,
    seed=42
)
```

## Performance Considerations

### Computational Complexity

Different strategies have different computational costs:

```python
import time

strategies = {
    'Split': SplitStrategy(calibration_size=0.2),
    'Cross-Val (5-fold)': CrossValidationStrategy(n_splits=5),
    'Bootstrap (50)': BootstrapStrategy(n_bootstraps=50, sample_ratio=0.8),
    'Jackknife': JackknifeStrategy()
}

for name, strategy in strategies.items():
    start_time = time.time()
    
    detector = ConformalDetector(
        detector=base_detector,
        strategy=strategy,
        aggregation=Aggregation.MEDIAN,
        seed=42,
        silent=True
    )
    detector.fit(X_train)
    p_values = detector.predict(X_test, raw=False)
    
    elapsed = time.time() - start_time
    print(f"{name}: {elapsed:.2f}s ({(p_values < 0.05).sum()} detections)")
```

### Memory Usage

For large datasets, consider:

```python
# Use batch processing for very large test sets
def predict_in_batches(detector, X_test, batch_size=1000):
    n_batches = (len(X_test) + batch_size - 1) // batch_size
    all_p_values = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_test))
        batch = X_test[start_idx:end_idx]
        
        batch_p_values = detector.predict(batch, raw=False)
        all_p_values.extend(batch_p_values)
    
    return np.array(all_p_values)

# Usage for large datasets
p_values = predict_in_batches(detector, X_test_large)
```

## Next Steps

- Learn about [different conformalization strategies](conformalization_strategies.md) in detail
- Understand [weighted conformal p-values](weighted_conformal.md) for handling distribution shift
- Explore [FDR control](fdr_control.md) for multiple testing scenarios
- Check out [best practices](best_practices.md) for production deployment
- Review the [troubleshooting guide](troubleshooting.md) for common issues