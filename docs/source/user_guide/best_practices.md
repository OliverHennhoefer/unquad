# Best Practices Guide

This guide provides recommendations for using unquad effectively in different scenarios.

## Data Preparation

### 1. Data Quality

- Ensure your data is clean and preprocessed
- Handle missing values appropriately
- Normalize or standardize features when necessary
- Remove or handle outliers in the training data
- Check for data leakage between training and test sets

### 2. Feature Engineering

- Use domain knowledge to create relevant features
- Consider feature selection to reduce dimensionality
- Handle categorical variables appropriately
- Create features that capture temporal patterns if applicable
- Consider feature interactions

## Model Selection

### 1. Choosing a Detector

Consider the following when selecting a detector:

- **Data Size**: 
  - Small datasets: Use simpler models (IForest, LOF)
  - Large datasets: Consider scalable models (SUOD, LODA)
  - High-dimensional data: Use PCA-based or deep learning models

- **Data Characteristics**:
  - Linear patterns: Use PCA, OCSVM
  - Non-linear patterns: Use IForest, LOF, KNN
  - Complex patterns: Use deep learning models (AE, DIF)

- **Computational Resources**:
  - Limited resources: Use lightweight models (IForest, LOF)
  - GPU available: Consider deep learning models
  - Distributed computing: Use SUOD

### 2. Ensemble Methods

Consider using ensemble methods for improved performance:

```python
from pyod.models.lscp import LSCP
from pyod.models.lof import LOF
from pyod.models.iforest import IForest

# Create base detectors
detectors = [
    LOF(contamination=0.1),
    IForest(contamination=0.1)
]

# Create ensemble
ensemble = LSCP(detectors, contamination=0.1)
```

## Conformal Strategy Selection

### 1. Split-Conformal

Best for:
- Large datasets
- When computational efficiency is important
- When you have enough data for calibration

```python
from unquad.strategy.split import Split

strategy = Split(calib_size=1000)
```

### 2. Leave-One-Out

Best for:
- Small datasets
- When you need maximum power
- When computational cost is not a concern

```python
from unquad.strategy.loo import LOO

strategy = LOO()
```

### 3. Bootstrap

Best for:
- Medium-sized datasets
- When you need robust estimates
- When you want to balance efficiency and power

```python
from unquad.strategy.bootstrap import Bootstrap

strategy = Bootstrap(resampling_ratio=0.8, n_bootstraps=20)
```

## Calibration

### 1. Calibration Set Size

- Use at least 1000 points for reliable calibration
- For small datasets, consider using LOO or bootstrap strategies
- Balance calibration set size with training set size

### 2. Calibration Data Quality

- Ensure calibration data is representative of normal class
- Avoid using contaminated data for calibration
- Consider using domain knowledge to select calibration data

## FDR Control

### 1. Alpha Selection

- Start with α = 0.05 for general use
- Use smaller α (e.g., 0.01) for critical applications
- Consider the cost of false positives in your domain

### 2. Multiple Testing

- Always use FDR control when testing multiple instances
- Consider using weighted p-values for dependent tests
- Monitor the empirical FDR in your application

## Performance Monitoring

### 1. Metrics to Track

- False Discovery Rate
- Statistical Power
- Computational Time
- Memory Usage
- Calibration Quality

### 2. Monitoring Tools

```python
from unquad.utils.metrics import false_discovery_rate, statistical_power
import time

def monitor_performance(cad, X_test, y_test):
    start_time = time.time()
    p_values = cad.predict_proba(X_test)
    end_time = time.time()
    
    fdr = false_discovery_rate(y_test, p_values < 0.05)
    power = statistical_power(y_test, p_values < 0.05)
    
    print(f"FDR: {fdr:.3f}")
    print(f"Power: {power:.3f}")
    print(f"Time: {end_time - start_time:.2f}s")
```

## Production Deployment

### 1. Model Updates

- Implement a strategy for model updates
- Monitor model drift
- Consider using weighted conformal p-values for distributional shifts

### 2. Scalability

- Use batch processing for large datasets
- Implement caching for frequently used computations
- Consider distributed computing for large-scale deployments

### 3. Monitoring

- Set up alerts for performance degradation
- Monitor resource usage
- Track detection rates and FDR

## Code Organization

### 1. Configuration Management

```python
from dataclasses import dataclass

@dataclass
class AnomalyDetectionConfig:
    alpha: float = 0.05
    calib_size: int = 1000
    detector_type: str = "iforest"
    strategy: str = "split"
```

### 2. Pipeline Organization

```python
class AnomalyDetectionPipeline:
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.detector = self._create_detector()
        self.strategy = self._create_strategy()
        self.cad = self._create_cad()
    
    def _create_detector(self):
        # Detector creation logic
        pass
    
    def _create_strategy(self):
        # Strategy creation logic
        pass
    
    def _create_cad(self):
        # CAD creation logic
        pass
    
    def fit(self, X):
        # Fitting logic
        pass
    
    def predict(self, X):
        # Prediction logic
        pass
``` 