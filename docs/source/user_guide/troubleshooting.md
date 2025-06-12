# Troubleshooting Guide

This guide addresses common issues you might encounter while using unquad and provides solutions.

## Common Issues and Solutions

### 1. Memory Issues

**Problem**: Running out of memory when using large datasets or certain detectors.

**Solutions**:
- Use batch processing for large datasets
- Consider using more memory-efficient detectors (e.g., IForest instead of KNN)
- Reduce the calibration set size
- Use sparse data structures when possible

### 2. Slow Performance

**Problem**: Processing takes too long, especially with large datasets.

**Solutions**:
- Use faster detectors (e.g., IForest, LOF)
- Reduce the calibration set size
- Use batch processing
- Consider using GPU-accelerated detectors when available
- Profile your code to identify bottlenecks

### 3. Invalid P-values

**Problem**: P-values don't seem to be properly calibrated.

**Solutions**:
- Ensure your calibration data is representative of the normal class
- Check for data leakage between training and calibration sets
- Verify that the detector is properly fitted
- Consider using a different conformal strategy
- Check for violations of the exchangeability assumption

### 4. High False Discovery Rate

**Problem**: Too many false positives even with FDR control.

**Solutions**:
- Increase the calibration set size
- Use a more conservative α level
- Consider using weighted conformal p-values if there's covariate shift
- Try different detectors
- Check for data quality issues

### 5. Low Detection Power

**Problem**: Missing too many anomalies.

**Solutions**:
- Decrease the α level
- Use more powerful detectors
- Consider using ensemble methods
- Try different conformal strategies
- Check if the anomalies are well-separated from normal data

### 6. Integration Issues

**Problem**: Problems integrating with other libraries or custom detectors.

**Solutions**:
- Ensure your detector implements the required interface
- Check for version compatibility
- Verify that the detector's output format matches expectations
- Consider using the adapter pattern for custom detectors

## Debugging Tips

1. **Enable Verbose Mode**:
```python
from unquad.estimation.properties.configuration import DetectorConfig

config = DetectorConfig(silent=False)
```

2. **Check Intermediate Results**:
```python
# Get raw scores before p-value conversion
raw_scores = detector.decision_function(X_test)
```

3. **Validate Data**:
```python
# Check for NaN values
print(f"NaN values in data: {np.isnan(X).any()}")

# Check data ranges
print(f"Data range: [{X.min()}, {X.max()}]")
```

4. **Monitor Memory Usage**:
```python
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

## Performance Optimization

1. **Batch Processing**:
```python
def process_in_batches(X, batch_size=1000):
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        # Process batch
```

2. **Caching**:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(x):
    # Your computation here
    pass
```

3. **Parallel Processing**:
```python
from joblib import Parallel, delayed

def parallel_process(X, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_item)(x) for x in X
    )
    return results
```

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/OliverHennhoefer/unquad/issues) for similar problems
2. Search the [Discussions](https://github.com/OliverHennhoefer/unquad/discussions) for solutions
3. Create a new issue with:
   - A minimal reproducible example
   - Expected vs actual behavior
   - System information
   - Relevant error messages 