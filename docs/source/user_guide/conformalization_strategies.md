# Conformalization Strategies

This guide explains the different strategies available in `unquad` for conformal anomaly detection.

## Available Strategies

### Split Strategy

The simplest strategy that splits the data into training and calibration sets.

```python
from unquad.strategy import Split

strategy = Split(calib_size=0.1)  # Use 10% of data for calibration
```

### Cross-validation Strategy

Uses k-fold cross-validation to create multiple training and calibration sets.

```python
from unquad.strategy import CrossValidation

strategy = CrossValidation(k=5, plus=False)  # 5-fold CV
```

### Bootstrap Strategy

Uses bootstrap resampling to create multiple training sets.

```python
from unquad.strategy import Bootstrap

strategy = Bootstrap(n_bootstraps=100, resampling_ratio=0.8)
```

### Jackknife Strategy

A special case of cross-validation where each sample is used once for calibration.

```python
from unquad.strategy import Jackknife

strategy = Jackknife(plus=False)
```

## Choosing a Strategy

- Use **Split** for simple, fast applications
- Use **Cross-validation** for robust calibration
- Use **Bootstrap** for uncertainty estimation
- Use **Jackknife** for small datasets

## Next Steps

- Learn about [weighted conformal p-values](weighted_conformal.md)
- Understand [FDR control](fdr_control.md) for multiple testing 