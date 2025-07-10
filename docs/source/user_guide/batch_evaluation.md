# Batch Evaluation

Controlled batch generation for systematic anomaly detection evaluation with configurable contamination levels.

## Overview

The `BatchGenerator` creates evaluation batches with precise anomaly contamination control. It ensures reproducible experiments by managing the mixing of normal and anomalous instances across multiple batches.

## Basic Usage

```python
from unquad.utils.data.generator.batch import BatchGenerator
from unquad.utils.data.load import load_shuttle

# Load and separate data
x_train, x_test, y_test = load_shuttle(setup=True)
x_normal = x_test[y_test == 0]
x_anomaly = x_test[y_test == 1]

# Create batch generator
batch_gen = BatchGenerator(
    x_normal=x_normal,
    x_anomaly=x_anomaly,
    batch_size=100,
    anomaly_proportion=0.1,  # 10% anomalies per batch
    random_state=42
)

# Generate batches
for x_batch, y_batch in batch_gen.generate(n_batches=5):
    print(f"Batch shape: {x_batch.shape}, Anomalies: {y_batch.sum()}")
```

## Contamination Control

### Percentage-based Contamination
```python
# 15% anomalies per batch
batch_gen = BatchGenerator(
    x_normal=x_normal,
    x_anomaly=x_anomaly,
    batch_size=200,
    anomaly_proportion=0.15
)
```

### Fixed Count Contamination
```python
# Exactly 25 anomalies per batch
batch_gen = BatchGenerator(
    x_normal=x_normal,
    x_anomaly=x_anomaly,
    batch_size=200,
    anomaly_proportion=25
)
```

## Evaluation Patterns

### Finite Batch Generation
```python
# Generate specific number of batches
for x_batch, y_batch in batch_gen.generate(n_batches=10):
    # Evaluate model on batch
    p_values = detector.predict(x_batch)
    fdr = false_discovery_rate(y_batch, p_values < 0.05)
    print(f"Batch FDR: {fdr:.3f}")
```

### Infinite Batch Generation
```python
# Generate batches indefinitely
batch_count = 0
for x_batch, y_batch in batch_gen.generate():
    # Process batch
    results = evaluate_batch(x_batch, y_batch)
    
    batch_count += 1
    if batch_count >= 100:  # Stop after 100 batches
        break
```

## Convenience Functions

### Dataset Splitting

```python
from unquad.utils.data.generator.batch import create_batch_generator

# Load complete dataset
df = load_shuttle()

# Create training data and batch generator
x_train, batch_gen = create_batch_generator(
    df,
    train_size=0.5,  # 50% of normal data for training
    batch_size=150,
    anomaly_proportion=0.2,  # 20% anomalies per batch
    random_state=42
)

# Train detector
detector.fit(x_train)

# Evaluate on batches
for x_batch, y_batch in batch_gen.generate(n_batches=20):
    p_values = detector.predict(x_batch)
    # Process results
```

## Advanced Configuration

### Reproducibility Control
```python
# Initialize with seed
batch_gen = BatchGenerator(
    x_normal=x_normal,
    x_anomaly=x_anomaly,
    batch_size=100,
    anomaly_proportion=0.1,
    random_state=42
)

# Reset generator to initial state
batch_gen.reset()

# Generate identical sequence again
for x_batch, y_batch in batch_gen.generate(n_batches=5):
    # Same batches as before
    pass
```

### Generator Properties
```python
# Check generator configuration
print(f"Normal instances: {batch_gen.n_normal}")
print(f"Anomaly instances: {batch_gen.n_anomaly}")
print(f"Anomalies per batch: {batch_gen.n_anomaly_per_batch}")
print(f"Max unique batches: {batch_gen.max_batches}")
```

## Performance Considerations

**Memory Usage:**
- Loads entire dataset into memory
- Efficient sampling without replacement
- Minimal overhead during generation

**Computational Efficiency:**
- Fast random sampling
- No duplicate prevention across batches
- Configurable batch size for memory management

## Use Cases

### Model Evaluation
```python
# Systematic evaluation across contamination levels
contamination_levels = [0.05, 0.1, 0.15, 0.2]
results = {}

for contamination in contamination_levels:
    batch_gen = BatchGenerator(
        x_normal=x_normal,
        x_anomaly=x_anomaly,
        batch_size=100,
        anomaly_proportion=contamination,
        random_state=42
    )
    
    fdrs = []
    for x_batch, y_batch in batch_gen.generate(n_batches=10):
        p_values = detector.predict(x_batch)
        fdr = false_discovery_rate(y_batch, p_values < 0.05)
        fdrs.append(fdr)
    
    results[contamination] = np.mean(fdrs)
```

### Hyperparameter Tuning
```python
# Evaluate different detector configurations
detectors = [LOF(n_neighbors=k) for k in [5, 10, 20, 50]]
performances = []

for detector in detectors:
    # Train detector
    detector.fit(x_train)
    
    # Evaluate on consistent batches
    batch_gen.reset()  # Ensure identical batches
    scores = []
    
    for x_batch, y_batch in batch_gen.generate(n_batches=10):
        p_values = detector.predict(x_batch)
        score = statistical_power(y_batch, p_values < 0.05)
        scores.append(score)
    
    performances.append(np.mean(scores))
```

## Integration with Evaluation Metrics

```python
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power


# Comprehensive evaluation
def evaluate_detector(detector, batch_gen, n_batches=10):
    fdrs = []
    powers = []

    for x_batch, y_batch in batch_gen.generate(n_batches=n_batches):
        p_values = detector.predict(x_batch)
        decisions = p_values < 0.05

        fdr = false_discovery_rate(y_batch, decisions)
        power = statistical_power(y_batch, decisions)

        fdrs.append(fdr)
        powers.append(power)

    return {
        'mean_fdr': np.mean(fdrs),
        'mean_power': np.mean(powers),
        'fdr_std': np.std(fdrs),
        'power_std': np.std(powers)
    }


# Usage
results = evaluate_detector(detector, batch_gen)
print(f"FDR: {results['mean_fdr']:.3f} ± {results['fdr_std']:.3f}")
print(f"Power: {results['mean_power']:.3f} ± {results['power_std']:.3f}")
```