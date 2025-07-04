# Streaming Evaluation

Deterministic single-instance streaming for online anomaly detection evaluation with precise contamination control.

## Overview

The `OnlineGenerator` creates deterministic streams of single instances with exact anomaly proportions. It uses a pre-generated pool to ensure reproducible experiments and precise contamination control for streaming evaluation scenarios.

## Basic Usage

```python
from unquad.utils.data.online_generator import OnlineGenerator
from unquad.utils.data.load import load_shuttle

# Load and separate data
x_train, x_test, y_test = load_shuttle(setup=True)
x_normal = x_test[y_test == 0]
x_anomaly = x_test[y_test == 1]

# Create online generator with 1% anomalies
online_gen = OnlineGenerator(
    x_normal=x_normal,
    x_anomaly=x_anomaly,
    anomaly_proportion=0.01,
    pool_size=1000,
    random_state=42
)

# Generate stream
anomaly_count = 0
for x_instance, y_label in online_gen.generate_stream(n_instances=100):
    # Process single instance
    p_value = detector.predict(x_instance)
    anomaly_count += y_label

print(f"Anomalies in 100 instances: {anomaly_count}")  # Exactly 1
```

## Deterministic Contamination

### Exact Proportion Control
```python
# Create generator with precisely 2.5% anomalies
online_gen = OnlineGenerator(
    x_normal=x_normal,
    x_anomaly=x_anomaly,
    anomaly_proportion=0.025,  # Exactly 2.5%
    pool_size=1000,           # 25 anomalies in pool
    random_state=42
)

# Verify exact count
expected_anomalies = online_gen.get_exact_anomaly_count(1000)
print(f"Anomalies in 1000 instances: {expected_anomalies}")  # Exactly 25
```

### Pool Size Configuration
```python
# Small pool for quick cycling
online_gen = OnlineGenerator(
    x_normal=x_normal,
    x_anomaly=x_anomaly,
    anomaly_proportion=0.1,
    pool_size=100,  # Small pool, cycles every 100 instances
    random_state=42
)

# Large pool for extended streams
online_gen = OnlineGenerator(
    x_normal=x_normal,
    x_anomaly=x_anomaly,
    anomaly_proportion=0.01,
    pool_size=10000,  # Large pool for long streams
    random_state=42
)
```

## Streaming Patterns

### Finite Streams
```python
# Process exactly 500 instances
predictions = []
labels = []

for x_instance, y_label in online_gen.generate_stream(n_instances=500):
    p_value = detector.predict(x_instance)
    predictions.append(p_value[0])
    labels.append(y_label)

# Analyze results
fdr = false_discovery_rate(labels, np.array(predictions) < 0.05)
print(f"Stream FDR: {fdr:.3f}")
```

### Infinite Streams
```python
# Continuous processing
instance_count = 0
running_fdr = []

for x_instance, y_label in online_gen.generate_stream():
    p_value = detector.predict(x_instance)
    
    # Update running statistics
    instance_count += 1
    
    # Evaluate every 100 instances
    if instance_count % 100 == 0:
        # Calculate FDR for last 100 instances
        recent_fdr = calculate_recent_fdr(last_100_predictions, last_100_labels)
        running_fdr.append(recent_fdr)
        
        if instance_count >= 1000:  # Stop after 1000 instances
            break
```

## Advanced Features

### Sampling Configuration
```python
# With replacement (default)
online_gen = OnlineGenerator(
    x_normal=x_normal,
    x_anomaly=x_anomaly,
    anomaly_proportion=0.05,
    replacement=True  # Can sample same instance multiple times
)

# Without replacement
online_gen = OnlineGenerator(
    x_normal=x_normal,
    x_anomaly=x_anomaly,
    anomaly_proportion=0.05,
    replacement=False,  # Each instance used only once
    pool_size=min(1000, len(x_normal) + len(x_anomaly))
)
```

### Stream Control
```python
# Check current position
print(f"Current position: {online_gen.current_position}")

# Reset to beginning
online_gen.reset()

# Predict exact anomaly count
anomalies_in_next_200 = online_gen.get_exact_anomaly_count(200)
print(f"Anomalies in next 200: {anomalies_in_next_200}")
```

## Performance Analysis

### Latency Measurement
```python
import time

# Measure per-instance processing time
latencies = []
start_time = time.time()

for x_instance, y_label in online_gen.generate_stream(n_instances=1000):
    instance_start = time.time()
    
    # Process instance
    p_value = detector.predict(x_instance)
    
    instance_end = time.time()
    latencies.append(instance_end - instance_start)

total_time = time.time() - start_time
print(f"Average latency: {np.mean(latencies):.4f}s")
print(f"Throughput: {1000/total_time:.1f} instances/second")
```

### Memory Usage
```python
# Monitor memory usage during streaming
import psutil
import os

process = psutil.Process(os.getpid())
memory_usage = []

for i, (x_instance, y_label) in enumerate(online_gen.generate_stream(n_instances=1000)):
    # Process instance
    p_value = detector.predict(x_instance)
    
    # Record memory usage every 100 instances
    if i % 100 == 0:
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_usage.append(memory_mb)

print(f"Memory usage: {np.mean(memory_usage):.1f} MB")
```

## Use Cases

### Online Model Evaluation
```python
# Evaluate model performance on streaming data
def evaluate_streaming_detector(detector, online_gen, n_instances=1000):
    predictions = []
    labels = []
    
    for x_instance, y_label in online_gen.generate_stream(n_instances=n_instances):
        p_value = detector.predict(x_instance)
        predictions.append(p_value[0])
        labels.append(y_label)
    
    # Calculate metrics
    decisions = np.array(predictions) < 0.05
    fdr = false_discovery_rate(labels, decisions)
    power = statistical_power(labels, decisions)
    
    return {
        'fdr': fdr,
        'power': power,
        'mean_p_value': np.mean(predictions),
        'anomaly_rate': np.mean(labels)
    }

# Usage
results = evaluate_streaming_detector(detector, online_gen)
print(f"Streaming FDR: {results['fdr']:.3f}")
print(f"Streaming Power: {results['power']:.3f}")
```

### Concept Drift Detection
```python
# Monitor performance over time windows
window_size = 100
window_results = []

current_window_preds = []
current_window_labels = []

for i, (x_instance, y_label) in enumerate(online_gen.generate_stream(n_instances=1000)):
    p_value = detector.predict(x_instance)
    
    current_window_preds.append(p_value[0])
    current_window_labels.append(y_label)
    
    # Process completed window
    if len(current_window_preds) == window_size:
        decisions = np.array(current_window_preds) < 0.05
        window_fdr = false_discovery_rate(current_window_labels, decisions)
        window_results.append(window_fdr)
        
        # Reset window
        current_window_preds = []
        current_window_labels = []

# Analyze drift
print(f"FDR variation: {np.std(window_results):.3f}")
```

## Integration with Batch Evaluation

```python
from unquad.utils.data.batch_generator import BatchGenerator

# Compare streaming vs batch evaluation
batch_gen = BatchGenerator(
    x_normal=x_normal,
    x_anomaly=x_anomaly,
    batch_size=100,
    anomaly_proportion=0.05,
    random_state=42
)

# Batch evaluation
batch_fdrs = []
for x_batch, y_batch in batch_gen.generate(n_batches=10):
    p_values = detector.predict(x_batch)
    fdr = false_discovery_rate(y_batch, p_values < 0.05)
    batch_fdrs.append(fdr)

# Streaming evaluation
online_gen.reset()
streaming_preds = []
streaming_labels = []

for x_instance, y_label in online_gen.generate_stream(n_instances=1000):
    p_value = detector.predict(x_instance)
    streaming_preds.append(p_value[0])
    streaming_labels.append(y_label)

streaming_fdr = false_discovery_rate(
    streaming_labels, 
    np.array(streaming_preds) < 0.05
)

print(f"Batch FDR: {np.mean(batch_fdrs):.3f} ± {np.std(batch_fdrs):.3f}")
print(f"Streaming FDR: {streaming_fdr:.3f}")
```