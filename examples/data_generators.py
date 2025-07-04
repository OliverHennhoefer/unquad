"""Example demonstrating different data providing methods in unquad.utils.data."""

import numpy as np
import pandas as pd
from unquad.utils.data.load import load_shuttle, load_breast, get_memory_cache_info
from unquad.utils.data.batch_generator import BatchGenerator, create_batch_generator
from unquad.utils.data.online_generator import OnlineGenerator


def demonstrate_data_loading():
    """Demonstrate dataset loading capabilities."""
    print("=== Dataset Loading Examples ===")

    # Load complete dataset
    df_shuttle = load_shuttle()
    print(
        f"Shuttle dataset: {df_shuttle.shape} (features: {df_shuttle.columns.tolist()[:5]}...)"
    )
    print(f"Class distribution: {df_shuttle['Class'].value_counts().to_dict()}")

    # Load with experimental setup
    x_train, x_test, y_test = load_shuttle(setup=True)
    print(
        f"Setup split - Train: {x_train.shape}, Test: {x_test.shape}, Test anomalies: {y_test.sum()}"
    )

    # Show memory cache info
    cache_info = get_memory_cache_info()
    print(
        f"Memory cache: {len(cache_info['datasets'])} datasets, {cache_info['total_size_mb']} MB"
    )
    print()


def demonstrate_batch_generation():
    """Demonstrate batch generation for evaluation."""
    print("=== Batch Generation Examples ===")

    # Load and prepare data
    x_train, x_test, y_test = load_breast(setup=True)
    x_normal = x_test[y_test == 0]
    x_anomaly = x_test[y_test == 1]

    print(f"Test data - Normal: {len(x_normal)}, Anomaly: {len(x_anomaly)}")

    # Create batch generator with 15% anomalies
    batch_gen = BatchGenerator(
        x_normal=x_normal,
        x_anomaly=x_anomaly,
        batch_size=50,
        anomaly_proportion=0.15,
        random_state=42,
    )

    print(f"Batch generator: {batch_gen}")
    print(f"Max theoretical batches: {batch_gen.max_batches}")

    # Generate sample batches
    total_instances = 0
    total_anomalies = 0

    for i, (x_batch, y_batch) in enumerate(batch_gen.generate(n_batches=3)):
        batch_anomalies = y_batch.sum()
        total_instances += len(x_batch)
        total_anomalies += batch_anomalies
        print(
            f"Batch {i+1}: {x_batch.shape}, Anomalies: {batch_anomalies} ({batch_anomalies/len(x_batch)*100:.1f}%)"
        )

    print(
        f"Total: {total_instances} instances, {total_anomalies} anomalies ({total_anomalies/total_instances*100:.1f}%)"
    )
    print()


def demonstrate_online_streaming():
    """Demonstrate online streaming generation."""
    print("=== Online Streaming Examples ===")

    # Use convenience function to create generator
    df_shuttle = load_shuttle()
    x_train, batch_gen = create_batch_generator(
        df_shuttle,
        train_size=0.6,
        batch_size=100,
        anomaly_proportion=0.05,
        random_state=42,
    )

    # Extract data for online generator
    x_normal = batch_gen.x_normal
    x_anomaly = batch_gen.x_anomaly

    print(f"Training data: {x_train.shape}")
    print(f"Streaming data - Normal: {len(x_normal)}, Anomaly: {len(x_anomaly)}")

    # Create online generator with exactly 2% anomalies
    online_gen = OnlineGenerator(
        x_normal=x_normal,
        x_anomaly=x_anomaly,
        anomaly_proportion=0.02,
        pool_size=500,
        random_state=42,
    )

    print(f"Online generator: {online_gen}")

    # Demonstrate exact anomaly count prediction
    n_instances = 100
    predicted_anomalies = online_gen.get_exact_anomaly_count(n_instances)
    print(f"Predicted anomalies in next {n_instances} instances: {predicted_anomalies}")

    # Generate stream and verify
    actual_anomalies = 0
    feature_means = []

    for i, (x_instance, y_label) in enumerate(
        online_gen.generate_stream(n_instances=n_instances)
    ):
        actual_anomalies += y_label
        feature_means.append(x_instance.iloc[0].mean())

        if i < 10:  # Show first 10 instances
            print(
                f"Instance {i+1}: shape={x_instance.shape}, label={y_label}, mean_features={feature_means[-1]:.3f}"
            )
        elif i == 10:
            print("...")

    print(
        f"Actual anomalies: {actual_anomalies} (matches prediction: {actual_anomalies == predicted_anomalies})"
    )
    print()


def demonstrate_advanced_scenarios():
    """Demonstrate advanced usage scenarios."""
    print("=== Advanced Scenarios ===")

    # Different batch sizes and anomaly proportions
    df = load_shuttle()
    scenarios = [
        (50, 0.1, "Small batches, 10% anomalies"),
        (200, 0.05, "Large batches, 5% anomalies"),
        (100, 5, "Fixed 5 anomalies per batch"),
    ]

    for batch_size, anomaly_prop, description in scenarios:
        try:
            x_train, batch_gen = create_batch_generator(
                df,
                batch_size=batch_size,
                anomaly_proportion=anomaly_prop,
                random_state=42,
            )

            # Generate one batch to test
            x_batch, y_batch = next(batch_gen.generate(n_batches=1))
            anomaly_count = y_batch.sum()

            print(
                f"{description}: {anomaly_count} anomalies in batch of {len(x_batch)}"
            )

        except Exception as e:
            print(f"{description}: Error - {e}")

    print()


def main():
    """Run all demonstrations."""
    print("Data Generation Examples")
    print("=" * 50)

    demonstrate_data_loading()
    demonstrate_batch_generation()
    demonstrate_online_streaming()
    demonstrate_advanced_scenarios()


if __name__ == "__main__":
    main()
