from nonconform.utils.data.generator import BatchGenerator
from nonconform.utils.data.load import load_shuttle


def test_batch_generation():
    """Test batch generation."""
    print("=== Batch Generation Test ===")

    # Proportional mode - fixed 10% anomalies per batch
    print("1. Proportional Mode (10% anomalies per batch):")
    batch_gen = BatchGenerator(
        load_data_func=load_shuttle,
        batch_size=100,
        anomaly_proportion=0.1,
        anomaly_mode="proportional",
        n_batches=3,
        train_size=0.5,
        random_state=42,
    )

    print(f"   Generator: {batch_gen}")
    print(f"   Training data: {batch_gen.get_training_data().shape}")

    for i, (x_batch, y_batch) in enumerate(batch_gen.generate()):
        anomaly_count = y_batch.sum()
        print(f"   Batch {i+1}: {x_batch.shape}, Anomalies: {anomaly_count}")


if __name__ == "__main__":
    test_batch_generation()
