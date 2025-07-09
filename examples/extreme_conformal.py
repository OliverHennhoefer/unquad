from online_fdr import BatchStoreyBH

from pyod.models.iforest import IForest
from unquad.estimation.extreme_conformal import EVTConformalDetector
from unquad.strategy.split import Split
from unquad.utils.data import create_batch_generator
from unquad.utils.data.load import load_shuttle
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power

if __name__ == "__main__":

    df = load_shuttle()

    x_train, batch_gen = create_batch_generator(
        df, train_size=0.6, batch_size=100, anomaly_proportion=0.01, random_state=42
    )

    # EVT-Enhanced Conformal Anomaly Detector
    ce = EVTConformalDetector(
        detector=IForest(behaviour="new"),
        strategy=Split(calib_size=5_000),
        evt_threshold_method="percentile",
        evt_threshold_value=0.95,
        evt_min_tail_size=10,
    )

    ce.fit(x_train)

    batch_fdr = BatchStoreyBH(alpha=0.2, lambda_=0.125)

    label = []
    decision = []

    for batch_id, (x_batch, y_batch) in enumerate(batch_gen.generate(n_batches=10)):
        p_values = ce.predict(x_batch)
        decisions = batch_fdr.test_batch(p_values.tolist())

        label.extend(y_batch.tolist())
        decision.extend(decisions)

    print(f"Empirical FDR: {false_discovery_rate(y=label, y_hat=decision)}")
    print(f"Empirical Power: {statistical_power(y=label, y_hat=decision)}")
