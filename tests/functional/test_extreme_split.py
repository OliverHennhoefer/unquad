import unittest
from online_fdr import BatchStoreyBH

from pyod.models.iforest import IForest

from unquad.strategy.bootstrap import Bootstrap
from unquad.utils.data.load import load_shuttle, load_fraud
from unquad.utils.data.batch_generator import create_batch_generator
from unquad.estimation.extreme_conformal import EVTConformalDetector
from unquad.strategy.split import Split
from unquad.utils.stat.metrics import false_discovery_rate, statistical_power
from online_fdr.batching.bh import BatchBH
from online_fdr.batching.prds import BatchPRDS


class TestCaseExtremeSplit(unittest.TestCase):
    def test_extreme_split_batch_bh_shuttle(self):

        df = load_shuttle()
        x_train, batch_gen = create_batch_generator(
            df, train_size=0.6, batch_size=300, anomaly_proportion=0.13, random_state=42
        )

        evt_detector = EVTConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Split(calib_size=1000),
            evt_threshold_method="percentile",
            evt_threshold_value=0.95,
            evt_min_tail_size=25,
            silent=True,
        )

        evt_detector.fit(x_train)

        batch_bh = BatchBH(alpha=0.1)

        label = []
        decision = []

        for batch_id, (x_batch, y_batch) in enumerate(batch_gen.generate(n_batches=10)):

            p_values = evt_detector.predict(x_batch)
            decisions = batch_bh.test_batch(p_values)

            label.extend(y_batch.tolist())
            decision.extend(decisions)

        self.assertEqual(statistical_power(label, decision), 0.946)
        self.assertEqual(false_discovery_rate(label, decision), 0.068)

    def test_extreme_split_batch_st_bh_shuttle(self):

        df = load_shuttle()
        x_train, batch_gen = create_batch_generator(
            df, train_size=0.6, batch_size=300, anomaly_proportion=0.01, random_state=42
        )

        evt_detector = EVTConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Split(calib_size=1000),
            evt_threshold_method="percentile",
            evt_threshold_value=0.95,
            evt_min_tail_size=25,
            silent=True,
        )

        evt_detector.fit(x_train)

        batch_st_bh = BatchStoreyBH(alpha=0.1, lambda_=0.1)

        label = []
        decision = []

        for batch_id, (x_batch, y_batch) in enumerate(batch_gen.generate(n_batches=10)):

            p_values = evt_detector.predict(x_batch)
            decisions = batch_st_bh.test_batch(p_values.tolist())

            label.extend(y_batch.tolist())
            decision.extend(decisions)

        self.assertEqual(statistical_power(label, decision), 0.8)
        self.assertEqual(false_discovery_rate(label, decision), 0.04)

    def test_extreme_split_batch_st_bh_single_anomaly_batch_shuttle(self):

        df = load_shuttle()
        x_train, batch_gen = create_batch_generator(
            df,
            train_size=0.6,
            batch_size=1000,
            anomaly_proportion=0.001,
            random_state=42,
        )

        evt_detector = EVTConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Split(calib_size=1000),
            evt_threshold_method="percentile",
            evt_threshold_value=0.95,
            evt_min_tail_size=25,
            silent=True,
        )

        evt_detector.fit(x_train)

        batch_st_bh = BatchStoreyBH(alpha=0.1, lambda_=0.05)

        label = []
        decision = []

        for batch_id, (x_batch, y_batch) in enumerate(batch_gen.generate(n_batches=10)):

            p_values = evt_detector.predict(x_batch)
            decisions = batch_st_bh.test_batch(p_values.tolist())

            label.extend(y_batch.tolist())
            decision.extend(decisions)

        self.assertEqual(statistical_power(label, decision), 0.4)
        self.assertEqual(false_discovery_rate(label, decision), 0.0)

    def test_extreme_split_batch_st_bh_single_anomaly_batch_musk(self):

        df = load_fraud()
        x_train, batch_gen = create_batch_generator(
            df, train_size=0.6, batch_size=100, anomaly_proportion=0.01, random_state=42
        )

        evt_detector = EVTConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_calib=5_000, resampling_ratio=0.995),
            evt_threshold_method="mean_excess",
            evt_threshold_value=0.95,
            evt_min_tail_size=25,
            silent=True,
        )

        evt_detector.fit(x_train)

        batch_st_bh = BatchPRDS(alpha=0.2)

        label = []
        decision = []

        for batch_id, (x_batch, y_batch) in enumerate(batch_gen.generate(n_batches=10)):

            p_values = evt_detector.predict(x_batch)
            decisions = batch_st_bh.test_batch(p_values.tolist())

            label.extend(y_batch.tolist())
            decision.extend(decisions)

        self.assertEqual(statistical_power(label, decision), 0.1)
        self.assertEqual(false_discovery_rate(label, decision), 0.0)


if __name__ == "__main__":
    unittest.main()
