"""FDR and power regression checks for conformal selection on benchmark data.

Weighted cases use Weighted Conformalized Selection (WCS), which is more
conservative than standard BH: it accounts for covariate shift via weighted
conformal p-values, uses auxiliary p-values and pruning for FDR control, and
needs sufficient calibration data for good resolution. Probabilistic (KDE-based)
estimation gives continuous p-values that often improve power while
intentionally dropping the finite-sample guarantee.
"""

import pytest

pytest.importorskip("pyod", reason="pyod not installed")
pytest.importorskip("oddball", reason="oddball not installed")

from oddball import Dataset, load
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.knn import KNN

from nonconform import (
    ConformalDetector,
    CrossValidation,
    Empirical,
    JackknifeBootstrap,
    Probabilistic,
    Split,
    logistic_weight_estimator,
)
from nonconform.enums import ConformalMode
from nonconform.metrics import false_discovery_rate, statistical_power

METRIC_ATOL = 5e-3

# Benchmark: dataset, detector factory, strategy factory, and FDR target alpha.
BENCHMARKS = {
    "split": (Dataset.SHUTTLE, HBOS, lambda: Split(n_calib=1_000), 0.2),
    "jackknife": (
        Dataset.WBC,
        IForest,
        lambda: CrossValidation.jackknife(mode=ConformalMode.SINGLE_MODEL),
        0.25,
    ),
    "jackknife_bootstrap": (
        Dataset.MAMMOGRAPHY,
        lambda: KNN(method="mean", n_neighbors=7),
        lambda: JackknifeBootstrap(n_bootstraps=100),
        0.1,
    ),
    "cv": (Dataset.FRAUD, INNE, lambda: CrossValidation(k=5), 0.2),
}

ESTIMATIONS = {
    "empirical": lambda: Empirical(tie_break="classical"),
    "randomized": lambda: Empirical(tie_break="randomized"),
    "probabilistic": lambda: Probabilistic(n_trials=10),
}

CASES = [
    # (benchmark, estimation, weighted, expected FDR, expected power)
    ("split", "empirical", False, 0.203389830508, 0.94),
    ("jackknife", "empirical", False, 0.0, 1.0),
    ("jackknife_bootstrap", "empirical", False, 0.076923076923, 0.24),
    ("cv", "empirical", False, 0.175925925926, 0.89),
    ("split", "probabilistic", False, 0.160714285714, 0.94),
    ("jackknife", "probabilistic", False, 0.0, 1.0),
    ("jackknife_bootstrap", "probabilistic", False, 0.071428571429, 0.26),
    # HBOS produces heavily tied scores. Counting tied calibration mass makes
    # the classical p-values conservative, giving no discoveries here; the
    # randomized p-values (Jin & Candes 2023) smooth the ties.
    ("split", "empirical", True, 0.0, 0.0),
    ("split", "randomized", True, 0.0, 0.0),
    # The small calibration set (106 samples) limits WCS resolution.
    ("jackknife", "empirical", True, 0.0, 0.0),
    ("jackknife_bootstrap", "empirical", True, 0.1, 0.18),
    ("cv", "empirical", True, 0.205357142857, 0.89),
    ("split", "probabilistic", True, 0.104761904762, 0.94),
    ("jackknife", "probabilistic", True, 0.0, 0.333333333333),
    ("jackknife_bootstrap", "probabilistic", True, 0.083333333333, 0.22),
]


@pytest.mark.parametrize(
    ("benchmark", "estimation", "weighted", "expected_fdr", "expected_power"),
    CASES,
    ids=[
        f"{'weighted' if weighted else 'standard'}-{estimation}-{benchmark}"
        for benchmark, estimation, weighted, *_ in CASES
    ],
)
def test_selection_matches_reference_fdr_and_power(
    benchmark, estimation, weighted, expected_fdr, expected_power
):
    dataset, make_detector, make_strategy, alpha = BENCHMARKS[benchmark]
    x_train, x_test, y_test = load(dataset, setup=True, seed=1)

    ce = ConformalDetector(
        detector=make_detector(),
        strategy=make_strategy(),
        estimation=ESTIMATIONS[estimation](),
        weight_estimator=logistic_weight_estimator() if weighted else None,
        seed=1,
    )

    ce.fit(x_train)
    decisions = ce.select(x_test, alpha=alpha)
    assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
        expected_fdr, rel=0.0, abs=METRIC_ATOL
    )
    assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
        expected_power, rel=0.0, abs=METRIC_ATOL
    )
