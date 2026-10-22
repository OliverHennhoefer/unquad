import numpy as np
import pytest

from nonconform.metrics import false_discovery_rate, statistical_power


@pytest.mark.parametrize(
    ("y", "y_hat", "expected_fdp", "expected_power"),
    [
        ([0, 0, 1, 1, 1], [0, 0, 1, 1, 1], 0.0, 1.0),
        ([0, 0, 1, 1, 1], [0, 0, 0, 0, 0], 0.0, 0.0),
        ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], 1.0, 0.0),
        ([0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0),
        ([1, 1, 1, 1], [1, 1, 1, 1], 0.0, 1.0),
        ([1], [1], 0.0, 1.0),
        ([0, 0, 1, 1], [1, 0, 1, 0], 0.5, 0.5),
        ([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0], 0.5, 2 / 3),
        ([1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], 0.5, 1 / 3),
        ([1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0], 0.0, 0.5),
        ([1, 1, 0, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 0], 0.5, 0.5),
        ([False, True, True, False], [False, True, False, True], 0.5, 0.5),
    ],
    ids=[
        "perfect",
        "no-predictions",
        "only-false-positives",
        "no-positives-or-predictions",
        "all-positive",
        "single-element",
        "mixed",
        "two-thirds-recall",
        "one-third-recall",
        "partial-detection",
        "balanced-errors",
        "boolean-arrays",
    ],
)
def test_realized_fdp_and_power_match_confusion_counts(
    y, y_hat, expected_fdp, expected_power
):
    y, y_hat = np.asarray(y), np.asarray(y_hat)

    fdp = false_discovery_rate(y, y_hat)
    power = statistical_power(y, y_hat)

    assert isinstance(fdp, float)
    assert isinstance(power, float)
    assert fdp == pytest.approx(expected_fdp)
    assert power == pytest.approx(expected_power)
