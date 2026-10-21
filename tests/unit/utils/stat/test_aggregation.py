import numpy as np
import pytest

import nonconform._internal.math_utils as math_utils
from nonconform.metrics import aggregate


@pytest.mark.parametrize(
    ("method", "reducer", "known_scores", "expected"),
    [
        ("mean", np.mean, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [2.5, 3.5, 4.5]),
        ("median", np.median, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [3.0, 4.0]),
        ("minimum", np.min, [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], [1.0, 2.0, 3.0]),
        ("maximum", np.max, [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], [4.0, 5.0, 6.0]),
    ],
)
def test_aggregation_reduces_over_models(
    sample_2d_scores, method, reducer, known_scores, expected
):
    scores = sample_2d_scores(n_models=5, n_samples=10)

    result = aggregate(method, scores)

    assert result.shape == (10,)
    np.testing.assert_array_almost_equal(result, reducer(scores, axis=0))
    np.testing.assert_array_almost_equal(
        aggregate(method, np.array(known_scores)), expected
    )
    np.testing.assert_array_equal(aggregate(f" {method.upper()} ", scores), result)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([[1.0, 2.0, 3.0]], [1.0, 2.0, 3.0]),
        ([[1.0], [2.0], [3.0]], [2.0]),
        ([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]], [5.0, 5.0, 5.0]),
    ],
    ids=["single-model", "single-sample", "constant"],
)
def test_mean_aggregation_handles_degenerate_shapes(scores, expected):
    np.testing.assert_array_equal(aggregate("mean", np.array(scores)), expected)


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [(np.float32, np.float32), (np.float64, np.float64), (np.int64, np.float64)],
)
def test_mean_aggregation_result_dtype(dtype, expected_dtype):
    result = aggregate("mean", np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype))

    assert result.dtype == expected_dtype


@pytest.mark.parametrize(
    ("method", "error", "match"),
    [
        ("invalid", ValueError, "Unsupported aggregation method"),
        (1, TypeError, "aggregation method must be a string"),
    ],
)
def test_invalid_aggregation_method_raises(method, error, match):
    with pytest.raises(error, match=match):
        aggregate(method, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_normalize_bootstrap_aggregation_method_rejects_minimum() -> None:
    with pytest.raises(ValueError, match="Unsupported bootstrap aggregation method"):
        math_utils.normalize_bootstrap_aggregation_method("minimum")


def test_aggregate_assert_never_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    def _invalid_normalizer(method: str) -> str:
        return "invalid"  # pragma: no cover - runtime-only guard injection

    monkeypatch.setattr(math_utils, "normalize_aggregation_method", _invalid_normalizer)
    with pytest.raises(AssertionError):
        math_utils.aggregate("mean", np.array([[1.0, 2.0]]))
