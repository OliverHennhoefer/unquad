"""Tests for classical and randomized empirical conformal p-values."""

import numpy as np
import pytest

from nonconform.enums import TieBreakMode
from nonconform.scoring import Empirical, calculate_p_val, calculate_weighted_p_val


@pytest.mark.parametrize(
    ("test_scores", "calib_scores", "expected"),
    [
        ([5.0], [1.0, 2.0, 3.0, 4.0], [1 / 5]),
        ([3.0], [1.0, 2.0, 4.0, 5.0], [3 / 5]),
        ([10.0], [1.0, 2.0, 3.0], [1 / 4]),
        ([0.5], [1.0, 2.0, 3.0, 4.0, 5.0], [1.0]),
        ([3.0], [2.0], [1 / 2]),
        ([3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0], [1.0, 1.0, 1.0]),
        ([1.0, 3.0, 5.0], [2.0, 4.0], [1.0, 2 / 3, 1 / 3]),
        ([1e10, -1e10], [0.0, 1.0, -1.0], [1 / 4, 1.0]),
        ([1.0000001], [1.0, 1.0000002, 1.0000003], [3 / 4]),
        (np.array([1, 2, 3]), np.array([0, 1, 2, 3, 4]), [5 / 6, 4 / 6, 3 / 6]),
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([0.5, 1.5, 2.5], dtype=np.float32),
            [3 / 4, 2 / 4, 1 / 4],
        ),
    ],
    ids=[
        "above-all",
        "median",
        "conservative-plus-one",
        "below-all",
        "single-calibration-score",
        "all-tied",
        "batch",
        "extreme-values",
        "close-values",
        "integer-scores",
        "float32-scores",
    ],
)
def test_classical_p_values_count_calibration_scores_at_or_above(
    test_scores, calib_scores, expected
):
    test_scores, calib_scores = np.asarray(test_scores), np.asarray(calib_scores)

    p_values = calculate_p_val(test_scores, calib_scores, tie_break="classical")

    np.testing.assert_allclose(p_values, expected)
    np.testing.assert_allclose(
        Empirical().compute_p_values(test_scores, calib_scores), expected
    )


@pytest.mark.parametrize(
    ("test_scores", "calib_scores", "test_weights", "calib_weights", "expected"),
    [
        ([5.0], [1.0, 4.0, 6.0], [1.0], [0.5, 0.5, 2.0], [3 / 4]),
        ([2.0], [1.0, 2.0, 2.0, 3.0], [5.0], [1.0, 2.0, 3.0, 4.0], [14 / 15]),
        ([2.5], [1.0, 2.0, 3.0, 4.0], [2.0], [1.0, 2.0, 3.0, 4.0], [9 / 12]),
        ([5.0], [1.0, 2.0, 3.0], [0.0], [1.0, 1.0, 1.0], [0.0]),
        ([5.0], [1.0, 6.0], [1e6], [1e6, 1e6], [2 / 3]),
        (
            [3.0, 5.0],
            [1.0, 2.0, 4.0, 6.0],
            [0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1 / 2, 1 / 3],
        ),
    ],
    ids=[
        "known-weights",
        "calibration-ties-included",
        "strict-formula-without-ties",
        "zero-test-weight",
        "large-weights",
        "mixed-zero-weights",
    ],
)
def test_classical_weighted_p_values_use_mass_at_or_above(
    test_scores, calib_scores, test_weights, calib_weights, expected
):
    p_values = calculate_weighted_p_val(
        np.array(test_scores),
        np.array(calib_scores),
        np.array(test_weights),
        np.array(calib_weights),
        tie_break="classical",
    )

    np.testing.assert_allclose(p_values, expected)


@pytest.mark.parametrize(
    ("test_scores", "calib_scores"),
    [([3.0, 5.0], [1.0, 2.0, 4.0, 6.0]), ([2.0, 3.0, 2.0], [1.0, 2.0, 2.0, 3.0])],
    ids=["without-ties", "with-ties"],
)
def test_uniform_weights_reproduce_unweighted_classical_p_values(
    test_scores, calib_scores
):
    test_scores, calib_scores = np.array(test_scores), np.array(calib_scores)

    weighted = calculate_weighted_p_val(
        test_scores,
        calib_scores,
        np.ones(len(test_scores)),
        np.ones(len(calib_scores)),
        tie_break="classical",
    )

    np.testing.assert_allclose(
        weighted, calculate_p_val(test_scores, calib_scores, tie_break="classical")
    )


@pytest.mark.parametrize("tie_break", ["classical", "randomized"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_p_values_match_test_shape_and_lie_in_unit_interval(
    sample_scores, sample_weights, tie_break, weighted
):
    test_scores, calib_scores = sample_scores(n_test=1000, n_calib=5000)
    if weighted:
        test_weights, calib_weights = sample_weights(n_test=1000, n_calib=5000)
        p_values = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights, tie_break=tie_break
        )
    else:
        p_values = calculate_p_val(test_scores, calib_scores, tie_break=tie_break)

    assert p_values.shape == test_scores.shape
    assert np.all((p_values >= 0.0) & (p_values <= 1.0))
    if tie_break == "classical":
        assert np.all(p_values > 0.0)


@pytest.mark.parametrize(
    ("test_scores", "calib_scores", "weights", "seed", "expected_from_uniform"),
    [
        ([3.0], [3.0, 3.0, 3.0, 3.0], None, 42, lambda u: u),
        ([2.5], [1.0, 2.0, 3.0, 4.0], None, 42, lambda u: (2.0 + u) / 5.0),
        (
            [2.0],
            [1.0, 2.0, 2.0, 3.0],
            ([5.0], [1.0, 2.0, 3.0, 4.0]),
            123,
            lambda u: (4.0 + (2.0 + 3.0 + 5.0) * u) / 15.0,
        ),
        ([5.0] * 20, [5.0] * 19, ([1.0] * 20, [1.0] * 19), 456, lambda u: u),
    ],
    ids=["all-tied", "no-ties", "weighted-partial-ties", "weighted-all-tied"],
)
def test_randomized_p_values_interpolate_tied_and_own_mass(
    test_scores, calib_scores, weights, seed, expected_from_uniform
):
    uniforms = np.random.default_rng(seed).uniform(size=len(test_scores))
    rng = np.random.default_rng(seed)
    if weights is None:
        p_values = calculate_p_val(
            np.array(test_scores), np.array(calib_scores), "randomized", rng
        )
    else:
        p_values = calculate_weighted_p_val(
            np.array(test_scores),
            np.array(calib_scores),
            np.array(weights[0]),
            np.array(weights[1]),
            tie_break="randomized",
            rng=rng,
        )

    np.testing.assert_allclose(p_values, expected_from_uniform(uniforms))


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("tie_break", ["randomized", TieBreakMode.RANDOMIZED])
def test_seeded_randomized_empirical_matches_seeded_formula(weighted, tie_break):
    test_scores = np.array([3.0, 5.0])
    calib_scores = np.array([1.0, 3.0, 5.0, 7.0])
    calib_weights, test_weights = np.ones(4), np.array([1.0, 2.0])
    estimation = Empirical(tie_break=tie_break)
    estimation.set_seed(42)

    if weighted:
        p_values = estimation.compute_p_values(
            test_scores, calib_scores, (calib_weights, test_weights)
        )
        expected = calculate_weighted_p_val(
            test_scores,
            calib_scores,
            test_weights,
            calib_weights,
            tie_break="randomized",
            rng=np.random.default_rng(42),
        )
    else:
        p_values = estimation.compute_p_values(test_scores, calib_scores)
        expected = calculate_p_val(
            test_scores, calib_scores, "randomized", np.random.default_rng(42)
        )

    np.testing.assert_array_equal(p_values, expected)


def test_invalid_tie_break_is_rejected():
    with pytest.raises(ValueError, match="tie_break must be one of"):
        calculate_p_val(np.array([1.0]), np.array([0.0, 2.0]), tie_break="invalid")
    with pytest.raises(ValueError, match="tie_break must be one of"):
        calculate_weighted_p_val(
            np.array([1.0]),
            np.array([0.0, 2.0]),
            np.array([1.0]),
            np.array([1.0, 1.0]),
            tie_break="invalid",
        )


@pytest.mark.parametrize(
    ("arrays", "match"),
    [
        (
            ([1.0, 2.0, 3.0], [0.5, 1.5, 2.5], [1.0, 1.0], [1.0, 1.0, 1.0]),
            "scores and test_weights must have the same length",
        ),
        (
            ([1.0, 2.0], [0.5, 1.5, 2.5], [1.0, 1.0], [1.0, 1.0]),
            "calibration_set and calib_weights must have the same length",
        ),
        (
            ([5.0], [1.0, 6.0, 7.0], [1.0], [0.0, 0.0, 0.0]),
            "calib_weights must sum to a positive value",
        ),
        (
            ([[1.0, 2.0], [3.0, 4.0]], [0.5, 1.5], [1.0, 1.0], [1.0, 1.0]),
            "must be numeric",
        ),
    ],
    ids=[
        "test-weight-length",
        "calibration-weight-length",
        "zero-calibration-mass",
        "matrix-scores",
    ],
)
def test_invalid_weighted_inputs_are_rejected(arrays, match):
    with pytest.raises(ValueError, match=match):
        calculate_weighted_p_val(*(np.array(values) for values in arrays))


@pytest.mark.parametrize("name", ["test_weights", "calib_weights"])
@pytest.mark.parametrize(
    ("bad_value", "match"),
    [
        (-0.1, "must be non-negative"),
        (np.nan, "must be finite"),
        (np.inf, "must be finite"),
        (-np.inf, "must be finite"),
    ],
)
def test_negative_or_non_finite_weights_are_rejected(name, bad_value, match):
    weights = {"test_weights": np.ones(10), "calib_weights": np.ones(50)}
    weights[name][0] = bad_value

    with pytest.raises(ValueError, match=f"{name} {match}"):
        calculate_weighted_p_val(
            np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 50), **weights
        )


def test_empty_weighted_test_batch_returns_empty_p_values():
    p_values = calculate_weighted_p_val(
        np.array([]), np.array([1.0, 2.0]), np.array([]), np.array([1.0, 1.0])
    )

    assert p_values.shape == (0,)
