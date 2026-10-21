"""Tests for Weighted Conformalized Selection (WCS)."""

import numpy as np
import pytest

from nonconform._internal.wcs import (
    _calib_weight_mass_strictly_above,
    _compute_r_star,
    _rejection_set_size_for_instance,
    _WCSInputs,
    run,
)
from nonconform.enums import Pruning
from nonconform.fdr import (
    weighted_false_discovery_control,
    weighted_false_discovery_control_from_arrays,
)
from nonconform.scoring import calculate_weighted_p_val


def _bh_rejection_count(p_values: np.ndarray, alpha: float) -> int:
    m = len(p_values)
    if m == 0:
        return 0
    sorted_p = np.sort(p_values)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = np.nonzero(sorted_p <= thresholds)[0]
    return 0 if len(below) == 0 else int(below[-1] + 1)


def _reference_wcs_deterministic(
    *,
    p_values: np.ndarray,
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    test_weights: np.ndarray,
    calib_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    m = len(test_scores)
    sum_calib_weight = float(np.sum(calib_weights))
    r_sizes = np.zeros(m, dtype=float)

    for j in range(m):
        aux_p = np.zeros(m, dtype=float)
        denominator = sum_calib_weight + test_weights[j]
        for k in range(m):
            if k == j:
                continue
            calib_mass = float(np.sum(calib_weights[calib_scores > test_scores[k]]))
            aux_p[k] = (
                calib_mass + test_weights[k] * float(test_scores[k] > test_scores[j])
            ) / denominator
        r_sizes[j] = _bh_rejection_count(aux_p, alpha)

    thresholds = alpha * r_sizes / m
    first_sel_idx = np.flatnonzero(p_values <= thresholds)
    if len(first_sel_idx) == 0:
        return np.zeros(m, dtype=bool)

    sizes_sel = r_sizes[first_sel_idx]
    sorted_sizes = np.sort(sizes_sel)
    r_star = 0
    for k in range(len(sorted_sizes), 0, -1):
        if sorted_sizes[k - 1] <= k:
            r_star = k
            break
    if r_star == 0:
        return np.zeros(m, dtype=bool)

    final_idx = np.sort(first_sel_idx[sizes_sel <= r_star])
    mask = np.zeros(m, dtype=bool)
    mask[final_idx] = True
    return mask


def _classical_wcs_arrays(test_scores, calib_scores, test_weights, calib_weights):
    """Return explicit WCS arrays with classical weighted p-values."""
    p_values = calculate_weighted_p_val(
        scores=test_scores,
        calibration_set=calib_scores,
        test_weights=test_weights,
        calib_weights=calib_weights,
        tie_break="classical",
    )
    return {
        "p_values": p_values,
        "test_scores": test_scores,
        "calib_scores": calib_scores,
        "test_weights": test_weights,
        "calib_weights": calib_weights,
    }


def _signal_arrays():
    """Return weighted arrays whose strong anomalies survive the first WCS stage."""
    rng = np.random.default_rng(0)
    test_scores = np.concatenate([rng.normal(size=20), rng.normal(size=10) + 4.0])
    return _classical_wcs_arrays(
        test_scores,
        rng.normal(size=200),
        rng.uniform(0.5, 1.5, size=30),
        rng.uniform(0.5, 1.5, size=200),
    )


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.15, 0.20])
@pytest.mark.parametrize("n_calib,n_test", [(40, 12), (80, 20), (100, 30)])
def test_weighted_fdr_deterministic_matches_reference_formulation(
    alpha: float,
    n_calib: int,
    n_test: int,
) -> None:
    rng = np.random.default_rng(17)
    calib_scores = rng.normal(size=n_calib)
    test_scores = rng.normal(size=n_test)
    calib_scores += np.linspace(0, 1e-4, len(calib_scores))
    test_scores += np.linspace(0, 1e-4, len(test_scores))

    calib_weights = rng.uniform(0.4, 1.8, size=len(calib_scores))
    test_weights = rng.uniform(0.4, 1.8, size=len(test_scores))
    arrays = _classical_wcs_arrays(
        test_scores, calib_scores, test_weights, calib_weights
    )

    expected = _reference_wcs_deterministic(**arrays, alpha=alpha)
    actual = weighted_false_discovery_control_from_arrays(
        **arrays, alpha=alpha, pruning=Pruning.DETERMINISTIC
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("n_test", [5, 10, 20, 30, 50])
def test_result_selection_returns_one_boolean_per_test_point(conformal_result, n_test):
    discoveries = weighted_false_discovery_control(
        result=conformal_result(n_test=n_test, n_calib=100), alpha=0.1
    )

    assert isinstance(discoveries, np.ndarray)
    assert discoveries.dtype == bool
    assert discoveries.shape == (n_test,)


def test_result_selection_is_monotone_in_alpha(conformal_result):
    result = conformal_result(n_test=20, n_calib=100, seed=42)

    masks = {
        alpha: weighted_false_discovery_control(result=result, alpha=alpha)
        for alpha in (0.001, 0.01, 0.05, 0.10, 0.20, 0.999)
    }

    assert all(mask.shape == (20,) for mask in masks.values())
    assert masks[0.01].sum() <= masks[0.10].sum() <= masks[0.20].sum()


def test_high_p_values_are_never_selected(conformal_result):
    result = conformal_result(n_test=20, n_calib=100, seed=99)
    result.p_values = np.full(20, 0.9)

    assert not weighted_false_discovery_control(result=result, alpha=0.01).any()


@pytest.mark.parametrize(
    "metadata",
    [
        {"kde_support": np.array([0.0, 1.0, 2.0])},
        {
            "nonconform": {
                "strategy": "Split",
                "estimation": "Empirical",
                "weighted": True,
            },
            "tie_break": "classical",
        },
        {
            "nonconform": {
                "strategy": "Split",
                "estimation": "CustomEstimation",
                "weighted": True,
            },
        },
    ],
    ids=["unrelated-metadata", "classical-empirical", "custom-estimator"],
)
def test_result_selection_accepts_non_kde_metadata(conformal_result, metadata):
    result = conformal_result(n_test=5, n_calib=20)
    result.metadata = metadata

    assert weighted_false_discovery_control(result=result, alpha=0.1).shape == (5,)


def test_result_selection_requires_a_complete_weighted_result(conformal_result):
    with pytest.raises(ValueError, match="result must be a ConformalResult"):
        weighted_false_discovery_control(result=None, alpha=0.1)
    with pytest.raises(ValueError, match="missing required WCS fields: p_values"):
        weighted_false_discovery_control(
            result=conformal_result(include_p_values=False), alpha=0.1
        )
    with pytest.raises(TypeError):
        weighted_false_discovery_control(alpha=0.1)  # type: ignore[call-arg]


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_outside_open_unit_interval_is_rejected(conformal_result, alpha):
    with pytest.raises(ValueError, match="alpha must be in"):
        weighted_false_discovery_control(result=conformal_result(), alpha=alpha)


@pytest.mark.parametrize(
    ("make_arrays", "alpha", "expected_size"),
    [
        (
            lambda: {
                "p_values": np.array([0.01]),
                "test_scores": np.array([5.0]),
                "calib_scores": np.array([1.0, 2.0, 3.0]),
                "test_weights": np.array([1.0]),
                "calib_weights": np.array([1.0, 1.0, 1.0]),
            },
            0.05,
            1,
        ),
        (
            lambda: _classical_wcs_arrays(
                np.random.default_rng(42).standard_normal(10),
                np.random.default_rng(43).standard_normal(50),
                np.full(10, 1e6),
                np.full(50, 1e6),
            ),
            0.1,
            10,
        ),
    ],
    ids=["single-test-point", "large-weights"],
)
def test_array_selection_handles_small_batches_and_large_weights(
    make_arrays, alpha, expected_size
):
    discoveries = weighted_false_discovery_control_from_arrays(
        **make_arrays(), alpha=alpha
    )

    assert discoveries.shape == (expected_size,)


@pytest.mark.parametrize("compute_p_values", [False, True])
def test_array_selection_accepts_supplied_or_recomputed_p_values(
    sample_scores, sample_weights, sample_p_values, compute_p_values
):
    test_scores, calib_scores = sample_scores(n_test=15, n_calib=80)
    test_weights, calib_weights = sample_weights(n_test=15, n_calib=80)
    if compute_p_values:
        arrays = _classical_wcs_arrays(
            test_scores, calib_scores, test_weights, calib_weights
        )
    else:
        arrays = {
            "p_values": sample_p_values(n=15),
            "test_scores": test_scores,
            "calib_scores": calib_scores,
            "test_weights": test_weights,
            "calib_weights": calib_weights,
        }

    assert weighted_false_discovery_control_from_arrays(**arrays, alpha=0.1).shape == (
        15,
    )


def _valid_arrays():
    return {
        "p_values": np.array([0.2, 0.7]),
        "test_scores": np.array([1.0, 2.0]),
        "calib_scores": np.array([0.0, 1.0, 2.0]),
        "test_weights": np.ones(2),
        "calib_weights": np.ones(3),
    }


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"p_values": np.array([-0.1, 0.5])}, ValueError, r"within \[0, 1\]"),
        ({"p_values": np.array([1.1, 0.5])}, ValueError, r"within \[0, 1\]"),
        ({"p_values": np.array([np.nan, 0.5])}, ValueError, "must be finite"),
        ({"p_values": np.array([np.inf, 0.5])}, ValueError, "must be finite"),
        ({"p_values": np.array([-np.inf, 0.5])}, ValueError, "must be finite"),
        ({"p_values": np.array([[0.1], [0.2]])}, ValueError, "p_values must be a 1D"),
        ({"test_weights": np.ones(1)}, ValueError, "must have the same length"),
        ({"calib_weights": np.ones(2)}, ValueError, "must have the same length"),
        ({"test_weights": np.array([-1.0, 1.0])}, ValueError, "non-negative"),
        ({"calib_weights": np.array([1.0, np.nan, 1.0])}, ValueError, "finite"),
        ({"pruning": "deterministic"}, TypeError, "instance of Pruning"),
    ],
    ids=[
        "negative-p-value",
        "p-value-above-one",
        "nan-p-value",
        "inf-p-value",
        "negative-inf-p-value",
        "matrix-p-values",
        "test-length-mismatch",
        "calibration-length-mismatch",
        "negative-test-weight",
        "nan-calibration-weight",
        "pruning-string",
    ],
)
def test_invalid_array_inputs_are_rejected(overrides, error, match):
    kwargs = {**_valid_arrays(), "alpha": 0.1, **overrides}

    with pytest.raises(error, match=match):
        weighted_false_discovery_control_from_arrays(**kwargs)


def test_array_selection_requires_every_array():
    arrays = _valid_arrays()
    del arrays["test_scores"]

    with pytest.raises(TypeError):
        weighted_false_discovery_control_from_arrays(**arrays, alpha=0.1)


def test_all_tied_scores_contrast_classical_and_strict_primary_inputs():
    n_calib = 19
    n_test = 20
    arrays = _classical_wcs_arrays(
        np.full(n_test, 5.0), np.full(n_calib, 5.0), np.ones(n_test), np.ones(n_calib)
    )
    np.testing.assert_array_equal(arrays["p_values"], np.ones(n_test))

    # The explicit-array API consumes precomputed p-values without
    # provenance checks. Classical all-ties p-values equal one.
    for pruning in Pruning:
        classical_discoveries = weighted_false_discovery_control_from_arrays(
            **arrays, alpha=0.1, pruning=pruning, seed=42
        )
        assert not np.any(classical_discoveries)

    # A strict primary p-value is 1 / (n_calib + 1).
    # In this all-null example it makes the current array-level WCS path
    # select all test points, demonstrating why the classical formula must
    # retain tied calibration mass.
    strict_arrays = {**arrays, "p_values": np.full(n_test, 1.0 / (n_calib + 1))}
    strict_discoveries = weighted_false_discovery_control_from_arrays(
        **strict_arrays, alpha=0.1, pruning=Pruning.DETERMINISTIC
    )
    assert np.all(strict_discoveries)


def test_wcs_auxiliary_mass_is_strictly_above():
    calib_scores = np.array([1.0, 2.0, 2.0, 3.0])
    calib_weights = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([1.0, 2.0, 3.0])

    mass = _calib_weight_mass_strictly_above(calib_scores, calib_weights, targets)

    np.testing.assert_allclose(mass, [9.0, 4.0, 0.0])


@pytest.mark.parametrize("pruning", list(Pruning))
def test_pruning_is_reproducible_and_keeps_deterministic_selections(pruning):
    arrays = _signal_arrays()
    deterministic = weighted_false_discovery_control_from_arrays(
        **arrays, alpha=0.1, pruning=Pruning.DETERMINISTIC
    )

    masks = [
        weighted_false_discovery_control_from_arrays(
            **arrays, alpha=0.1, pruning=pruning, seed=777
        )
        for _ in range(3)
    ]

    assert deterministic.any()
    for mask in masks:
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, masks[0])
    # Uniform pruning metrics never exceed the deterministic rejection sizes.
    assert np.all(masks[0][deterministic])


def test_deterministic_pruning_ignores_seed():
    arrays = _signal_arrays()

    masks = [
        weighted_false_discovery_control_from_arrays(
            **arrays, alpha=0.1, pruning=Pruning.DETERMINISTIC, seed=seed
        )
        for seed in (None, 1, 42, 99, 1000)
    ]

    for mask in masks[1:]:
        np.testing.assert_array_equal(mask, masks[0])


def _kde_metadata(**overrides):
    metadata = {
        "eval_grid": np.array([0.0, 1.0, 2.0]),
        "cdf_values": np.array([0.0, 0.5, 1.0]),
        "total_weight": 10.0,
    }
    metadata.update(overrides)
    return {"kde": metadata}


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"kde": 1}, "must be a dictionary"),
        ({"kde": {"eval_grid": np.array([0.0, 1.0])}}, "missing keys"),
        (_kde_metadata(total_weight="not-a-number"), "finite positive float"),
        (
            _kde_metadata(eval_grid=np.array([0.0]), cdf_values=np.array([0.0])),
            "at least 2 points",
        ),
        (_kde_metadata(cdf_values=np.array([0.0, 1.0])), "must have equal length"),
        (_kde_metadata(eval_grid=np.array([0.0, 1.0, 1.0])), "strictly increasing"),
        (_kde_metadata(cdf_values=np.array([0.0, 0.5, 0.49])), "non-decreasing"),
        (_kde_metadata(cdf_values=np.array([0.0, 1.1, 1.2])), r"within \[0, 1\]"),
        (_kde_metadata(total_weight=0.0), "finite positive value"),
    ],
    ids=[
        "not-a-dictionary",
        "missing-keys",
        "non-numeric-total-weight",
        "single-point-grid",
        "length-mismatch",
        "non-increasing-grid",
        "decreasing-cdf",
        "cdf-out-of-bounds",
        "non-positive-total-weight",
    ],
)
def test_malformed_kde_metadata_is_rejected(conformal_result, metadata, match):
    result = conformal_result(n_test=5, n_calib=20)
    result.metadata = metadata

    with pytest.raises(ValueError, match=match):
        weighted_false_discovery_control(result=result, alpha=0.1)


def test_tiny_kde_cdf_jitter_is_tolerated(conformal_result):
    result = conformal_result(n_test=5, n_calib=20)
    result.metadata = _kde_metadata(cdf_values=np.array([0.0, 0.5, 0.5 - 5e-13]))

    assert weighted_false_discovery_control(result=result, alpha=0.1).shape == (5,)


def test_kde_selection_rejects_non_positive_total_calibration_weight():
    arrays = _valid_arrays()
    with pytest.raises(ValueError, match="positive finite total calibration weight"):
        run(
            **{**arrays, "calib_weights": np.zeros(3)},
            alpha=0.1,
            pruning=Pruning.DETERMINISTIC,
            seed=0,
            include_self_weight=False,
        )


def test_compute_r_star_empty_metrics_returns_zero() -> None:
    assert _compute_r_star(np.array([], dtype=float)) == 0


@pytest.mark.parametrize(
    ("include_self_weight", "sum_calib_weight", "match"),
    [
        (True, 2.0, "missing score-rank cache"),
        (False, 0.0, "positive finite effective calibration mass"),
    ],
)
def test_rejection_set_size_rejects_inconsistent_internal_state(
    include_self_weight, sum_calib_weight, match
):
    with pytest.raises(ValueError, match=match):
        _rejection_set_size_for_instance(
            index=0,
            inputs=_WCSInputs(**_valid_arrays()),
            sum_calib_weight=sum_calib_weight,
            bh_thresholds=np.array([0.05, 0.10]),
            calib_mass_strictly_above=np.array([1.0, 0.5]),
            scratch=np.empty(2, dtype=float),
            include_self_weight=include_self_weight,
            sorted_test_idx=None,
            lt_cutoffs=None,
        )
