import numpy as np
import pytest

from nonconform._internal.tuning import (
    IQR_TO_STD_FACTOR,
    MAX_BANDWIDTH_RATIO,
    SCOTT_CONSTANT,
    SILVERMAN_CONSTANT,
    _collect_heuristic_bandwidths,
    _scott_bandwidth,
    _sheather_jones_bandwidth,
    _silverman_bandwidth,
    compute_bandwidth_range,
)

BANDWIDTH_RULES = [_scott_bandwidth, _silverman_bandwidth, _sheather_jones_bandwidth]
RULE_IDS = ["scott", "silverman", "sheather-jones"]
NORMAL_DATA = np.random.default_rng(42).standard_normal(100)


def test_rule_of_thumb_bandwidths_match_reference_formulas():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = len(data)
    sigma = np.std(data, ddof=1)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)

    assert np.isclose(_scott_bandwidth(data), SCOTT_CONSTANT * sigma * n ** (-0.2))
    assert np.isclose(
        _silverman_bandwidth(data),
        SILVERMAN_CONSTANT * min(sigma, iqr / IQR_TO_STD_FACTOR) * n ** (-0.2),
    )


@pytest.mark.parametrize("rule", BANDWIDTH_RULES, ids=RULE_IDS)
@pytest.mark.parametrize(
    "data",
    [
        NORMAL_DATA,
        np.array([1.0, 2.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 100.0]),
        np.random.default_rng(42).standard_normal(50) * 1e10,
        np.random.default_rng(42).standard_normal(50) * 1e-10,
    ],
    ids=["normal", "two-values", "outlier", "large-values", "small-values"],
)
def test_bandwidth_rules_are_finite_and_positive(rule, data):
    bandwidth = rule(data)

    assert np.isfinite(bandwidth)
    assert bandwidth > 0


@pytest.mark.parametrize(
    "data", [np.full(100, 5.0), np.zeros(100)], ids=["constant", "zeros"]
)
def test_zero_spread_data_gives_finite_nonnegative_bandwidths(data):
    assert np.percentile(data, 75) - np.percentile(data, 25) == 0
    for rule in BANDWIDTH_RULES:
        bandwidth = rule(data)
        assert np.isfinite(bandwidth)
        assert bandwidth >= 0

    bw_min, bw_max = compute_bandwidth_range(data)

    assert np.isfinite(bw_min)
    assert np.isfinite(bw_max)
    assert 0 <= bw_min <= bw_max


def test_rule_of_thumb_bandwidths_scale_with_spread_and_sample_size():
    rng = np.random.default_rng(42)
    narrow, wide = rng.standard_normal(100), rng.standard_normal(100) * 10
    assert _scott_bandwidth(wide) > _scott_bandwidth(narrow)

    rng = np.random.default_rng(42)
    small, large = rng.standard_normal(10), rng.standard_normal(1000)
    assert _scott_bandwidth(small) > _scott_bandwidth(large)

    rng = np.random.default_rng(123)
    small, large = rng.standard_normal(10), rng.standard_normal(1000)
    assert _silverman_bandwidth(small) > _silverman_bandwidth(large)


def test_bandwidth_range_uses_robust_spread_and_scales_with_data():
    data = np.array([0.0, 10.0])
    q1, q99 = np.percentile(data, [1, 99])
    robust_range = q99 - q1
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    robust_std = iqr / IQR_TO_STD_FACTOR

    bw_min, bw_max = compute_bandwidth_range(data)

    assert np.isclose(bw_min, min(robust_range * 0.001, robust_std * 0.01))
    assert np.isclose(bw_max, max(robust_range * 0.5, robust_std * 2))
    assert (
        compute_bandwidth_range(np.array([0.0, 100.0]))[1]
        > compute_bandwidth_range(np.array([0.0, 1.0]))[1]
    )
    normal_range = compute_bandwidth_range(NORMAL_DATA)
    assert isinstance(normal_range, tuple)
    assert 0 < normal_range[0] < normal_range[1]


def test_bandwidth_range_caps_extreme_ratio():
    near_zero = np.linspace(0.0, 1e-6, 9800, dtype=float)
    high_tail = np.linspace(1000.0, 1000.1, 200, dtype=float)
    data = np.concatenate([near_zero, high_tail])

    bw_min, bw_max = compute_bandwidth_range(data)

    assert bw_min > 0.0
    assert bw_max > 0.0
    assert bw_max / bw_min <= MAX_BANDWIDTH_RATIO + 1e-12
    assert np.isclose(bw_max, bw_min * MAX_BANDWIDTH_RATIO)


@pytest.mark.parametrize(
    ("data", "bw_min", "bw_max"),
    [
        (NORMAL_DATA, 0.01, 10.0),
        (NORMAL_DATA, 0.001, 0.002),
        (NORMAL_DATA, 100.0, 200.0),
        (NORMAL_DATA, 0.0001, 0.001),
        (np.array([1.0, 1.0, 1.0, 2.0]), 0.01, 10.0),
    ],
    ids=["within-range", "deduplicated", "clipped-up", "clipped-down", "ties"],
)
def test_heuristic_bandwidths_are_distinct_and_clipped(data, bw_min, bw_max):
    heuristics = _collect_heuristic_bandwidths(data, bw_min, bw_max)

    assert heuristics
    assert all(np.isfinite(bw) and bw_min <= bw <= bw_max for bw in heuristics)
    assert len(heuristics) == len(set(heuristics))


@pytest.mark.parametrize(
    ("data", "bw_min", "bw_max"),
    [(np.zeros(100), 0.01, 10.0), (np.array([1.0, 1.0]), 1e10, 1e11)],
    ids=["zeros", "constant-pair"],
)
def test_degenerate_heuristics_fall_back_to_range_floor(data, bw_min, bw_max):
    assert _collect_heuristic_bandwidths(data, bw_min, bw_max) == [bw_min]
