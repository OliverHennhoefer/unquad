"""Input validation and default configuration of public martingales."""

import numpy as np
import pytest

from nonconform.martingales import (
    AlarmConfig,
    PowerMartingale,
    SimpleMixtureMartingale,
)


def test_simple_mixture_default_grid_spans_min_epsilon_to_one():
    np.testing.assert_allclose(
        SimpleMixtureMartingale().epsilons, np.linspace(0.01, 1.0, 100)
    )
    np.testing.assert_allclose(
        SimpleMixtureMartingale(n_grid=5, min_epsilon=0.2).epsilons,
        [0.2, 0.4, 0.6, 0.8, 1.0],
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_epsilon": 0.0}, r"min_epsilon must be in \(0, 1\]"),
        ({"min_epsilon": 1.5}, r"min_epsilon must be in \(0, 1\]"),
        ({"epsilons": [0.5, np.nan]}, "epsilons must be finite"),
        ({"epsilons": [0.0, 0.5]}, r"All epsilons must be in \(0, 1\]"),
        ({"epsilons": [0.5, 1.5]}, r"All epsilons must be in \(0, 1\]"),
    ],
)
def test_simple_mixture_rejects_invalid_epsilons(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SimpleMixtureMartingale(**kwargs)


@pytest.mark.parametrize(
    ("p_value", "expected"), [(-1e-13, 0.0), (1.0 + 1e-13, 1.0)], ids=["low", "high"]
)
def test_p_values_within_tolerance_are_clipped_to_unit_interval(p_value, expected):
    assert PowerMartingale(epsilon=0.5).update(p_value).p_value == expected


def test_alarm_thresholds_must_be_numeric():
    with pytest.raises(ValueError, match="ville_threshold must be a positive finite"):
        AlarmConfig(ville_threshold="high")
