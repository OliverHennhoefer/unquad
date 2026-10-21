import numpy as np
import pytest

from nonconform._internal.validation import (
    as_1d_numeric,
    validate_finite,
    validate_non_negative_finite,
    validate_p_values,
    validate_positive_finite,
    validate_probability,
)


def test_array_validators_accept_empty_arrays() -> None:
    empty = np.array([], dtype=float)

    validate_non_negative_finite("weights", empty)
    validate_finite("scores", empty)
    validate_p_values(empty)


@pytest.mark.parametrize(
    ("validator", "values", "match"),
    [
        (
            as_1d_numeric,
            np.array(["x"], dtype=object),
            "values must be a numeric array",
        ),
        (validate_finite, np.array([np.nan]), "values must be finite"),
        (validate_non_negative_finite, np.array([np.inf]), "values must be finite"),
        (validate_non_negative_finite, np.array([-1.0]), "values must be non-negative"),
    ],
    ids=["non-numeric", "nan", "infinite-weight", "negative-weight"],
)
def test_array_validators_reject_invalid_values(validator, values, match) -> None:
    with pytest.raises(ValueError, match=match):
        validator("values", values)


@pytest.mark.parametrize(
    ("validator", "value", "match"),
    [
        (validate_probability, "high", r"level must be a numeric value in \(0, 1\)"),
        (validate_probability, 1.0, r"level must be in \(0, 1\)"),
        (validate_positive_finite, None, "level must be a positive finite value"),
        (validate_positive_finite, np.inf, "level must be a positive finite value"),
    ],
    ids=[
        "probability-non-numeric",
        "probability-boundary",
        "positive-non-numeric",
        "positive-infinite",
    ],
)
def test_scalar_validators_reject_invalid_values(validator, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        validator("level", value)


def test_scalar_validators_return_floats() -> None:
    assert validate_probability("level", "0.25") == 0.25
    assert validate_positive_finite("scale", 3) == 3.0
