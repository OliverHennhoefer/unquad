from enum import Enum

import pytest

from nonconform.enums import (
    ConformalMode,
    Distribution,
    Kernel,
    Pruning,
    ScorePolarity,
    TieBreakMode,
)


@pytest.mark.parametrize(
    ("enum_cls", "expected_names"),
    [
        (Distribution, {"BETA_BINOMIAL", "UNIFORM", "GRID"}),
        (Pruning, {"HETEROGENEOUS", "HOMOGENEOUS", "DETERMINISTIC"}),
        (ScorePolarity, {"AUTO", "HIGHER_IS_ANOMALOUS", "HIGHER_IS_NORMAL"}),
        (ConformalMode, {"PLUS", "SINGLE_MODEL"}),
        (TieBreakMode, {"CLASSICAL", "RANDOMIZED"}),
        (
            Kernel,
            {
                "GAUSSIAN",
                "EXPONENTIAL",
                "BOX",
                "TRIANGULAR",
                "EPANECHNIKOV",
                "BIWEIGHT",
                "TRIWEIGHT",
                "TRICUBE",
                "COSINE",
            },
        ),
    ],
    ids=[
        "Distribution",
        "Pruning",
        "ScorePolarity",
        "ConformalMode",
        "TieBreakMode",
        "Kernel",
    ],
)
def test_public_enum_has_exact_members_with_unique_values(enum_cls, expected_names):
    assert issubclass(enum_cls, Enum)
    assert set(enum_cls.__members__) == expected_names
    # Duplicate values would become aliases that iteration skips.
    assert len(enum_cls) == len(expected_names)


@pytest.mark.parametrize(
    ("enum_cls", "expected_values"),
    [
        (ConformalMode, {"PLUS": "plus", "SINGLE_MODEL": "single_model"}),
        (TieBreakMode, {"CLASSICAL": "classical", "RANDOMIZED": "randomized"}),
        (
            Kernel,
            {
                "GAUSSIAN": "gaussian",
                "EXPONENTIAL": "exponential",
                "BOX": "box",
                "TRIANGULAR": "tri",
                "EPANECHNIKOV": "epa",
                "BIWEIGHT": "biweight",
                "TRIWEIGHT": "triweight",
                "TRICUBE": "tricube",
                "COSINE": "cosine",
            },
        ),
    ],
    ids=["ConformalMode", "TieBreakMode", "Kernel"],
)
def test_string_valued_enum_values_are_stable(enum_cls, expected_values):
    assert {member.name: member.value for member in enum_cls} == expected_values
