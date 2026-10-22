"""Unit tests for the runtime-checkable AnomalyDetector protocol."""

import numpy as np
import pytest

from nonconform.structures import AnomalyDetector

PROTOCOL_METHODS = ("fit", "decision_function", "get_params", "set_params")


class SklearnStyleDetector:
    """sklearn-style detector that implements the full protocol."""

    def __init__(self, param1=1, param2="value"):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        return np.zeros(len(X))

    def get_params(self, deep=True):
        return {"param1": self.param1, "param2": self.param2}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def _returns_self(self, *args, **kwargs):
    return self


def test_detector_with_every_protocol_method_is_compliant():
    assert isinstance(SklearnStyleDetector(), AnomalyDetector)


@pytest.mark.parametrize("missing", PROTOCOL_METHODS)
def test_detector_missing_a_protocol_method_is_not_compliant(missing):
    methods = {name: _returns_self for name in PROTOCOL_METHODS if name != missing}
    detector = type(f"DetectorWithout_{missing}", (), methods)()

    assert not isinstance(detector, AnomalyDetector)


def test_runtime_check_only_requires_attribute_presence():
    # @runtime_checkable checks attribute existence, not callability; the
    # actual check happens at usage time.
    detector = type("FakeDetector", (), dict.fromkeys(PROTOCOL_METHODS, "x"))()

    assert isinstance(detector, AnomalyDetector)
