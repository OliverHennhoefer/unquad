"""Private group labels for group-conditional (Mondrian) calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nonconform.scoring import BaseEstimation

_MAX_LISTED_LABELS = 5


def as_group_labels(groups: Any, *, n_rows: int) -> np.ndarray:
    """Return one non-missing group label per row as an object array."""
    labels = np.asarray(groups, dtype=object)
    if labels.ndim != 1 or labels.shape[0] != n_rows:
        raise ValueError(
            "groups must be one-dimensional with one label per row of x; "
            f"got shape {labels.shape} for {n_rows} rows."
        )
    if pd.isna(labels).any():
        raise ValueError("groups must not contain missing labels.")
    return labels


def _listed(labels: np.ndarray) -> str:
    """Format a few distinct labels for an error message."""
    distinct = pd.unique(labels)
    shown = ", ".join(repr(label) for label in distinct[:_MAX_LISTED_LABELS])
    return shown + (", ..." if distinct.size > _MAX_LISTED_LABELS else "")


@dataclass(frozen=True, slots=True)
class CalibrationGroups:
    """Reference group vocabulary and the group code of each calibration score."""

    labels: np.ndarray
    codes: np.ndarray

    @classmethod
    def from_rows(
        cls,
        reference_labels: np.ndarray,
        calibration_rows: np.ndarray,
    ) -> CalibrationGroups:
        """Encode reference labels and keep the codes of the calibration rows."""
        codes, uniques = pd.factorize(reference_labels)
        return cls(
            labels=np.asarray(uniques, dtype=object),
            codes=codes[calibration_rows].astype(np.intp),
        )

    @property
    def counts(self) -> np.ndarray:
        """Number of calibration scores per group, in vocabulary order."""
        return np.bincount(self.codes, minlength=self.labels.size)

    def calibration_labels(self) -> np.ndarray:
        """Return the group label of each calibration score."""
        return self.labels[self.codes]

    def encode(self, groups: Any, *, n_rows: int) -> np.ndarray:
        """Map test labels to group codes; reject groups without calibration."""
        labels = as_group_labels(groups, n_rows=n_rows)
        codes = pd.Index(self.labels, dtype=object).get_indexer(labels)
        unseen = codes < 0
        if unseen.any():
            raise ValueError(
                "groups contains labels that were not passed to fit() or "
                f"calibrate(): {_listed(labels[unseen])}."
            )
        uncalibrated = self.counts[codes] == 0
        if uncalibrated.any():
            raise ValueError(
                f"No calibration scores for groups {_listed(labels[uncalibrated])}: "
                "all of their reference rows went to training. Increase n_calib "
                "or add reference rows for these groups."
            )
        return codes


def group_conditional_p_values(
    estimation: BaseEstimation,
    scores: np.ndarray,
    test_codes: np.ndarray,
    calibration_scores: np.ndarray,
    calibration_codes: np.ndarray,
) -> np.ndarray:
    """Rank each test score only among the calibration scores of its group."""
    p_values = np.empty(scores.shape[0], dtype=float)
    for code in np.unique(test_codes):
        rows = test_codes == code
        p_values[rows] = estimation.compute_p_values(
            scores[rows], calibration_scores[calibration_codes == code]
        )
    return p_values
