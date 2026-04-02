"""Accuracy surrogate models A_k(η) for compression rate selection.

Each model fits a smooth curve mapping compression rate η ∈ [0, 1] to
expected task accuracy.  Models are trained during the ``accuracy_model``
sub-experiment phase and consumed by optimizers in the slot loop.

Available backends
------------------
- ``SurrogateAccuracyModel`` — polynomial regression (degree 1–5) fit with
  numpy.  Suitable for offline fitting against collected (η, accuracy) pairs.
- ``ConstantAccuracyModel``  — returns a fixed accuracy regardless of η.
  Used as a placeholder when a real model is not yet available.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AccuracyModel(ABC):
    """Abstract accuracy surrogate model.

    Subclasses must implement ``fit`` (optional if loaded from file) and
    ``predict``.
    """

    @abstractmethod
    def predict(self, eta: float) -> float:
        """Return predicted accuracy for the given compression rate.

        Args:
            eta: Compression rate in [0, 1].  1.0 = no compression.

        Returns:
            Predicted accuracy in [0, 1].
        """

    def fit(self, samples: list[tuple[float, float]]) -> None:  # noqa: B027
        """Fit the model from (η, accuracy) training samples.

        Args:
            samples: List of (eta, accuracy) tuples.
        """

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Short name of the model backend.

        Returns:
            Model type string for metric tagging.
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if the model has been fitted or loaded.

        Returns:
            True when predictions are meaningful.
        """


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class ConstantAccuracyModel(AccuracyModel):
    """Returns a fixed accuracy value regardless of η.

    Useful as a placeholder when no fitted model is available or for
    pipelines where accuracy does not depend on compression rate.

    Args:
        value: Constant accuracy to return (default 1.0).
    """

    def __init__(self, value: float = 1.0) -> None:
        self._value = value

    def predict(self, eta: float) -> float:
        """Return the fixed accuracy.

        Args:
            eta: Ignored.

        Returns:
            The constant accuracy value.
        """
        return self._value

    @property
    def model_type(self) -> str:
        """Short name of the model backend.

        Returns:
            ``"constant"``
        """
        return "constant"

    @property
    def is_fitted(self) -> bool:
        """Return True (no fitting required).

        Returns:
            Always True.
        """
        return True


class SurrogateAccuracyModel(AccuracyModel):
    """Polynomial regression surrogate A_k(η).

    Fits a degree-``d`` polynomial to (η, accuracy) samples using
    ``numpy.polyfit``.  Predictions are clipped to [0, 1].

    Args:
        degree: Polynomial degree.  ``"poly3"`` in the spec → degree=3.
    """

    def __init__(self, degree: int = 3) -> None:
        self._degree = degree
        self._coeffs: Any = None  # numpy array after fitting

    def fit(self, samples: list[tuple[float, float]]) -> None:
        """Fit the polynomial from (η, accuracy) samples.

        Requires at least ``degree + 1`` distinct η values.  If fewer
        samples are available the degree is silently reduced.

        Args:
            samples: List of (eta, accuracy) tuples.
        """
        import numpy as np  # type: ignore[import]

        if not samples:
            logger.warning("SurrogateAccuracyModel.fit called with no samples")
            return

        etas = [s[0] for s in samples]
        accs = [s[1] for s in samples]
        n_unique = len(set(etas))
        deg = min(self._degree, n_unique - 1)
        if deg < 1:
            logger.warning(
                "Not enough distinct η values (%d) to fit degree-%d polynomial; "
                "using constant",
                n_unique,
                self._degree,
            )
            self._coeffs = np.array([0.0] * self._degree + [sum(accs) / len(accs)])
            return
        self._coeffs = np.polyfit(etas, accs, deg)
        logger.debug(
            "SurrogateAccuracyModel fitted: degree=%d samples=%d", deg, len(samples)
        )

    def predict(self, eta: float) -> float:
        """Evaluate the fitted polynomial at the given η.

        Args:
            eta: Compression rate in [0, 1].

        Returns:
            Predicted accuracy clipped to [0, 1].  Returns 0.0 if the model
            has not been fitted.
        """
        import numpy as np  # type: ignore[import]

        if self._coeffs is None:
            return 0.0
        raw = float(np.polyval(self._coeffs, eta))
        return max(0.0, min(1.0, raw))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the fitted model to a JSON-compatible dict.

        Returns:
            Dict with ``degree`` and ``coeffs`` keys, or empty dict if not
            fitted.
        """
        if self._coeffs is None:
            return {}
        return {"degree": self._degree, "coeffs": list(self._coeffs.tolist())}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurrogateAccuracyModel:
        """Restore a previously serialised model.

        Args:
            data: Dict produced by ``to_dict()``.

        Returns:
            Restored SurrogateAccuracyModel instance.
        """
        import numpy as np  # type: ignore[import]

        m = cls(degree=data["degree"])
        m._coeffs = np.array(data["coeffs"])
        return m

    @property
    def model_type(self) -> str:
        """Short name of the model backend.

        Returns:
            ``"surrogate"``
        """
        return "surrogate"

    @property
    def is_fitted(self) -> bool:
        """Return True when the polynomial has been fitted.

        Returns:
            True after ``fit()`` or ``from_dict()`` with valid coefficients.
        """
        return self._coeffs is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_accuracy_model(model_type: str) -> AccuracyModel:
    """Instantiate an accuracy model from a type string.

    Args:
        model_type: One of ``"poly1"``–``"poly5"`` for surrogate polynomial
            models.  Unrecognised strings return a ``ConstantAccuracyModel``.

    Returns:
        Unfitted accuracy model instance.
    """
    if model_type.startswith("poly"):
        try:
            degree = int(model_type[4:])
        except ValueError:
            degree = 3
        return SurrogateAccuracyModel(degree=degree)
    logger.warning(
        "Unknown accuracy model type %r; falling back to ConstantAccuracyModel",
        model_type,
    )
    return ConstantAccuracyModel()
