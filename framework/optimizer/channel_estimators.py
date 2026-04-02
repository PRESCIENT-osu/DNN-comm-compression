"""Channel capacity estimators for the optimization loop.

Each estimator maintains a window of throughput observations and returns a
capacity estimate used by the optimizer to set compression rates.  The
``ChannelEstimator`` Protocol in ``link_prober.py`` defines the interface;
all classes here implement it.

Available estimators
--------------------
- ``LastObservation``  — returns the most recent probe value.
- ``MeanEstimator``    — running mean over all observations.
- ``RunningMin``       — running minimum over all observations.
- ``MovingAverage``    — mean over a sliding window of recent observations.
- ``LCB``              — lower-confidence bound (mean − z·σ over window).
"""

from __future__ import annotations

import statistics
from collections import deque

from framework.datamodels.opt_experiment import (
    ChannelEstimatorConfig,
    ChannelEstimatorType,
)

# ---------------------------------------------------------------------------
# Concrete estimators
# ---------------------------------------------------------------------------


class LastObservation:
    """Returns the single most recent probe value.

    Falls back to ``warmup_value_bps`` until at least one observation is
    recorded.

    Args:
        warmup_value_bps: Capacity estimate returned before any observations.
    """

    def __init__(self, warmup_value_bps: float = 0.0) -> None:
        self._last: float | None = None
        self._warmup = warmup_value_bps
        self._n = 0

    def update(self, bps: float) -> None:
        """Record a new capacity observation.

        Args:
            bps: Measured throughput in bits per second.
        """
        self._last = bps
        self._n += 1

    def estimate(self) -> float:
        """Return the last observed value (or warmup if no observations yet).

        Returns:
            Capacity estimate in bps.
        """
        return self._last if self._last is not None else self._warmup

    @property
    def estimator_type(self) -> str:
        """Short name of the estimator algorithm.

        Returns:
            ``"last_obs"``
        """
        return "last_obs"

    @property
    def n_observations(self) -> int:
        """Number of calls to ``update()`` so far.

        Returns:
            Observation count.
        """
        return self._n


class MeanEstimator:
    """Running mean over all observations.

    Args:
        warmup_value_bps: Capacity estimate returned before any observations.
    """

    def __init__(self, warmup_value_bps: float = 0.0) -> None:
        self._sum = 0.0
        self._n = 0
        self._warmup = warmup_value_bps

    def update(self, bps: float) -> None:
        """Record a new observation.

        Args:
            bps: Measured throughput in bits per second.
        """
        self._sum += bps
        self._n += 1

    def estimate(self) -> float:
        """Return running mean (or warmup if no observations yet).

        Returns:
            Capacity estimate in bps.
        """
        return self._sum / self._n if self._n > 0 else self._warmup

    @property
    def estimator_type(self) -> str:
        """Short name of the estimator algorithm.

        Returns:
            ``"mean"``
        """
        return "mean"

    @property
    def n_observations(self) -> int:
        """Number of calls to ``update()`` so far.

        Returns:
            Observation count.
        """
        return self._n


class RunningMin:
    """Running minimum over all observations.

    Conservative estimator; never reports a capacity higher than the worst
    observation seen.

    Args:
        warmup_value_bps: Capacity estimate returned before any observations.
    """

    def __init__(self, warmup_value_bps: float = 0.0) -> None:
        self._min: float | None = None
        self._n = 0
        self._warmup = warmup_value_bps

    def update(self, bps: float) -> None:
        """Record a new observation.

        Args:
            bps: Measured throughput in bits per second.
        """
        self._min = bps if self._min is None else min(self._min, bps)
        self._n += 1

    def estimate(self) -> float:
        """Return running minimum (or warmup if no observations yet).

        Returns:
            Capacity estimate in bps.
        """
        return self._min if self._min is not None else self._warmup

    @property
    def estimator_type(self) -> str:
        """Short name of the estimator algorithm.

        Returns:
            ``"running_min"``
        """
        return "running_min"

    @property
    def n_observations(self) -> int:
        """Number of calls to ``update()`` so far.

        Returns:
            Observation count.
        """
        return self._n


class MovingAverage:
    """Simple moving average over the last ``window_size`` observations.

    Args:
        window_size: Number of most recent observations to average.
        warmup_value_bps: Capacity estimate returned before any observations.
    """

    def __init__(self, window_size: int = 10, warmup_value_bps: float = 0.0) -> None:
        self._window: deque[float] = deque(maxlen=window_size)
        self._warmup = warmup_value_bps
        self._n = 0

    def update(self, bps: float) -> None:
        """Record a new observation.

        Args:
            bps: Measured throughput in bits per second.
        """
        self._window.append(bps)
        self._n += 1

    def estimate(self) -> float:
        """Return mean over the current window (or warmup if window is empty).

        Returns:
            Capacity estimate in bps.
        """
        if not self._window:
            return self._warmup
        return statistics.mean(self._window)

    @property
    def estimator_type(self) -> str:
        """Short name of the estimator algorithm.

        Returns:
            ``"moving_average"``
        """
        return "moving_average"

    @property
    def n_observations(self) -> int:
        """Number of calls to ``update()`` so far.

        Returns:
            Observation count.
        """
        return self._n


class LCB:
    """Lower-confidence-bound estimator: mean − z · σ over a sliding window.

    A conservative estimate suitable for CSI-aware optimizers that should
    not exceed actual channel capacity.

    Args:
        window_size: Number of most recent observations to use.
        z: Confidence multiplier (higher → more conservative).
        warmup_value_bps: Capacity estimate returned before enough observations.
    """

    def __init__(
        self,
        window_size: int = 20,
        z: float = 1.0,
        warmup_value_bps: float = 0.0,
    ) -> None:
        self._window: deque[float] = deque(maxlen=window_size)
        self._z = z
        self._warmup = warmup_value_bps
        self._n = 0

    def update(self, bps: float) -> None:
        """Record a new observation.

        Args:
            bps: Measured throughput in bits per second.
        """
        self._window.append(bps)
        self._n += 1

    def estimate(self) -> float:
        """Return mean − z·σ over the current window.

        Falls back to the most recent observation when the window has fewer
        than two samples, and to ``warmup_value_bps`` when empty.

        Returns:
            Capacity estimate in bps.
        """
        if len(self._window) == 0:
            return self._warmup
        if len(self._window) == 1:
            return self._window[0]
        mu = statistics.mean(self._window)
        sigma = statistics.stdev(self._window)
        return max(0.0, mu - self._z * sigma)

    @property
    def estimator_type(self) -> str:
        """Short name of the estimator algorithm.

        Returns:
            ``"lcb"``
        """
        return "lcb"

    @property
    def n_observations(self) -> int:
        """Number of calls to ``update()`` so far.

        Returns:
            Observation count.
        """
        return self._n


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ESTIMATOR_TYPE_MAP: dict[ChannelEstimatorType, type] = {
    ChannelEstimatorType.LAST_OBS: LastObservation,
    ChannelEstimatorType.MEAN: MeanEstimator,
    ChannelEstimatorType.RUNNING_MIN: RunningMin,
    ChannelEstimatorType.MOVING_AVG: MovingAverage,
    ChannelEstimatorType.LCB: LCB,
}


def build_estimator(
    config: ChannelEstimatorConfig,
) -> LastObservation | MeanEstimator | RunningMin | MovingAverage | LCB:
    """Instantiate a channel estimator from its config.

    Args:
        config: Channel estimator config from an optspec sub-experiment.

    Returns:
        Concrete estimator instance.

    Raises:
        ValueError: If ``config.type`` is not a recognised estimator type.
    """
    cls = _ESTIMATOR_TYPE_MAP.get(config.type)
    if cls is None:
        raise ValueError(f"Unknown channel estimator type: {config.type!r}")

    warmup = config.warmup_value_bps or 0.0

    if config.type == ChannelEstimatorType.MOVING_AVG:
        return MovingAverage(
            window_size=config.window_size or 10,
            warmup_value_bps=warmup,
        )
    if config.type == ChannelEstimatorType.LCB:
        return LCB(
            window_size=config.window_size or 20,
            z=config.z or 1.0,
            warmup_value_bps=warmup,
        )
    # LastObservation, MeanEstimator, RunningMin take only warmup.
    return cls(warmup_value_bps=warmup)  # type: ignore[call-arg]
