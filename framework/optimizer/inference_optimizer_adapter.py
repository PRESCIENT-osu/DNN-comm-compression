"""Adapter between framework sub-experiments and the external Inference_Optimizer.

Responsibilities
----------------
1. Convert a ``GeneratedOptExperimentConfig`` + profiling artifacts into a list of
   ``InferenceTask`` objects the external optimizers understand.
2. Wrap simulation pipelines with Stein gradient oracles for the accuracy and
   gradient callables required by ``InferenceTask``.
3. Expose a uniform ``BaseOptimizerAdapter`` interface (``step`` / ``observe_capacity``
   / ``update_dual``) so the opt_runner does not need to know which external optimizer
   is active.
4. Provide a ``build_adapter`` factory that instantiates the right concrete adapter
   given a parsed sub-experiment config model.
5. Provide ``extract_eta_per_pipeline_per_link`` to convert the optimizer's integer-
   indexed result back to the named ``{pipeline_id: {link_id: eta}}`` format used by
   ``push_opt_slot_config``.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# External dependency path setup
# ---------------------------------------------------------------------------

_EXTERNAL_ROOT = Path(__file__).parents[2] / "external" / "Inference_Optimizer"
if str(_EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_ROOT))

import torch  # noqa: E402  (after sys.path setup)
from src.core.task import InferenceTask  # noqa: E402
from src.core.toy_A import grad_oracle  # noqa: E402
from src.optimizers.baseline import (  # noqa: E402
    DecoupledEqualSplitStochasticDescentMultiTaskBaseline,
    EstimatedCSICompressionSingleTaskBaseline,
    HistoricalAverageCertaintyEquivalenceMultiTaskBaseline,
    MaxCompressionMultiTaskBaseline,
    MaxCompressionSingleTaskBaseline,
    NoCompressionMultiTaskBaseline,
    NoCompressionSingleTaskBaseline,
    ProportionalResourceAllocationMultiTaskBaseline,
    QueueProportionalHeuristicMultiTaskBaseline,
    StaticEqualShareMultiTaskBaseline,
    StrictPriorityGreedyMultiTaskBaseline,
    UniformCompressionSingleTaskBaseline,
)
from src.optimizers.csi_aware import (  # noqa: E402
    CSIAwareMultiTaskOptimizer,
    CSIAwareSingleTaskOptimizer,
)
from src.optimizers.estimators import (  # noqa: E402
    LastObservationEstimator,
    MeanEstimator,
    MeanMinusZStdLCB,
    MovingAverageEstimator,
    RunningMinEstimator,
)
from src.optimizers.no_csi import (  # noqa: E402
    NoCSIMultiTaskOptimizer,
    NoCSISingleTaskOptimizer,
)

from framework.datamodels.opt_experiment import (  # noqa: E402
    ChannelEstimatorConfig,
    ChannelEstimatorType,
    CsiAwareSubExperiment,
    DecoupledDescentSubExperiment,
    EstimatedCsiSingleSubExperiment,
    GeneratedOptExperimentConfig,
    HistoricalAverageCESubExperiment,
    MaxCompressionMultiSubExperiment,
    MaxCompressionSingleSubExperiment,
    NoCompressionMultiSubExperiment,
    NoCompressionSingleSubExperiment,
    NoCsiSubExperiment,
    ProportionalResourceSubExperiment,
    QueueProportionalSubExperiment,
    StaticEqualShareSubExperiment,
    SteinOracleConfig,
    StrictPriorityGreedySubExperiment,
    UniformCompressionSingleSubExperiment,
)

# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def build_global_order(exp: GeneratedOptExperimentConfig) -> list[str]:
    """Return a stable ordered list of all node names in the experiment.

    Node order matches the ``nodes`` list in the generated experiment config,
    which preserves the definition order from the optspec.

    Args:
        exp: Generated experiment config.

    Returns:
        Ordered list of node names, e.g. ``["A", "B", "C"]``.
    """
    return [n.name for n in exp.nodes]


def probe_dict_to_c_t_vector(
    probe_bps: dict[str, float],
    global_order: list[str],
) -> np.ndarray:
    """Convert a per-link probe dict to a global link capacity vector.

    The global link index for link ``i`` is defined by the ordered pair
    ``(global_order[i], global_order[i+1])``, i.e. the link from the i-th node
    to the (i+1)-th node.

    Args:
        probe_bps: Per-link throughput observations keyed by ``"FromNode-ToNode"``
            link IDs, in bits per second.
        global_order: Ordered list of all node names.

    Returns:
        Numpy array of shape ``(M-1,)`` where ``M = len(global_order)``.
        Links not present in ``probe_bps`` default to ``1.0`` bps (safe fallback
        that allows the optimizer to proceed without error).
    """
    M = len(global_order)
    c_t = np.ones(M - 1, dtype=float)
    for i in range(M - 1):
        link_id = f"{global_order[i]}-{global_order[i + 1]}"
        if link_id in probe_bps:
            c_t[i] = max(1.0, float(probe_bps[link_id]))
    return c_t


# ---------------------------------------------------------------------------
# Stein oracle wrapping
# ---------------------------------------------------------------------------


def _make_stein_callables(
    simulation: Any,
    n_links: int,
    stein_cfg: SteinOracleConfig,
    is_llama: bool,
) -> tuple[
    callable[[np.ndarray], float],
    callable[[np.ndarray], float],
    callable[[np.ndarray], np.ndarray],
]:
    """Build accuracy_callable, accuracy_callable_true, and gradient_callable.

    Both callables wrap the simulation pipeline's ``accuracy`` method.  The
    gradient callable uses the Stein antithetic gradient oracle (``grad_oracle``
    from ``toy_A.py``) applied to the fast accuracy callable.

    Args:
        simulation: A ``SimulatedResNetPipeline`` or ``SimulatedLlamaPipeline``.
        n_links: Number of inter-node links for this pipeline (dimension of eta).
        stein_cfg: Stein oracle hyperparameters.
        is_llama: ``True`` when the simulation is a ``SimulatedLlamaPipeline``.

    Returns:
        Tuple ``(accuracy_callable, accuracy_callable_true, gradient_callable)``
        matching the ``InferenceTask`` constructor signatures.
    """
    n_fast = stein_cfg.n_fast_samples
    sigma = stein_cfg.sigma
    N = stein_cfg.N

    if is_llama:

        def accuracy_callable(eta_np: np.ndarray) -> float:
            eta_t = torch.tensor(eta_np, dtype=torch.float32)
            return float(simulation.accuracy(eta_t, full=False))

        def accuracy_callable_true(eta_np: np.ndarray) -> float:
            eta_t = torch.tensor(eta_np, dtype=torch.float32)
            return float(simulation.accuracy(eta_t, full=True))

    else:

        def accuracy_callable(eta_np: np.ndarray) -> float:
            eta_t = torch.tensor(eta_np, dtype=torch.float32)
            return float(simulation.accuracy(eta_t, n_samples=n_fast))

        def accuracy_callable_true(eta_np: np.ndarray) -> float:
            eta_t = torch.tensor(eta_np, dtype=torch.float32)
            return float(simulation.accuracy(eta_t, n_samples=None))

    _grad_call_count = [0]

    def gradient_callable(eta_np: np.ndarray) -> np.ndarray:
        _grad_call_count[0] += 1
        call_no = _grad_call_count[0]
        logger.info(
            "gradient_callable invocation #%d (eta=%s)", call_no, np.round(eta_np, 3)
        )
        eta_t = torch.tensor(eta_np, dtype=torch.float32)

        def f(e: torch.Tensor) -> torch.Tensor:
            return torch.tensor(
                accuracy_callable(e.numpy(force=True)), dtype=torch.float32
            )

        grad = grad_oracle(f, eta_t, sigma=sigma, N=N)
        logger.info(
            "gradient_callable #%d done (2*N=%d sim calls, grad_norm=%.4f)",
            call_no,
            2 * N,
            float(np.linalg.norm(grad.numpy(force=True))),
        )
        return grad.numpy(force=True)

    return accuracy_callable, accuracy_callable_true, gradient_callable


def _dummy_callables() -> (
    tuple[
        callable[[np.ndarray], float],
        callable[[np.ndarray], float],
        callable[[np.ndarray], np.ndarray],
    ]
):
    """Return trivial callables for optimizers that do not evaluate A_k.

    CSI-aware baselines (MaxCompression, NoCompression, Uniform, StaticEqualShare,
    ProportionalResource, StrictPriorityGreedy, CSIAware) fix η without calling
    the accuracy function.

    Returns:
        Tuple of dummy accuracy_callable (→ 1.0), accuracy_callable_true (→ 1.0),
        gradient_callable (→ zeros).
    """

    def accuracy_callable(eta_np: np.ndarray) -> float:  # noqa: ARG001
        return 1.0

    def accuracy_callable_true(eta_np: np.ndarray) -> float:  # noqa: ARG001
        return 1.0

    def gradient_callable(eta_np: np.ndarray) -> np.ndarray:
        return np.zeros_like(eta_np)

    return accuracy_callable, accuracy_callable_true, gradient_callable


# ---------------------------------------------------------------------------
# InferenceTask construction
# ---------------------------------------------------------------------------


def build_inference_tasks(
    exp: GeneratedOptExperimentConfig,
    tau_per_node: dict[str, dict[str, float]],
    a_per_link_bytes: dict[str, float],
    global_order: list[str],
    simulations: dict[str, Any] | None = None,
    stein_cfg: SteinOracleConfig | None = None,
    accuracy_models: dict[str, Any] | None = None,
) -> tuple[list[InferenceTask], dict[int, str], dict[str, int]]:
    """Build an ``InferenceTask`` for each pipeline in the experiment.

    Pipelines are sorted by name for a stable task_id assignment: the
    i-th pipeline in sorted order gets ``task_id = i``.

    Callable selection priority per pipeline:

    1. **Stein oracle** — when ``simulations[pipeline_id]`` exists and
       ``stein_cfg`` is not None.  Real gradient via antithetic perturbation.
    2. **Surrogate** — when ``accuracy_models[pipeline_id]`` is a fitted
       ``AccuracyModel``.  Gradient via central finite differences.
    3. **Dummy** — constant 1.0 / zero gradient.  Used for CSI-aware and
       baseline adapters that never call the accuracy function.

    Args:
        exp: Generated experiment config.
        tau_per_node: Per-pipeline per-node compute latency in seconds, keyed by
            ``{pipeline_id: {node_id: seconds}}``.
        a_per_link_bytes: Mean activation payload bytes per link, keyed by link_id.
        global_order: Ordered list of all node names (index = global node index).
        simulations: Optional map of pipeline_id → simulation pipeline object.
            When provided and ``stein_cfg`` is set, real accuracy/gradient callables
            are installed.  Otherwise, dummy callables are used.
        stein_cfg: Stein oracle config.  Required when ``simulations`` is not None.
        accuracy_models: Optional map of pipeline_id → fitted ``AccuracyModel``.
            When provided and no Stein oracle is available for a pipeline, the
            surrogate callables are installed via ``make_surrogate_callables``.

    Returns:
        Tuple of:
        - ``inference_tasks``: Ordered list of ``InferenceTask`` objects.
        - ``task_id_to_pipeline``: ``{task_id: pipeline_id}`` mapping.
        - ``pipeline_to_task_id``: ``{pipeline_id: task_id}`` mapping.

    Raises:
        ValueError: If a pipeline's flow references a node not in ``global_order``
            or if ``tau_per_node`` is missing required entries.
    """
    node_to_idx = {name: i for i, name in enumerate(global_order)}
    sorted_pipeline_names = sorted(p.name for p in exp.pipelines)
    task_id_to_pipeline: dict[int, str] = {}
    pipeline_to_task_id: dict[str, int] = {}

    for task_id, pipeline_id in enumerate(sorted_pipeline_names):
        task_id_to_pipeline[task_id] = pipeline_id
        pipeline_to_task_id[pipeline_id] = task_id

    inference_tasks: list[InferenceTask] = []

    for task_id, pipeline_id in enumerate(sorted_pipeline_names):
        pipeline = exp.pipeline_for(pipeline_id)
        flow = pipeline.flow

        if not flow:
            raise ValueError(
                f"Pipeline '{pipeline_id}' has an empty flow; cannot build InferenceTask."
            )

        b_k = node_to_idx[flow[0]]
        L_k = len(flow)

        # tau: global_node_idx → compute seconds
        tau_map = tau_per_node.get(pipeline_id, {})
        if not tau_map:
            logger.warning(
                "build_inference_tasks: no profiling tau data for pipeline '%s' — "
                "all nodes will use fallback τ=1e-3 s.  "
                "Ensure the profiling sub-experiment ran successfully before this step.",
                pipeline_id,
            )
        tau: dict[int, float] = {}
        for node_name in flow:
            global_idx = node_to_idx[node_name]
            if node_name not in tau_map:
                logger.warning(
                    "build_inference_tasks: no profiling tau for pipeline '%s' node '%s' — "
                    "using fallback τ=1e-3 s.  Optimizer delay model will be inaccurate.",
                    pipeline_id,
                    node_name,
                )
            tau[global_idx] = float(tau_map.get(node_name, 1e-3))

        # a: global_link_idx → bits.  c_t is in bps; for the delay formula
        # a_i * eta_i / c_i(t) to yield seconds, a must be in bits.
        # a_per_link_bytes holds the compressed payload size in bytes, so
        # multiply by 8 to convert.
        a: dict[int, float] = {}
        eta_min: dict[int, float] = {}
        for i in range(len(flow) - 1):
            global_link_idx = node_to_idx[flow[i]]
            link_id = f"{flow[i]}-{flow[i + 1]}"
            if link_id not in a_per_link_bytes:
                logger.warning(
                    "build_inference_tasks: no profiling activation size for link '%s' — "
                    "using fallback a=8 bits.  Optimizer transmission delay will be inaccurate.",
                    link_id,
                )
            a[global_link_idx] = float(a_per_link_bytes.get(link_id, 1.0)) * 8
            try:
                lk_cfg = exp.link_for(flow[i], flow[i + 1])
                eta_min[global_link_idx] = float(lk_cfg.eta_min)
            except KeyError:
                eta_min[global_link_idx] = 0.05

        # Throughput target R_k(t) — constant across slots
        task_cfg = exp.tasks[pipeline_id]
        R_k = float(task_cfg.throughput_target)

        def _make_r_callable(r: float) -> callable[[int], float]:
            def R_k_callable(t: int) -> float:  # noqa: ARG001
                return r

            return R_k_callable

        w_k = float(task_cfg.task_weight)

        # Accuracy / gradient callables — three-way priority selection.
        simulation = (simulations or {}).get(pipeline_id)
        accuracy_model = (accuracy_models or {}).get(pipeline_id)

        if simulation is not None and stein_cfg is not None:
            # Priority 1: Stein oracle via simulation.
            try:
                from models.llama.simulation.pipeline import (  # noqa: PLC0415
                    SimulatedLlamaPipeline,
                )

                is_llama = isinstance(simulation, SimulatedLlamaPipeline)
            except ImportError:
                is_llama = False

            acc_fn, acc_true_fn, grad_fn = _make_stein_callables(
                simulation, L_k - 1, stein_cfg, is_llama
            )
        elif accuracy_model is not None and accuracy_model.is_fitted:
            # Priority 2: Fitted surrogate accuracy model.
            from framework.optimizer.accuracy_model import (  # noqa: PLC0415
                make_surrogate_callables,
            )

            acc_fn, acc_true_fn, grad_fn = make_surrogate_callables(
                accuracy_model, L_k - 1
            )
            logger.debug(
                "build_inference_tasks: pipeline '%s' using surrogate accuracy model "
                "(type=%s)",
                pipeline_id,
                accuracy_model.model_type,
            )
        else:
            # Priority 3: Dummy callables (CSI-aware / baseline adapters).
            acc_fn, acc_true_fn, grad_fn = _dummy_callables()

        inference_tasks.append(
            InferenceTask(
                task_id=task_id,
                b_k=b_k,
                L_k=L_k,
                tau=tau,
                a=a,
                eta_min=eta_min,
                R_k_callable=_make_r_callable(R_k),
                w_k=w_k,
                accuracy_callable=acc_fn,
                accuracy_callable_true=acc_true_fn,
                gradient_callable=grad_fn,
            )
        )
        logger.debug(
            "Built InferenceTask: task_id=%d pipeline=%s b_k=%d L_k=%d tau=%s",
            task_id,
            pipeline_id,
            b_k,
            L_k,
            tau,
        )

    return inference_tasks, task_id_to_pipeline, pipeline_to_task_id


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------


def extract_eta_per_pipeline_per_link(
    result: dict[int, dict[str, np.ndarray]],
    inference_tasks: list[InferenceTask],
    task_id_to_pipeline: dict[int, str],
    global_order: list[str],
    exp: GeneratedOptExperimentConfig,
) -> dict[str, dict[str, float]]:
    """Convert an optimizer result dict to ``eta_per_pipeline_per_link``.

    The optimizer returns ``{task_id: {"eta": np.ndarray, ...}}`` where the
    eta array is indexed by global link index offset from ``b_k``.  This
    function translates that to the named ``{pipeline_id: {link_id: eta}}``
    format expected by ``push_opt_slot_config``.

    Values are clamped to ``[eta_min, eta_max]`` for the corresponding link.

    Args:
        result: Optimizer output, or ``None`` if the problem was infeasible.
        inference_tasks: List of InferenceTask objects (for b_k / L_k).
        task_id_to_pipeline: Mapping from task_id to pipeline_id.
        global_order: Ordered list of all node names.
        exp: Generated experiment config (for link bounds).

    Returns:
        ``{pipeline_id: {link_id: eta}}`` for every pipeline with at least one link.
    """
    task_map = {t.task_id: t for t in inference_tasks}
    out: dict[str, dict[str, float]] = {}

    for task_id, alloc in result.items():
        pipeline_id = task_id_to_pipeline.get(task_id)
        if pipeline_id is None:
            logger.warning("Unknown task_id %d in optimizer result; skipping.", task_id)
            continue

        task = task_map[task_id]
        eta_vec: np.ndarray = alloc.get("eta", np.array([], dtype=float))
        link_etas: dict[str, float] = {}

        for local_idx in range(task.L_k - 1):
            global_link_idx = task.b_k + local_idx
            from_node = global_order[global_link_idx]
            to_node = global_order[global_link_idx + 1]
            link_id = f"{from_node}-{to_node}"

            eta_val = float(eta_vec[local_idx]) if local_idx < len(eta_vec) else 1.0

            try:
                lk_cfg = exp.link_for(from_node, to_node)
                eta_val = float(np.clip(eta_val, lk_cfg.eta_min, lk_cfg.eta_max))
            except KeyError:
                pass

            link_etas[link_id] = eta_val

        if link_etas:
            out[pipeline_id] = link_etas

    return out


def extract_s_comp_per_pipeline_per_node(
    result: dict[int, dict[str, np.ndarray]],
    inference_tasks: list[InferenceTask],
    task_id_to_pipeline: dict[int, str],
    global_order: list[str],
) -> dict[str, dict[str, float]]:
    """Convert an optimizer result dict to WFQ weights per node.

    The optimizer's s_comp array for task k has length L_k, with element i
    corresponding to global node index ``b_k + i``.  After
    ``scale_allocations_to_unit_sum`` the weights at each shared node sum to 1.

    Args:
        result: Optimizer output ``{task_id: {"s_comp": ndarray, ...}}``.
        inference_tasks: List of InferenceTask objects (for b_k / L_k).
        task_id_to_pipeline: Mapping from task_id to pipeline_id.
        global_order: Ordered list of all node names.

    Returns:
        ``{node_name: {pipeline_id: weight}}`` for every node traversed by at
        least one pipeline in the result.
    """
    task_map = {t.task_id: t for t in inference_tasks}
    out: dict[str, dict[str, float]] = {}

    for task_id, alloc in result.items():
        pipeline_id = task_id_to_pipeline.get(task_id)
        if pipeline_id is None:
            continue
        task = task_map[task_id]
        s_comp_vec: np.ndarray = alloc.get("s_comp", np.ones(task.L_k))

        for local_idx in range(task.L_k):
            global_node_idx = task.b_k + local_idx
            if global_node_idx >= len(global_order):
                continue
            node_name = global_order[global_node_idx]
            weight = (
                float(s_comp_vec[local_idx]) if local_idx < len(s_comp_vec) else 1.0
            )
            out.setdefault(node_name, {})[pipeline_id] = max(weight, 0.0)

    return out


def eta_max_fallback(
    exp: GeneratedOptExperimentConfig,
    pipeline_to_task_id: dict[str, int],
    inference_tasks: list[InferenceTask],
    global_order: list[str],
) -> dict[str, dict[str, float]]:
    """Build an eta_per_pipeline_per_link dict with eta_max on every link.

    Used as a fallback when the optimizer declares infeasibility.

    Args:
        exp: Generated experiment config.
        pipeline_to_task_id: Mapping from pipeline_id to task_id.
        inference_tasks: List of InferenceTask objects.
        global_order: Ordered list of all node names.

    Returns:
        ``{pipeline_id: {link_id: eta_max}}`` for every pipeline.
    """
    task_map = {t.task_id: t for t in inference_tasks}
    out: dict[str, dict[str, float]] = {}

    for pipeline_id, task_id in pipeline_to_task_id.items():
        task = task_map[task_id]
        link_etas: dict[str, float] = {}
        for local_idx in range(task.L_k - 1):
            global_link_idx = task.b_k + local_idx
            from_node = global_order[global_link_idx]
            to_node = global_order[global_link_idx + 1]
            link_id = f"{from_node}-{to_node}"
            try:
                lk_cfg = exp.link_for(from_node, to_node)
                link_etas[link_id] = float(lk_cfg.eta_max)
            except KeyError:
                link_etas[link_id] = 1.0
        if link_etas:
            out[pipeline_id] = link_etas

    return out


# ---------------------------------------------------------------------------
# BaseOptimizerAdapter
# ---------------------------------------------------------------------------


class BaseOptimizerAdapter(ABC):
    """Uniform interface over external optimizer classes.

    The opt_runner calls ``step`` once per slot (after acquiring a channel
    estimate), ``observe_capacity`` to update any internal estimator state,
    and ``update_dual`` to feed back the observed delay after the slot
    completes.

    Args:
        inference_tasks: Ordered list of InferenceTask objects.
        task_id_to_pipeline: Mapping task_id → pipeline_id.
        pipeline_to_task_id: Mapping pipeline_id → task_id.
        global_order: Ordered list of all node names.
        exp: Generated experiment config.
    """

    def __init__(
        self,
        inference_tasks: list[InferenceTask],
        task_id_to_pipeline: dict[int, str],
        pipeline_to_task_id: dict[str, int],
        global_order: list[str],
        exp: GeneratedOptExperimentConfig,
    ) -> None:
        self._inference_tasks = inference_tasks
        self._task_id_to_pipeline = task_id_to_pipeline
        self._pipeline_to_task_id = pipeline_to_task_id
        self._global_order = global_order
        self._exp = exp
        self.probe_every_slot: bool = False

    @abstractmethod
    def step(
        self, t: int, c_t: np.ndarray
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], bool]:
        """Run one optimization slot.

        Args:
            t: Current slot index.
            c_t: True link capacity vector of shape ``(M-1,)`` in bps.

        Returns:
            Tuple of (``eta_per_pipeline_per_link``, ``s_comp_per_node``,
            ``infeasible``).  ``s_comp_per_node`` maps node name to
            ``{pipeline_id: weight}`` for WFQ scheduling.  When ``infeasible``
            is True the etas are eta_max fallbacks and ``s_comp_per_node`` is
            empty.
        """

    def observe_capacity(self, c_t: np.ndarray) -> None:  # noqa: B027
        """Update internal channel estimator with the true observed capacity.

        Default implementation is a no-op (CSI-aware adapters do not need it).

        Args:
            c_t: True link capacity vector of shape ``(M-1,)`` in bps.
        """

    def update_dual(self, t: int, actual_delays: dict[int, float]) -> None:  # noqa: B027
        """Update Lagrangian dual variables from observed bottleneck delays.

        Default implementation is a no-op (CSI-aware adapters do not need it).

        Args:
            t: Current slot index.
            actual_delays: Mapping from task_id to the observed bottleneck
                delay in seconds.
        """

    def get_dual_variables(self) -> dict[str, float]:  # noqa: B027
        """Return current dual variable λ_k for each pipeline.

        Returns the lambda values that were used in the most recent ``step()``
        call — i.e., before the next ``update_dual()`` advances them.  Only
        meaningful for adapters that maintain a dual variable (subclasses of
        ``DualEstimatedAdapter``).  Default returns an empty dict so callers
        do not need to branch on adapter type.

        Returns:
            Mapping of pipeline_id → λ_k, or ``{}`` if this adapter type does
            not maintain dual variables.
        """
        return {}

    def _extract(
        self, result: dict[int, dict[str, np.ndarray]] | None
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], bool]:
        """Extract eta and s_comp from an optimizer result.

        Args:
            result: Optimizer output, or ``None`` if infeasible.

        Returns:
            Tuple of (eta_per_pipeline_per_link, s_comp_per_node, infeasible).
            On infeasibility, eta is the eta_max fallback and s_comp_per_node
            is empty (nodes keep their current WFQ weights).
        """
        if result is None:
            logger.warning(
                "Optimizer returned None (infeasible); using eta_max fallback."
            )
            return (
                eta_max_fallback(
                    self._exp,
                    self._pipeline_to_task_id,
                    self._inference_tasks,
                    self._global_order,
                ),
                {},
                True,
            )
        return (
            extract_eta_per_pipeline_per_link(
                result,
                self._inference_tasks,
                self._task_id_to_pipeline,
                self._global_order,
                self._exp,
            ),
            extract_s_comp_per_pipeline_per_node(
                result,
                self._inference_tasks,
                self._task_id_to_pipeline,
                self._global_order,
            ),
            False,
        )


# ---------------------------------------------------------------------------
# Concrete adapter: direct CSI (uses c_t as-is)
# ---------------------------------------------------------------------------


class DirectCsiAdapter(BaseOptimizerAdapter):
    """Adapter for optimizers that take the true link capacity ``c_t`` directly.

    Used for:
    - ``CSIAwareSingleTaskOptimizer`` / ``CSIAwareMultiTaskOptimizer``
    - ``MaxCompression*`` / ``NoCompression*`` baselines
    - ``UniformCompressionSingleTaskBaseline``
    - ``StaticEqualShare`` / ``ProportionalResource`` / ``StrictPriorityGreedy``

    Args:
        optimizer: External optimizer instance.
    """

    def __init__(
        self,
        optimizer: Any,
        inference_tasks: list[InferenceTask],
        task_id_to_pipeline: dict[int, str],
        pipeline_to_task_id: dict[str, int],
        global_order: list[str],
        exp: GeneratedOptExperimentConfig,
        probe_every_slot: bool = False,
    ) -> None:
        super().__init__(
            inference_tasks, task_id_to_pipeline, pipeline_to_task_id, global_order, exp
        )
        self._optimizer = optimizer
        self.probe_every_slot = probe_every_slot

    def step(
        self, t: int, c_t: np.ndarray
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], bool]:
        """Run one slot with true link capacities.

        Args:
            t: Current slot index.
            c_t: True link capacity vector.

        Returns:
            Tuple of (eta_per_pipeline_per_link, s_comp_per_node, infeasible).
        """
        result = self._optimizer.optimize(t, c_t)
        return self._extract(result)


# ---------------------------------------------------------------------------
# Concrete adapter: estimated CSI (uses c_hat from estimator)
# ---------------------------------------------------------------------------


class EstimatedAdapter(BaseOptimizerAdapter):
    """Adapter for optimizers that take an estimated channel capacity ``c_hat``.

    Maintains an external estimator (from ``src.optimizers.estimators``) that
    is updated with each true capacity observation via ``observe_capacity``.

    Used for:
    - ``EstimatedCSICompressionSingleTaskBaseline`` and its Myopic / Conservative
      / MovingAverage subclasses
    - ``HistoricalAverageCertaintyEquivalenceMultiTaskBaseline``

    Args:
        optimizer: External optimizer instance.
        estimator: External estimator instance (from ``src.optimizers.estimators``).
    """

    def __init__(
        self,
        optimizer: Any,
        estimator: Any,
        inference_tasks: list[InferenceTask],
        task_id_to_pipeline: dict[int, str],
        pipeline_to_task_id: dict[str, int],
        global_order: list[str],
        exp: GeneratedOptExperimentConfig,
    ) -> None:
        super().__init__(
            inference_tasks, task_id_to_pipeline, pipeline_to_task_id, global_order, exp
        )
        self._optimizer = optimizer
        self._estimator = estimator
        self._slot = 0

    def observe_capacity(self, c_t: np.ndarray) -> None:
        """Record the true observed capacity for the estimator.

        Args:
            c_t: True link capacity vector.
        """
        self._estimator.update(c_t)

    def step(
        self,
        t: int,
        c_t: np.ndarray,  # noqa: ARG002
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], bool]:
        """Run one slot with the current capacity estimate.

        Args:
            t: Current slot index.
            c_t: True capacity (unused; estimator provides c_hat).

        Returns:
            Tuple of (eta_per_pipeline_per_link, s_comp_per_node, infeasible).
        """
        c_hat = self._estimator.estimate(t)
        result = self._optimizer.optimize(t, c_hat)
        self._slot = t
        return self._extract(result)


# ---------------------------------------------------------------------------
# Concrete adapter: estimated CSI + dual update
# ---------------------------------------------------------------------------


class DualEstimatedAdapter(EstimatedAdapter):
    """Estimated-CSI adapter that also maintains Lagrangian dual variables.

    Extends ``EstimatedAdapter`` by calling ``optimizer.update_dual`` after
    each slot, enabling algorithms that learn online (e.g. decoupled descent,
    queue-proportional heuristic, no-CSI SLSQP).

    Used for:
    - ``NoCSISingleTaskOptimizer`` / ``NoCSIMultiTaskOptimizer``
    - ``DecoupledEqualSplitStochasticDescentMultiTaskBaseline``
    - ``QueueProportionalHeuristicMultiTaskBaseline``
    """

    def update_dual(self, t: int, actual_delays: dict[int, float]) -> None:
        """Update the optimizer's dual variables.

        Args:
            t: Current slot index.
            actual_delays: Mapping from task_id to the observed bottleneck
                delay in seconds.
        """
        self._optimizer.update_dual(t, actual_delays)

    def get_dual_variables(self) -> dict[str, float]:
        """Return current λ_k for each pipeline, keyed by pipeline_id.

        Handles both single-task optimizers (``lambda_t: float``) and
        multi-task optimizers (``lambda_k: dict[int, float]``).

        Returns:
            Mapping of pipeline_id → λ_k for all pipelines tracked by this
            adapter.
        """
        opt = self._optimizer
        if hasattr(opt, "lambda_k"):
            return {
                self._task_id_to_pipeline[tid]: float(lam)
                for tid, lam in opt.lambda_k.items()
                if tid in self._task_id_to_pipeline
            }
        if hasattr(opt, "lambda_t"):
            # Single-task optimizer — exactly one pipeline.
            pipeline_ids = list(self._task_id_to_pipeline.values())
            if pipeline_ids:
                return {pipeline_ids[0]: float(opt.lambda_t)}
        return {}


# ---------------------------------------------------------------------------
# Internal estimator factory
# ---------------------------------------------------------------------------


@dataclass
class WindowedLCBEstimator(MeanMinusZStdLCB):
    """Sliding-window LCB estimator extending the library's ``MeanMinusZStdLCB``.

    ``MeanMinusZStdLCB`` accumulates an unbounded history.  This subclass
    caps each per-link history to the last ``window`` observations so the
    estimate adapts to channel changes rather than converging to a global mean.

    Args:
        num_links: Number of links (dimension of the c_t vector).
        window: Maximum number of recent observations to retain per link.
        warmup_value: Value returned before any observations (and during the
            first ``warmup`` slots by the parent class).
        z: Confidence multiplier: estimate = mean − z·σ.
        warmup: Number of initial slots for which ``warmup_value`` is returned
            regardless of observed history.
        eps: Floor applied to the estimate to prevent zero or negative values.
    """

    window: int = 20

    def update(self, c_t: np.ndarray) -> None:
        """Record a new per-link capacity observation, retaining only the last ``window``.

        Args:
            c_t: Per-link capacity vector of shape ``(num_links,)``.
        """
        super().update(c_t)
        for i in range(self.num_links):
            if len(self._hist[i]) > self.window:
                self._hist[i] = self._hist[i][-self.window :]


def _build_external_estimator(
    cfg: ChannelEstimatorConfig,
    num_links: int,
) -> Any:
    """Build an external estimator for use inside an optimizer adapter.

    Dispatches to the appropriate class from ``src.optimizers.estimators`` or
    to ``WindowedLCBEstimator`` for the windowed LCB variant.

    Args:
        cfg: Channel estimator config from the sub-experiment.
        num_links: Number of links in the topology (vector size).

    Returns:
        An estimator instance with ``estimate(t)`` and ``update(c_t)`` methods.
    """
    warmup_kwargs: dict[str, float] = (
        {"warmup_value": float(cfg.warmup_value_bps)}
        if cfg.warmup_value_bps is not None
        else {}
    )

    if cfg.type == ChannelEstimatorType.LAST_OBS:
        return LastObservationEstimator(num_links=num_links, **warmup_kwargs)

    if cfg.type == ChannelEstimatorType.MEAN:
        if not warmup_kwargs:
            raise ValueError(
                "warmup_value_bps is required for channel_estimator type 'mean'"
            )
        return MeanEstimator(num_links=num_links, **warmup_kwargs)

    if cfg.type == ChannelEstimatorType.RUNNING_MIN:
        return RunningMinEstimator(num_links=num_links, **warmup_kwargs)

    if cfg.type == ChannelEstimatorType.MOVING_AVG:
        window_kwargs: dict[str, int] = (
            {"window": cfg.window_size} if cfg.window_size is not None else {}
        )
        return MovingAverageEstimator(
            num_links=num_links, **warmup_kwargs, **window_kwargs
        )

    if cfg.type == ChannelEstimatorType.LCB:
        z_kwargs: dict[str, float] = {"z": cfg.z} if cfg.z is not None else {}
        return MeanMinusZStdLCB(num_links=num_links, **warmup_kwargs, **z_kwargs)

    if cfg.type == ChannelEstimatorType.WINDOWED_LCB:
        z_kwargs = {"z": cfg.z} if cfg.z is not None else {}
        window_kwargs = (
            {"window": cfg.window_size} if cfg.window_size is not None else {}
        )
        return WindowedLCBEstimator(
            num_links=num_links, **warmup_kwargs, **z_kwargs, **window_kwargs
        )

    raise ValueError(f"Unsupported channel estimator type: {cfg.type!r}")


# ---------------------------------------------------------------------------
# build_adapter factory
# ---------------------------------------------------------------------------


def build_adapter(
    sub_exp: Any,
    inference_tasks: list[InferenceTask],
    task_id_to_pipeline: dict[int, str],
    pipeline_to_task_id: dict[str, int],
    global_order: list[str],
    exp: GeneratedOptExperimentConfig,
    mu: float | None = None,
) -> BaseOptimizerAdapter:
    """Instantiate the correct ``BaseOptimizerAdapter`` for a sub-experiment.

    Args:
        sub_exp: Parsed sub-experiment config (an ``OptSubExperiment`` variant).
        inference_tasks: List of ``InferenceTask`` objects (from
            ``build_inference_tasks``).
        task_id_to_pipeline: Mapping task_id → pipeline_id.
        pipeline_to_task_id: Mapping pipeline_id → task_id.
        global_order: Ordered list of all node names.
        exp: Generated experiment config.
        mu: Override μ for ``NoCsiSubExperiment`` (one adapter is built per mu
            value in the sweep, so the caller passes in the specific mu).

    Returns:
        A ``BaseOptimizerAdapter`` instance ready for use in the slot loop.

    Raises:
        ValueError: If ``sub_exp.type`` is not a recognised optimization type.
    """
    M = len(global_order)
    K = len(inference_tasks)
    common = dict(
        inference_tasks=inference_tasks,
        task_id_to_pipeline=task_id_to_pipeline,
        pipeline_to_task_id=pipeline_to_task_id,
        global_order=global_order,
        exp=exp,
    )

    # ------------------------------------------------------------------
    # CSI-aware optimizers (single or multi task)
    # ------------------------------------------------------------------
    if isinstance(sub_exp, CsiAwareSubExperiment):
        if K == 1:
            opt = CSIAwareSingleTaskOptimizer(M=M, tasks=inference_tasks)
        else:
            opt = CSIAwareMultiTaskOptimizer(M=M, tasks=inference_tasks)
        return DirectCsiAdapter(optimizer=opt, probe_every_slot=True, **common)

    # ------------------------------------------------------------------
    # No-CSI optimizer (single or multi task)
    # ------------------------------------------------------------------
    if isinstance(sub_exp, NoCsiSubExperiment):
        actual_mu = mu if mu is not None else 1.0
        epsilon = float(exp.optimization_loop.epsilon)
        num_links = M - 1
        estimator = _build_external_estimator(sub_exp.channel_estimator, num_links)
        if K == 1:
            opt = NoCSISingleTaskOptimizer(
                M=M, tasks=inference_tasks, mu=actual_mu, epsilon=epsilon
            )
        else:
            opt = NoCSIMultiTaskOptimizer(
                M=M,
                tasks=inference_tasks,
                mu=actual_mu,
                epsilon=epsilon,
                J=sub_exp.bcd_iterations,
            )
        return DualEstimatedAdapter(optimizer=opt, estimator=estimator, **common)

    # ------------------------------------------------------------------
    # Single-task baselines
    # ------------------------------------------------------------------
    if isinstance(sub_exp, MaxCompressionSingleSubExperiment):
        opt = MaxCompressionSingleTaskBaseline(M=M, tasks=inference_tasks)
        return DirectCsiAdapter(optimizer=opt, **common)

    if isinstance(sub_exp, NoCompressionSingleSubExperiment):
        opt = NoCompressionSingleTaskBaseline(M=M, tasks=inference_tasks)
        return DirectCsiAdapter(optimizer=opt, **common)

    if isinstance(sub_exp, UniformCompressionSingleSubExperiment):
        opt = UniformCompressionSingleTaskBaseline(M=M, tasks=inference_tasks)
        return DirectCsiAdapter(optimizer=opt, **common)

    if isinstance(sub_exp, EstimatedCsiSingleSubExperiment):
        num_links = M - 1
        estimator = _build_external_estimator(sub_exp.channel_estimator, num_links)
        opt = EstimatedCSICompressionSingleTaskBaseline(M=M, tasks=inference_tasks)
        return EstimatedAdapter(optimizer=opt, estimator=estimator, **common)

    # ------------------------------------------------------------------
    # Multi-task baselines
    # ------------------------------------------------------------------
    if isinstance(sub_exp, MaxCompressionMultiSubExperiment):
        opt = MaxCompressionMultiTaskBaseline(M=M, tasks=inference_tasks)
        return DirectCsiAdapter(optimizer=opt, **common)

    if isinstance(sub_exp, NoCompressionMultiSubExperiment):
        opt = NoCompressionMultiTaskBaseline(M=M, tasks=inference_tasks)
        return DirectCsiAdapter(optimizer=opt, **common)

    if isinstance(sub_exp, StaticEqualShareSubExperiment):
        opt = StaticEqualShareMultiTaskBaseline(M=M, tasks=inference_tasks)
        return DirectCsiAdapter(optimizer=opt, **common)

    if isinstance(sub_exp, ProportionalResourceSubExperiment):
        opt = ProportionalResourceAllocationMultiTaskBaseline(
            M=M, tasks=inference_tasks
        )
        return DirectCsiAdapter(optimizer=opt, **common)

    if isinstance(sub_exp, StrictPriorityGreedySubExperiment):
        opt = StrictPriorityGreedyMultiTaskBaseline(M=M, tasks=inference_tasks)
        return DirectCsiAdapter(optimizer=opt, **common)

    if isinstance(sub_exp, DecoupledDescentSubExperiment):
        num_links = M - 1
        estimator = _build_external_estimator(sub_exp.channel_estimator, num_links)
        opt = DecoupledEqualSplitStochasticDescentMultiTaskBaseline(
            M=M, tasks=inference_tasks, mu=sub_exp.mu, epsilon=sub_exp.epsilon
        )
        return DualEstimatedAdapter(optimizer=opt, estimator=estimator, **common)

    if isinstance(sub_exp, QueueProportionalSubExperiment):
        num_links = M - 1
        estimator = _build_external_estimator(sub_exp.channel_estimator, num_links)
        opt = QueueProportionalHeuristicMultiTaskBaseline(
            M=M, tasks=inference_tasks, mu=sub_exp.mu, epsilon=sub_exp.epsilon
        )
        return DualEstimatedAdapter(optimizer=opt, estimator=estimator, **common)

    if isinstance(sub_exp, HistoricalAverageCESubExperiment):
        num_links = M - 1
        estimator = _build_external_estimator(sub_exp.channel_estimator, num_links)
        opt = HistoricalAverageCertaintyEquivalenceMultiTaskBaseline(
            M=M, tasks=inference_tasks
        )
        return EstimatedAdapter(optimizer=opt, estimator=estimator, **common)

    raise ValueError(
        f"build_adapter: unrecognised sub-experiment type '{getattr(sub_exp, 'type', type(sub_exp))}'. "
        "Expected one of the OptSubExperiment variants."
    )
