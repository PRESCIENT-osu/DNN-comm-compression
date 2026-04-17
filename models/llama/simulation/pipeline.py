"""Simulated Llama pipeline for Stein gradient oracle accuracy evaluation.

Loads the full HF Llama model and installs per-link activation compression
hooks at the inter-node partition boundaries defined by the pipeline config.
No partition .pt files are required.  The full model's forward() handles all
position_ids, RoPE embeddings, and attention masking internally; hooks only
intercept and compress the hidden states flowing between nodes.

Partition boundary layer indices match partition_llama.py:
  p1 → layers[0 .. n//3]        →  post-forward hook at decoder layer n//3
  p2 → layers[n//3+1 .. 2*n//3] →  post-forward hook at decoder layer 2*n//3
  p3 → layers[2*n//3+1 .. n-1]  →  no outgoing hook (final partition)

The eta vector has one element per inter-node link (len(flow) - 1).
Element eta[i] is the compression ratio for link flow[i] → flow[i+1].

Compression is applied via injected ``compress_fns`` callables, one per link,
matching the scheme used by the deployed compressors on the nodes.

Evaluation strategy (MMLU accuracy or WikiText perplexity) is provided by
the adapter as a LlamaEvaluator callable, keeping this module dataset-agnostic.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn as nn

from framework.optimizer.compression_simulator import topk_sparsify_per_sample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Partition boundary index table
# ---------------------------------------------------------------------------

# Maps partition name → callable that computes the last decoder layer index
# for that partition given the total number of decoder layers n.
# These formulas replicate partition_llama.py exactly.
_PARTITION_LAST_LAYER: dict[str, Callable[[int], int]] = {
    "p1": lambda n: n // 3,
    "p2": lambda n: (2 * n) // 3,
}
# p3 has no outgoing link so it is not in the table.


# ---------------------------------------------------------------------------
# Evaluator protocol
# ---------------------------------------------------------------------------


class LlamaEvaluator(Protocol):
    """Protocol for task-specific accuracy/perplexity evaluation.

    Implementations are provided by the adapter (e.g. MMLUEvaluator wrapper
    or a WikiText perplexity evaluator) and injected at construction time.
    """

    def evaluate(self, model: torch.nn.Module, device: torch.device) -> float:
        """Run evaluation and return a scalar metric in [0, 1] (higher = better).

        For MMLU: returns classification accuracy.
        For WikiText: returns exp(-perplexity) or another normalized metric.
        """
        ...


# ---------------------------------------------------------------------------
# Simulation pipeline
# ---------------------------------------------------------------------------


class SimulatedLlamaPipeline:
    """Full-model Llama simulation with inter-node compression hooks.

    Loads the full HF Llama model and installs post-forward hooks at the
    decoder layer indices that correspond to the inter-node partition
    boundaries.  Because the full model's forward() runs all layers in a
    single pass, position_ids, RoPE embeddings, and the 4D causal mask are
    computed once and threaded internally — no manual threading between
    partition calls, no calling-convention mismatch.

    Each hook compresses the hidden state tensor ``[B, L, D]`` using the
    injected ``compress_fns[link_idx]`` callable, simulating the per-sample
    compression applied by the deployed compressor on the sending node.

    The evaluator is injected by the adapter to keep this module agnostic to
    the downstream task (MMLU, WikiText, etc.).

    Args:
        model_name: HuggingFace repo ID or local model directory.
        partitions: Mapping of node_id → list of partition names assigned to
            that node, e.g. ``{"A": ["p1"], "B": ["p2"], "C": ["p3"]}``.
        flow: Ordered list of node IDs, e.g. ``["A", "B", "C"]``.
        fast_evaluator: Evaluator used for accuracy_callable and the Stein
            gradient oracle (low sample count for speed).
        full_evaluator: Evaluator used for accuracy_callable_true (full
            evaluation set, called once per slot).
        compress_fns: One callable per inter-node link.  Each callable has
            signature ``(tensor: Tensor, eta: float) -> Tensor`` and simulates
            the deployed compressor's round-trip information loss for that link.
            If ``None``, defaults to ``topk_sparsify_per_sample`` for all links.
        device: Compute device.  Defaults to CUDA if available, else CPU.
        torch_dtype: Model weight dtype.  Defaults to float16 on CUDA.
    """

    def __init__(
        self,
        model_name: str,
        partitions: dict[str, list[str]],
        flow: list[str],
        fast_evaluator: LlamaEvaluator,
        full_evaluator: LlamaEvaluator,
        compress_fns: list[Callable[[torch.Tensor, float], torch.Tensor]] | None = None,
        device: torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.fast_evaluator = fast_evaluator
        self.full_evaluator = full_evaluator

        logger.info("Loading Llama model '%s' in %s ...", model_name, torch_dtype)
        from transformers import AutoModelForCausalLM  # lazy import

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Derive the decoder layer indices for each inter-node link.
        n_layers = len(self.model.model.layers)
        self._n_links: int = len(flow) - 1
        self._eta: list[float] = [1.0] * self._n_links
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        if compress_fns is None:
            self._compress_fns: list[Callable[[torch.Tensor, float], torch.Tensor]] = [
                topk_sparsify_per_sample for _ in range(self._n_links)
            ]
        else:
            if len(compress_fns) != self._n_links:
                raise ValueError(
                    f"Expected {self._n_links} compress_fns (one per link), "
                    f"got {len(compress_fns)}"
                )
            self._compress_fns = compress_fns

        for link_idx in range(self._n_links):
            sender = flow[link_idx]
            last_partition = partitions[sender][-1]
            if last_partition not in _PARTITION_LAST_LAYER:
                raise ValueError(
                    f"Partition '{last_partition}' is not a valid sending partition "
                    f"(it may be the final partition with no outgoing link). "
                    f"Supported: {list(_PARTITION_LAST_LAYER)}"
                )
            layer_idx = _PARTITION_LAST_LAYER[last_partition](n_layers)
            layer: nn.Module = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(self._make_hook(link_idx))
            self._handles.append(handle)
            logger.debug(
                "Installed post hook at decoder layer %d for link %s→%s (eta[%d])",
                layer_idx,
                flow[link_idx],
                flow[link_idx + 1],
                link_idx,
            )

    @property
    def n_links(self) -> int:
        """Number of inter-node links (= dimension of eta)."""
        return self._n_links

    def set_eta(self, eta: torch.Tensor) -> None:
        """Update compression ratios in-place.

        Args:
            eta: Tensor of shape (n_links,) with values in [0, 1].

        Raises:
            ValueError: If eta length does not match the number of links.
        """
        if eta.numel() != self._n_links:
            raise ValueError(
                f"Expected eta of length {self._n_links}, got {eta.numel()}"
            )
        self._eta = [float(torch.clamp(v, 0.0, 1.0).item()) for v in eta]

    def accuracy(self, eta: torch.Tensor, *, full: bool = False) -> float:
        """Evaluate task accuracy with given compression ratios.

        Uses the fast evaluator by default; pass ``full=True`` for the full
        evaluation set (accuracy_callable_true).

        Args:
            eta: Compression ratios of shape (n_links,), values in [0, 1].
            full: If True, use the full evaluator instead of the fast one.

        Returns:
            Scalar metric in [0, 1] (higher = better).

        Raises:
            ValueError: If eta length does not match the number of links.
        """
        self.set_eta(eta)
        evaluator = self.full_evaluator if full else self.fast_evaluator
        return evaluator.evaluate(self.model, self.device)

    def remove_hooks(self) -> None:
        """Remove all installed forward hooks."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def _make_hook(self, link_idx: int):
        def hook(
            module: nn.Module,
            args: tuple,
            output: torch.Tensor | tuple,
        ) -> torch.Tensor | tuple:
            # HF decoder layers return a tuple: (hidden_state, ...).
            # Compress only the hidden state (index 0).
            if isinstance(output, tuple):
                hidden = output[0]
                compressed = self._compress_fns[link_idx](hidden, self._eta[link_idx])
                return (compressed,) + output[1:]
            return self._compress_fns[link_idx](output, self._eta[link_idx])

        return hook
