"""Simulated Llama pipeline for Stein gradient oracle accuracy evaluation.

Loads the full HF Llama model and installs LLMActivationCompressor hooks at
the inter-node partition boundaries defined by the pipeline config.  No
partition .pt files are required.  The full model's forward() handles all
position_ids, RoPE embeddings, and attention masking internally; hooks only
intercept and compress the hidden states flowing between nodes.

Partition boundary layer indices match partition_llama.py:
  p1 → layers[0 .. n//3]        →  post-forward hook at decoder layer n//3
  p2 → layers[n//3+1 .. 2*n//3] →  post-forward hook at decoder layer 2*n//3
  p3 → layers[2*n//3+1 .. n-1]  →  no outgoing hook (final partition)

The eta vector has one element per inter-node link (len(flow) - 1).
Element eta[i] is the compression ratio for link flow[i] → flow[i+1].

Evaluation strategy (MMLU accuracy or WikiText perplexity) is provided by
the adapter as an Evaluator callable, keeping this module dataset-agnostic.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# External dependency path setup
# ---------------------------------------------------------------------------

_EXTERNAL_ROOT = Path(__file__).parents[3] / "external" / "Inference_Optimizer"
if str(_EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_ROOT))

from src.core.llm_compression import (  # noqa: E402
    ActivationCompressionConfig,
    LLMActivationCompressor,
)

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

    Loads the full HF Llama model and installs LLMActivationCompressor hooks
    at the decoder layer indices that correspond to the inter-node partition
    boundaries.  Because the full model's forward() runs all layers in a
    single pass, position_ids, RoPE embeddings, and the 4D causal mask are
    computed once and threaded internally — no manual threading between
    partition calls, no calling-convention mismatch.

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
        activation_strategy: Compression strategy passed to
            LLMActivationCompressor (``"topk_per_token"``, ``"magnitude"``,
            ``"random"``, or ``"quantization"``).
        device: Compute device.  Defaults to CUDA if available, else CPU.
        torch_dtype: Model weight dtype.  Defaults to bfloat16 on CUDA.
    """

    def __init__(
        self,
        model_name: str,
        partitions: dict[str, list[str]],
        flow: list[str],
        fast_evaluator: LlamaEvaluator,
        full_evaluator: LlamaEvaluator,
        activation_strategy: str = "topk_per_token",
        device: torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if torch_dtype is None:
            torch_dtype = (
                torch.bfloat16 if self.device.type == "cuda" else torch.float32
            )

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
        layer_indices: list[int] = []
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
            layer_indices.append(layer_idx)
            logger.debug(
                "Link %s→%s: last partition %s → hook at decoder layer %d",
                flow[link_idx],
                flow[link_idx + 1],
                last_partition,
                layer_idx,
            )

        compression_cfg = ActivationCompressionConfig(
            strategy=activation_strategy,
            layer_indices=layer_indices,
        )
        self.compressor = LLMActivationCompressor(self.model, compression_cfg)
        logger.info(
            "LLMActivationCompressor installed at layers %s (strategy=%s)",
            layer_indices,
            activation_strategy,
        )

    @property
    def n_links(self) -> int:
        """Number of inter-node links (= dimension of eta)."""
        return self._n_links

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
        if eta.numel() != self._n_links:
            raise ValueError(
                f"Expected eta of length {self._n_links}, got {eta.numel()}"
            )
        self.compressor.set_eta(eta)
        evaluator = self.full_evaluator if full else self.fast_evaluator
        return evaluator.evaluate(self.model, self.device)

    def remove_hooks(self) -> None:
        """Remove all installed forward hooks."""
        self.compressor.remove_hooks()
