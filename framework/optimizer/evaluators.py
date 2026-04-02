"""Llama evaluation wrappers for SimulatedLlamaPipeline.

Provides concrete implementations of the ``LlamaEvaluator`` protocol defined
in ``models.llama.simulation.pipeline``.  Each wrapper delegates to an
external evaluator and adapts its interface to the protocol.

The adapter (``framework.optimizer.inference_optimizer_adapter``) instantiates
these from the optspec dataset config and injects them into
``SimulatedLlamaPipeline``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# External dependency path setup
# ---------------------------------------------------------------------------

_EXTERNAL_ROOT = Path(__file__).parents[2] / "external" / "Inference_Optimizer"
if str(_EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_ROOT))


# ---------------------------------------------------------------------------
# MMLU evaluator wrapper
# ---------------------------------------------------------------------------


class MMLUEvaluatorWrapper:
    """Wraps the external MMLUEvaluator for use as a LlamaEvaluator.

    Pre-loads the MMLU dataset subset at construction time so that repeated
    evaluations (as required by the Stein gradient oracle) do not re-tokenize
    or re-sample questions.

    Args:
        tokenizer: HuggingFace tokenizer for the Llama model.
        subjects: MMLU subjects to include in the evaluation.
        samples_per_subject: Number of questions per subject.
        seed: Random seed for deterministic question sampling.
        max_length: Maximum token length for prompt truncation.
        batch_size: Batch size for model inference during evaluation.
    """

    def __init__(
        self,
        tokenizer,
        subjects: list[str],
        samples_per_subject: int,
        seed: int = 0,
        max_length: int | None = 1024,
        batch_size: int = 8,
    ) -> None:
        from src.core.mmlu_eval import MMLUEvalConfig, MMLUEvaluator  # noqa: PLC0415

        cfg = MMLUEvalConfig(
            subjects=tuple(subjects),
            samples_per_subject=samples_per_subject,
            seed=seed,
            max_length=max_length,
            batch_size=batch_size,
        )
        self._evaluator = MMLUEvaluator(tokenizer=tokenizer, config=cfg)

    def evaluate(self, model: torch.nn.Module, device: torch.device) -> float:
        """Run MMLU evaluation and return top-1 accuracy in [0, 1].

        Args:
            model: The (compressed) Llama model to evaluate.
            device: Target device for inference.

        Returns:
            MMLU accuracy in [0, 1].
        """
        return self._evaluator.accuracy(model, device=device)


# ---------------------------------------------------------------------------
# WikiText perplexity evaluator (placeholder)
# ---------------------------------------------------------------------------


class WikiTextPerplexityEvaluator:
    """Evaluates a Llama model on WikiText-2 and returns a normalized metric.

    Returns ``exp(-perplexity / baseline_perplexity)`` so that the metric is
    in [0, 1] with higher = better (consistent with the MMLU accuracy scale).
    The baseline perplexity is the uncompressed model's perplexity measured
    at construction time.

    Args:
        tokenizer: HuggingFace tokenizer for the Llama model.
        dataset_path: Local path to the WikiText-2 dataset directory.
        max_length: Maximum sequence length per sample.
        stride: Sliding window stride for perplexity computation.
        batch_size: Batch size for inference.
        n_samples: Maximum number of tokens to evaluate over (None = full set).
    """

    def __init__(
        self,
        tokenizer,
        dataset_path: str | Path,
        max_length: int = 1024,
        stride: int = 512,
        batch_size: int = 1,
        n_samples: int | None = None,
    ) -> None:
        from datasets import load_from_disk  # noqa: PLC0415

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.n_samples = n_samples
        self._baseline_nll: float | None = None

        dataset = load_from_disk(str(dataset_path))
        # WikiText-2 test split text
        text = "\n\n".join(dataset["test"]["text"])
        encodings = tokenizer(text, return_tensors="pt")
        self._input_ids: torch.Tensor = encodings["input_ids"]

    def _compute_nll(self, model: torch.nn.Module, device: torch.device) -> float:
        """Compute mean negative log-likelihood over the token sequence."""
        seq_len = self._input_ids.size(1)
        max_len = self.max_length
        stride = self.stride
        limit = min(seq_len, self.n_samples) if self.n_samples else seq_len

        nlls: list[float] = []
        prev_end = 0
        for begin in range(0, limit, stride):
            end = min(begin + max_len, limit)
            input_ids = self._input_ids[:, begin:end].to(device)
            target_len = end - prev_end
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            # Use only the un-conditioned portion of the loss.
            loss = outputs.loss
            nlls.append(float(loss) * target_len)
            prev_end = end
            if end >= limit:
                break

        return sum(nlls) / limit if limit > 0 else 0.0

    def set_baseline(self, model: torch.nn.Module, device: torch.device) -> None:
        """Measure uncompressed model perplexity for normalization.

        Call once before starting the optimization loop.

        Args:
            model: Uncompressed Llama model (hooks at η=1.0).
            device: Inference device.
        """
        self._baseline_nll = self._compute_nll(model, device)
        logger.info(
            "WikiText baseline NLL=%.4f  perplexity=%.2f",
            self._baseline_nll,
            torch.exp(torch.tensor(self._baseline_nll)).item(),
        )

    def evaluate(self, model: torch.nn.Module, device: torch.device) -> float:
        """Return normalized perplexity metric in [0, 1] (higher = better).

        Args:
            model: The (compressed) Llama model.
            device: Inference device.

        Returns:
            ``exp(-nll / baseline_nll)`` if baseline is set; otherwise
            ``exp(-nll)`` clamped to [0, 1].
        """
        nll = self._compute_nll(model, device)
        if self._baseline_nll and self._baseline_nll > 0:
            ratio = nll / self._baseline_nll
        else:
            ratio = nll
        # Map to [0, 1]: lower nll relative to baseline → closer to 1.
        return float(torch.exp(torch.tensor(-max(0.0, ratio - 1.0))).item())
