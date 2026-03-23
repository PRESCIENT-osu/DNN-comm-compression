"""Partition a Llama-3.1-8B model into 3 sequential nn.Module stages.

Partition layout:
  p1 — embed_tokens + layers[0..10]     in: int64 [B, L]       out: float [B, L, 4096]
  p2 — layers[11..21]                   in: float [B, L, 4096]  out: float [B, L, 4096]
  p3 — layers[22..31] + norm + lm_head  in: float [B, L, 4096]  out: float [B, L, 128256]

Each partition reconstructs position_ids from input shape. Passes attention_mask=None so
LlamaSdpaAttention uses is_causal=True for prefill (q_len > 1). Prefill-only: no KV cache.

Saved with torch.save(module, path) — NOT TorchScript. At load time the LlamaPartition*
classes must be importable (handled by llama_server.py importing this module before torch.load).

Usage:
  python models/llama/partition_llama.py \\
    --model meta-llama/Llama-3.1-8B \\
    --output-dir models/llama/.partitions \\
    --dtype bf16 \\
    [--verify]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


# ---------------------------------------------------------------------------
# Partition modules
# ---------------------------------------------------------------------------


class LlamaPartition1(nn.Module):
    """embed_tokens + decoder layers[0..10].

    Args:
        embed_tokens: Token embedding module from the base model.
        layers: ModuleList of the first 11 decoder layers.
    """

    def __init__(self, embed_tokens: nn.Module, layers: nn.ModuleList) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = layers

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens and run the first 11 decoder layers.

        Args:
            input_ids: Token IDs of shape [B, L].

        Returns:
            Hidden states of shape [B, L, hidden_size].
        """
        hidden_states = self.embed_tokens(input_ids)
        B, L = input_ids.shape
        position_ids = (
            torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        )
        for layer in self.layers:
            out = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False,
            )
            hidden_states = out[0]
        return hidden_states


class LlamaPartition2(nn.Module):
    """Decoder layers[11..21].

    Args:
        layers: ModuleList of decoder layers 11 through 21 inclusive.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run 11 middle decoder layers.

        Args:
            hidden_states: Float tensor of shape [B, L, hidden_size].

        Returns:
            Hidden states of shape [B, L, hidden_size].
        """
        B, L, _ = hidden_states.shape
        position_ids = (
            torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        )
        for layer in self.layers:
            out = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False,
            )
            hidden_states = out[0]
        return hidden_states


class LlamaPartition3(nn.Module):
    """Decoder layers[22..31] + final norm + lm_head.

    Args:
        layers: ModuleList of the last 10 decoder layers.
        norm: Final RMSNorm module.
        lm_head: Linear projection to vocabulary size.
    """

    def __init__(
        self, layers: nn.ModuleList, norm: nn.Module, lm_head: nn.Module
    ) -> None:
        super().__init__()
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the final 10 decoder layers, norm, and project to vocab logits.

        Args:
            hidden_states: Float tensor of shape [B, L, hidden_size].

        Returns:
            Logits of shape [B, L, vocab_size].
        """
        B, L, _ = hidden_states.shape
        position_ids = (
            torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        )
        for layer in self.layers:
            out = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False,
            )
            hidden_states = out[0]
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_model(
    model_name: str, dtype: torch.dtype
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a Llama model and tokenizer from HuggingFace hub or local path.

    Args:
        model_name: HuggingFace repo ID or local directory path.
        dtype: Torch dtype to load weights in (bf16, fp16, fp32).

    Returns:
        Tuple of (model in eval mode, tokenizer).
    """
    logger.info("Loading tokenizer from %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading model %s in %s ...", model_name, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info("Model loaded: %d layers", len(model.model.layers))
    return model, tokenizer


# ---------------------------------------------------------------------------
# Partitioning and saving
# ---------------------------------------------------------------------------


def partition_and_save(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
) -> dict[str, Path]:
    """Split model into 3 partition modules and save to disk.

    Also saves the tokenizer alongside the partition files so that the
    data client can load it from the same directory without needing HF access.

    Args:
        model: Loaded Llama model in eval mode.
        tokenizer: Corresponding tokenizer.
        output_dir: Directory to write p1.pt, p2.pt, p3.pt and tokenizer/ into.

    Returns:
        Dict mapping partition name to saved file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = model.model.layers
    n = len(layers)
    cut1 = n // 3  # 10 for 32-layer model → layers 0..10 in p1 (11 layers)
    cut2 = 2 * n // 3  # 21 → layers 11..21 in p2 (11 layers), 22..31 in p3

    logger.info(
        "Partitioning %d layers: p1=layers[0:%d], p2=layers[%d:%d], p3=layers[%d:%d]",
        n,
        cut1 + 1,
        cut1 + 1,
        cut2 + 1,
        cut2 + 1,
        n,
    )

    p1 = LlamaPartition1(
        embed_tokens=model.model.embed_tokens,
        layers=nn.ModuleList(list(layers[: cut1 + 1])),
    )
    p2 = LlamaPartition2(
        layers=nn.ModuleList(list(layers[cut1 + 1 : cut2 + 1])),
    )
    p3 = LlamaPartition3(
        layers=nn.ModuleList(list(layers[cut2 + 1 :])),
        norm=model.model.norm,
        lm_head=model.lm_head,
    )

    saved: dict[str, Path] = {}
    for name, module in [("p1", p1), ("p2", p2), ("p3", p3)]:
        module.eval()
        path = output_dir / f"{name}.pt"
        torch.save(module, str(path))
        logger.info("Saved %s → %s", name, path)
        saved[name] = path

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    logger.info("Tokenizer saved → %s", tokenizer_dir)

    return saved


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_partitions(
    model: AutoModelForCausalLM,
    saved: dict[str, Path],
    seq_len: int = 16,
    batch_size: int = 2,
) -> None:
    """Run a dummy batch through the full model and chained partitions.

    Compares top-1 predicted token IDs (argmax of logits) at every position.
    Uses argmax agreement rather than numeric tolerance because FP16/BF16
    logits over a 128k vocab will have larger floating-point differences.

    Args:
        model: Original full model for reference output.
        saved: Dict mapping partition name to .pt file path.
        seq_len: Sequence length for the dummy input.
        batch_size: Batch size for the dummy input.

    Raises:
        RuntimeError: If top-1 token agreement is below 99%.
    """
    vocab_size = model.config.vocab_size
    dummy_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        expected_logits = model(dummy_ids).logits  # [B, L, V]
    expected_tokens = expected_logits.argmax(dim=-1)  # [B, L]

    parts = [
        torch.load(str(saved[f"p{i}"]), map_location="cpu", weights_only=False)
        for i in range(1, 4)
    ]
    x: torch.Tensor = dummy_ids
    with torch.no_grad():
        for part in parts:
            part.eval()
            x = part(x)
    actual_tokens = x.argmax(dim=-1)  # [B, L]

    agreement = (expected_tokens == actual_tokens).float().mean().item()
    if agreement >= 0.99:
        logger.info(
            "Verification passed — top-1 token agreement: %.1f%%", agreement * 100
        )
    else:
        raise RuntimeError(
            f"Verification failed — top-1 token agreement {agreement:.1%} < 99%"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Llama partition script."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; HF_TOKEN must be set manually

    parser = argparse.ArgumentParser(
        description="Partition Llama-3.1-8B into 3 sequential nn.Module stages."
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace repo ID or local model path (default: meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/llama/.partitions"),
        help="Directory to save partition .pt files and tokenizer/ (default: models/llama/.partitions)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Weight dtype (default: bf16)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify partitions by comparing top-1 token agreement with the full model",
    )
    args = parser.parse_args()

    dtype = _DTYPE_MAP[args.dtype]
    model, tokenizer = load_model(args.model, dtype)
    saved = partition_and_save(model, tokenizer, args.output_dir)

    if args.verify:
        verify_partitions(model, saved)

    logger.info(
        "Done. %d partition files + tokenizer saved to %s", len(saved), args.output_dir
    )


if __name__ == "__main__":
    main()
