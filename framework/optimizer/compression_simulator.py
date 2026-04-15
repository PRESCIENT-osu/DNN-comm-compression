"""GPU-native compression simulators for the Stein gradient oracle.

Provides simulate-then-reconstruct functions that model the information loss of
each deployed compression scheme without the CPU round-trip cost of the actual
compressors.  Used exclusively inside the Stein oracle loop where compress and
decompress are called hundreds of times per slot.

All functions operate on tensors in-place on their current device — no `.cpu()`
calls, no numpy.  ``torch.kthvalue`` is used instead of ``torch.quantile`` to
avoid PyTorch's internal CPU fallback for large tensors.

Supported schemes
-----------------
- ``topk``         — per-sample top-k magnitude sparsification
- ``quantization`` — per-tensor AbsMax quantization to the nearest discrete level
- ``llmint8``      — mixed-precision: outlier fraction in fp16/int8,
                     remaining values per-row AbsMax INT8/INT4

``build_compress_fn`` is the public entry point used by ``simulation_factory``.
It returns a ``(tensor, eta) -> tensor`` callable keyed on what the
``CompressionMapper`` would select for a given (link_id, pipeline_id) pair.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from framework.optimizer.compression_mapper import CompressionMapper

logger = logging.getLogger(__name__)

# Valid quantization rates and their bit-widths, sorted descending (least
# compressed first) to match the framework compressor's rate mapping.
_QUAN_RATES: list[float] = [0.5, 0.25, 0.125, 0.0625]  # fp16, int8, int4, int2
_QUAN_BITS: dict[float, int] = {0.5: 16, 0.25: 8, 0.125: 4, 0.0625: 2}


# ---------------------------------------------------------------------------
# Per-sample top-k sparsification
# ---------------------------------------------------------------------------


def topk_sparsify_per_sample(x: torch.Tensor, eta: float) -> torch.Tensor:
    """Simulate TopK compression: zero out non-top-k elements.

    Dispatch is shape-based to match ``TopK.compress`` / ``TopK.decompress``:

    - 3-D tensors ``[B, L, D]``: top-k applied **per token** — each of the
      ``B×L`` token vectors independently retains its ``k`` largest-magnitude
      hidden dimensions.
    - All other shapes: top-k applied **per sample** — each sample's
      activations are flattened and ranked globally.

    Args:
        x: Input activation tensor of any shape with batch on dim 0.
        eta: Fraction of elements to keep, in (0, 1].  1.0 returns x unchanged.

    Returns:
        Sparsified tensor with same shape as ``x``, on the same device.
    """
    if eta >= 1.0:
        return x
    if eta <= 0.0:
        return torch.zeros_like(x)

    if x.ndim == 3:
        # Per-token path for transformer hidden states [B, L, D]
        B, L, D = x.shape
        flat = x.reshape(B * L, D)
        k = max(1, int(eta * D))
        thresh_rank = D - k + 1
        threshold = torch.kthvalue(flat.abs(), thresh_rank, dim=1, keepdim=True).values
        mask = flat.abs() >= threshold
        return (flat * mask).reshape(x.shape)

    # Per-sample path for CNN activations [B, C, H, W] etc.
    shape = x.shape
    flat = x.reshape(x.shape[0], -1)  # [B, D]
    D = flat.shape[1]
    k = max(1, int(eta * D))
    thresh_rank = D - k + 1
    threshold = torch.kthvalue(flat.abs(), thresh_rank, dim=1, keepdim=True).values
    mask = flat.abs() >= threshold  # [B, D] bool, stays on device
    return (flat * mask).reshape(shape)


# ---------------------------------------------------------------------------
# Quantization (discrete levels)
# ---------------------------------------------------------------------------


def _snap_quan_rate(eta: float) -> float:
    """Snap eta to the nearest valid quantization rate."""
    return min(_QUAN_RATES, key=lambda r: abs(r - eta))


def _absmax_quantize_dequantize(
    x: torch.Tensor,
    bits: int,
    stochastic: bool = False,
) -> torch.Tensor:
    """Quantize and immediately dequantize x using per-tensor AbsMax scaling.

    Args:
        x: Floating-point tensor, any shape.
        bits: Target bit-width (16, 8, 4, or 2).
        stochastic: If True, use stochastic rounding (for INT4/INT2).

    Returns:
        Dequantized tensor with same shape as ``x``, on the same device.
    """
    if bits == 16:
        return x.half().to(x.dtype)

    max_val = 2 ** (bits - 1) - 1  # 127 for INT8, 7 for INT4, 1 for INT2
    abs_max = x.abs().max().clamp(min=1e-8)
    scale = abs_max / max_val

    if stochastic:
        scaled = x / scale
        floored = scaled.floor()
        quantized = (floored + torch.bernoulli(scaled - floored)).clamp(
            -max_val, max_val
        )
    else:
        quantized = (x / scale).round().clamp(-max_val, max_val)

    return quantized * scale


def quantization_simulate(x: torch.Tensor, eta: float) -> torch.Tensor:
    """Simulate Quantization compression: quantize then dequantize.

    Snaps ``eta`` to the nearest valid rate ``{0.5, 0.25, 0.125, 0.0625}``
    corresponding to ``{fp16, int8, int4, int2}``, applies per-tensor AbsMax
    quantization, and reconstructs.  Matches the round-trip of
    ``Quantization.compress`` / ``Quantization.decompress``.

    Args:
        x: Input activation tensor, any shape.
        eta: Target compression rate; snapped to the nearest discrete level.

    Returns:
        Reconstructed tensor with same shape and device as ``x``.
    """
    rate = _snap_quan_rate(eta)
    bits = _QUAN_BITS[rate]
    stochastic = bits < 8
    return _absmax_quantize_dequantize(x, bits, stochastic=stochastic)


# ---------------------------------------------------------------------------
# LLMInt8 mixed-precision
# ---------------------------------------------------------------------------


def llmint8_simulate(
    x: torch.Tensor,
    rate: float,
    outlier_precision: str = "fp16",
    regular_precision: str = "int8",
) -> torch.Tensor:
    """Simulate LLMInt8 compression: split outliers/regular, quantize, reconstruct.

    Applies a global magnitude threshold (top ``rate`` fraction = outliers),
    quantizes outliers at ``outlier_precision`` and regular values per-row at
    ``regular_precision``, then reconstructs.  Matches the round-trip of
    ``LLMInt8.compress`` / ``LLMInt8.decompress``.

    All operations stay on the input tensor's device; ``torch.kthvalue`` is
    used instead of ``torch.quantile`` to avoid CPU fallback.

    Args:
        x: Input activation tensor, any shape.
        rate: Outlier fraction in (0, 1] (top-``rate`` elements by abs value).
        outlier_precision: ``"fp16"`` or ``"int8"``.
        regular_precision: ``"fp16"``, ``"int8"``, ``"int4"``, or ``"int2"``.

    Returns:
        Reconstructed tensor with same shape and device as ``x``.
    """
    orig_dtype = x.dtype
    if x.dtype != torch.float32:
        x = x.float()

    numel = x.numel()
    flat = x.reshape(-1)

    # Global outlier threshold via kthvalue (GPU-native, no CPU fallback).
    k_outlier = max(1, int(rate * numel))
    thresh_rank = numel - k_outlier + 1
    threshold = torch.kthvalue(flat.abs(), thresh_rank).values

    outlier_mask = flat.abs() >= threshold  # [numel] bool

    # --- Outlier group ---
    outlier_vals = flat[outlier_mask]
    if outlier_precision == "fp16":
        outlier_recon = outlier_vals.half().float()
    else:  # int8
        abs_max = outlier_vals.abs().max().clamp(min=1e-8)
        scale = abs_max / 127.0
        outlier_recon = (outlier_vals / scale).round().clamp(-127, 127) * scale

    # --- Regular group (per-row AbsMax) ---
    shape = x.shape
    regular = x.clone()
    regular.reshape(-1)[outlier_mask] = 0.0
    rows = regular.reshape(-1, shape[-1])  # [R, D_last]
    row_max = rows.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)

    if regular_precision == "fp16":
        regular_recon = rows.half().float()
    elif regular_precision == "int8":
        scale_r = row_max / 127.0
        regular_recon = (rows / scale_r).round().clamp(-127, 127) * scale_r
    elif regular_precision == "int4":
        scale_r = row_max / 7.0
        scaled = rows / scale_r
        floored = scaled.floor()
        stoch = (floored + torch.bernoulli(scaled - floored)).clamp(-7, 7)
        regular_recon = stoch * scale_r
    else:  # int2
        scale_r = row_max / 1.0
        scaled = rows / scale_r
        floored = scaled.floor()
        stoch = (floored + torch.bernoulli(scaled - floored)).clamp(-1, 1)
        regular_recon = stoch * scale_r

    # Reconstruct full tensor.
    out = regular_recon.reshape(shape).clone()
    out.reshape(-1)[outlier_mask] = outlier_recon

    return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_compress_fn(
    mapper: CompressionMapper,
    link_id: str,
    pipeline_id: str,
) -> Callable[[torch.Tensor, float], torch.Tensor]:
    """Build a compress-simulate callable for a specific (link, pipeline) pair.

    Inspects the mapper to determine which scheme would be selected for this
    link/pipeline pair, then returns a GPU-native ``(tensor, eta) -> tensor``
    that simulates the deployed compressor's round-trip information loss.

    The returned callable is scheme-aware: at call time it dispatches to
    ``topk_sparsify_per_sample``, ``quantization_simulate``, or
    ``llmint8_simulate`` depending on the mapper's decision for the given eta.

    Args:
        mapper: Initialised CompressionMapper for the experiment.
        link_id: Link identifier.
        pipeline_id: Pipeline identifier.

    Returns:
        Callable ``(tensor: Tensor, eta: float) -> Tensor``.
    """
    # Probe the mapper at eta=0.5 to determine which scheme it will use.
    # The scheme is fixed by allowed_methods and llmint8_mapping; only the
    # rate snapping differs per eta, so probing once is sufficient to pick
    # the right branch.
    probe = mapper.map(link_id, pipeline_id, 0.5)
    method = probe.method

    if method == "topk":

        def compress_fn(tensor: torch.Tensor, eta: float) -> torch.Tensor:
            return topk_sparsify_per_sample(tensor, eta)

    elif method == "quantization":

        def compress_fn(tensor: torch.Tensor, eta: float) -> torch.Tensor:
            return quantization_simulate(tensor, eta)

    elif method == "llmint8":
        # Precision values are fixed per link/pipeline from the mapping table.
        outlier_prec = probe.params.get("outlier_precision", "fp16")
        regular_prec = probe.params.get("regular_precision", "int8")

        def compress_fn(tensor: torch.Tensor, eta: float) -> torch.Tensor:
            return llmint8_simulate(tensor, eta, outlier_prec, regular_prec)

    else:
        logger.warning(
            "Unknown compression method '%s' for link %s pipeline %s; "
            "defaulting to topk_sparsify_per_sample",
            method,
            link_id,
            pipeline_id,
        )

        def compress_fn(tensor: torch.Tensor, eta: float) -> torch.Tensor:
            return topk_sparsify_per_sample(tensor, eta)

    return compress_fn


# ---------------------------------------------------------------------------
# Convenience: identity (no compression)
# ---------------------------------------------------------------------------


def identity_compress_fn(tensor: torch.Tensor, _eta: float) -> torch.Tensor:
    """No-op compress function used when eta_max == 1.0 and no compression is needed."""
    return tensor
