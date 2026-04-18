from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from framework.datamodels.experiment import CompressionMethod

logger = logging.getLogger(__name__)


class Compressor(ABC):
    """Abstract base class for activation compressors.

    Each implementation compresses an outgoing activation tensor to bytes
    and decompresses received bytes back to a tensor.  Compression and
    decompression must be symmetric: a tensor compressed by an instance
    of a given subclass must be correctly decompressed by any other
    instance of the same subclass.
    """

    @abstractmethod
    def compress(self, tensor: torch.Tensor, rate: float) -> bytes:
        """Compress an activation tensor.

        Args:
            tensor: The activation tensor to compress.
            rate: Compression rate in (0, 1].  Interpretation is
                method-specific; see subclass documentation.

        Returns:
            Compressed byte payload suitable for transmission.
        """

    @abstractmethod
    def decompress(self, data: bytes, device: str) -> torch.Tensor:
        """Decompress a received byte payload back to a tensor.

        Args:
            data: Compressed bytes produced by the matching compress call.
            device: Target device for the reconstructed tensor
                (e.g. ``"cpu"`` or ``"cuda:0"``).

        Returns:
            Reconstructed activation tensor on the specified device.
        """


# ---------------------------------------------------------------------------
# NoCompression
# ---------------------------------------------------------------------------


class NoCompression(Compressor):
    """Pass-through compressor that serializes tensors without modification.

    Used as the baseline: activations are pickled and sent as-is.
    The rate parameter is ignored.
    """

    def compress(self, tensor: torch.Tensor, rate: float) -> bytes:
        return pickle.dumps({"tensor": tensor.cpu()})

    def decompress(self, data: bytes, device: str) -> torch.Tensor:
        return pickle.loads(data)["tensor"].to(device)


# ---------------------------------------------------------------------------
# TopK
# ---------------------------------------------------------------------------


class TopK(Compressor):
    """Sparse compressor retaining the top-k% activations by magnitude.

    Dispatch is shape-based:
    - 3-D tensors ``[B, L, D]`` (e.g. transformer hidden states): top-k is
      applied **per token** — each of the ``B×L`` token vectors independently
      retains its ``k`` largest-magnitude hidden dimensions.
    - All other shapes (e.g. ``[B, C, H, W]`` for CNNs): top-k is applied
      **per sample** — each sample's activations are flattened and ranked
      globally.

    In both cases the selected positions are encoded as a bit-packed mask
    (numpy.packbits, 8 positions per byte) and the corresponding values are
    stored in position order.

    Compression ratio ≈ rate + 1/8  (values + mask).
    e.g. rate=0.1 → ~22.5% of original size.

    rate controls the fraction of elements to retain,
    e.g. 0.1 keeps the 10% largest-magnitude activations.
    """

    def compress(self, tensor: torch.Tensor, rate: float) -> bytes:
        if tensor.ndim == 3:
            return self._compress_per_token(tensor, rate)
        return self._compress_per_sample(tensor, rate)

    def decompress(self, data: bytes, device: str) -> torch.Tensor:
        payload = pickle.loads(data)
        if len(payload["shape"]) == 3:
            return self._decompress_per_token(payload, device)
        return self._decompress_per_sample(payload, device)

    # ------------------------------------------------------------------
    # Per-sample (CNN) path — original implementation
    # ------------------------------------------------------------------

    def _compress_per_sample(self, tensor: torch.Tensor, rate: float) -> bytes:
        original_shape = tensor.shape
        batch_size = original_shape[0]
        reshaped = tensor.reshape(batch_size, -1)
        elements_per_sample = reshaped.shape[1]
        k = max(1, int(elements_per_sample * rate))

        _, top_indices = torch.topk(reshaped.abs(), k, dim=1)

        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(1, top_indices, True)

        sorted_indices = torch.sort(top_indices, dim=1)[0]
        sorted_values = torch.gather(reshaped, 1, sorted_indices)

        mask_np = mask.cpu().numpy().astype(np.uint8)
        packed_mask = np.packbits(mask_np, axis=1)

        payload = {
            "values": sorted_values.cpu().to(torch.float16).numpy()
            if tensor.dtype == torch.bfloat16
            else sorted_values.cpu().numpy(),
            "packed_mask": packed_mask,
            "shape": original_shape,
            "elements_per_sample": elements_per_sample,
            "dtype": tensor.dtype,
        }
        return pickle.dumps(payload)

    def _decompress_per_sample(self, payload: dict, device: str) -> torch.Tensor:
        values = torch.from_numpy(payload["values"]).to(device)
        shape = payload["shape"]
        elements_per_sample = payload["elements_per_sample"]
        batch_size = shape[0]

        mask_np = np.unpackbits(payload["packed_mask"], axis=1)[:, :elements_per_sample]
        mask = torch.from_numpy(mask_np).bool().to(device)

        reshaped = torch.zeros(
            batch_size, elements_per_sample, device=device, dtype=values.dtype
        )
        reshaped[mask] = values.flatten()
        return reshaped.reshape(shape).to(payload.get("dtype", torch.float32))

    # ------------------------------------------------------------------
    # Per-token (transformer) path — for [B, L, D] hidden states
    # ------------------------------------------------------------------

    def _compress_per_token(self, tensor: torch.Tensor, rate: float) -> bytes:
        original_shape = tensor.shape  # [B, L, D]
        B, L, D = original_shape
        reshaped = tensor.reshape(B * L, D)  # [B*L, D]
        k = max(1, int(D * rate))

        _, top_indices = torch.topk(reshaped.abs(), k, dim=1)

        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(1, top_indices, True)

        sorted_indices = torch.sort(top_indices, dim=1)[0]
        sorted_values = torch.gather(reshaped, 1, sorted_indices)

        mask_np = mask.cpu().numpy().astype(np.uint8)
        packed_mask = np.packbits(mask_np, axis=1)

        payload = {
            "values": sorted_values.cpu().to(torch.float16).numpy()
            if tensor.dtype == torch.bfloat16
            else sorted_values.cpu().numpy(),
            "packed_mask": packed_mask,
            "shape": original_shape,
            "elements_per_sample": D,
            "dtype": tensor.dtype,
        }
        return pickle.dumps(payload)

    def _decompress_per_token(self, payload: dict, device: str) -> torch.Tensor:
        values = torch.from_numpy(payload["values"]).to(device)
        shape = payload["shape"]  # [B, L, D]
        B, L, D = shape

        mask_np = np.unpackbits(payload["packed_mask"], axis=1)[:, :D]
        mask = torch.from_numpy(mask_np).bool().to(device)

        reshaped = torch.zeros(B * L, D, device=device, dtype=values.dtype)
        reshaped[mask] = values.flatten()
        return reshaped.reshape(shape).to(payload.get("dtype", torch.float32))


# ---------------------------------------------------------------------------
# RandomK
# ---------------------------------------------------------------------------


class RandomK(Compressor):
    """Sparse compressor that retains a random k% of activations.

    Similar to TopK but the retained elements are chosen uniformly at
    random rather than by magnitude.  Only the indices and values of the
    sampled elements are transmitted.

    rate controls the fraction of elements to retain, e.g. 0.1 samples
    10% of activations at random.
    """

    def compress(self, tensor: torch.Tensor, rate: float) -> bytes:
        flat = tensor.flatten()
        k = max(1, int(flat.numel() * rate))
        indices = torch.randperm(flat.numel())[:k]
        values = flat[indices]
        payload = {
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "numel": flat.numel(),
            "indices": indices.cpu().numpy(),
            "values": values.cpu().numpy(),
        }
        return pickle.dumps(payload)

    def decompress(self, data: bytes, device: str) -> torch.Tensor:
        payload = pickle.loads(data)
        flat = torch.zeros(payload["numel"], dtype=payload["dtype"])
        indices = torch.from_numpy(payload["indices"])
        values = torch.from_numpy(payload["values"])
        flat[indices] = values
        return flat.reshape(payload["shape"]).to(device)


# ---------------------------------------------------------------------------
# Quantization — private sub-quantizers
# ---------------------------------------------------------------------------


class _QuanFP16:
    """FP16 quantizer: cast FP32 → FP16 and back.  Compression ratio: 2×."""

    def compress(self, tensor: torch.Tensor) -> dict[str, Any]:
        return {
            "method": "quan_fp16",
            "values": tensor.half().cpu().numpy(),
            "shape": tensor.shape,
        }

    def decompress(self, data: dict[str, Any], device: str) -> torch.Tensor:
        return (
            torch.from_numpy(data["values"]).float().to(device).reshape(data["shape"])
        )


class _QuanInt8:
    """INT8 symmetric quantizer.  Scale = abs_max / 127.  Compression ratio: 4×."""

    def compress(self, tensor: torch.Tensor) -> dict[str, Any]:
        abs_max = tensor.abs().max()
        scale = float(abs_max) / 127.0 if abs_max > 0 else 1.0
        quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
        return {
            "method": "quan_int8",
            "values": quantized.cpu().numpy(),
            "scale": scale,
            "shape": tensor.shape,
        }

    def decompress(self, data: dict[str, Any], device: str) -> torch.Tensor:
        values = torch.from_numpy(data["values"]).to(device)
        return (values.float() * data["scale"]).reshape(data["shape"])


class _QuanInt4:
    """4-bit symmetric quantizer with stochastic rounding.  Compression ratio: 8×.

    Scale = abs_max / 7.  Two 4-bit values packed per uint8 byte.
    """

    def compress(self, tensor: torch.Tensor) -> dict[str, Any]:
        abs_max = tensor.abs().max()
        scale = float(abs_max) / 7.0 if abs_max > 0 else 1.0
        x = tensor / scale
        floor_x = x.floor()
        quantized = (
            (floor_x + torch.bernoulli(x - floor_x).to(x.device))
            .clamp(-7, 7)
            .to(torch.int8)
        )
        shifted = (quantized + 7).to(torch.uint8)
        flat = shifted.flatten()
        n = flat.numel()
        padding = (2 - n % 2) % 2
        if padding:
            flat = torch.cat(
                [flat, torch.zeros(padding, dtype=torch.uint8, device=flat.device)]
            )
        pairs = flat.reshape(-1, 2).cpu().numpy()
        packed = ((pairs[:, 0] << 4) | (pairs[:, 1] & 0x0F)).astype(np.uint8)
        return {
            "method": "quan_int4",
            "packed_values": packed,
            "scale": scale,
            "shape": tensor.shape,
            "n_elements": n,
            "padding": padding,
        }

    def decompress(self, data: dict[str, Any], device: str) -> torch.Tensor:
        packed = torch.from_numpy(data["packed_values"]).to(device)
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        unpacked = torch.stack([high, low], dim=1).flatten()
        if data["padding"]:
            unpacked = unpacked[: -data["padding"]]
        quantized = unpacked.to(torch.int8) - 7
        return (quantized.float() * data["scale"]).reshape(data["shape"])


class _QuanInt2:
    """2-bit symmetric quantizer.  Compression ratio: 16×.

    Scale = abs_max.  Four 2-bit values packed per uint8 byte.
    """

    def compress(self, tensor: torch.Tensor) -> dict[str, Any]:
        abs_max = tensor.abs().max()
        scale = float(abs_max) if abs_max > 0 else 1.0
        quantized = (tensor / scale).round().clamp(-1, 1).to(torch.int8)
        shifted = (quantized + 1).to(torch.uint8)
        flat = shifted.flatten()
        n = flat.numel()
        padding = (4 - n % 4) % 4
        if padding:
            flat = torch.cat(
                [flat, torch.zeros(padding, dtype=torch.uint8, device=flat.device)]
            )
        quads = flat.reshape(-1, 4).cpu().numpy()
        packed = (
            (quads[:, 0] << 6) | (quads[:, 1] << 4) | (quads[:, 2] << 2) | quads[:, 3]
        ).astype(np.uint8)
        return {
            "method": "quan_int2",
            "packed_values": packed,
            "scale": scale,
            "shape": tensor.shape,
            "n_elements": n,
            "padding": padding,
        }

    def decompress(self, data: dict[str, Any], device: str) -> torch.Tensor:
        packed = torch.from_numpy(data["packed_values"]).to(device)
        b0 = (packed >> 6) & 0x03
        b1 = (packed >> 4) & 0x03
        b2 = (packed >> 2) & 0x03
        b3 = packed & 0x03
        unpacked = torch.stack([b0, b1, b2, b3], dim=1).flatten()
        if data["padding"]:
            unpacked = unpacked[: -data["padding"]]
        quantized = unpacked.to(torch.int8) - 1
        return (quantized.float() * data["scale"]).reshape(data["shape"])


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

_QUAN_VALID_RATES = {0.5, 0.25, 0.125, 0.0625}


class Quantization(Compressor):
    """Uniform quantization compressor.  Bit-width is selected by rate.

    The rate encodes the desired bit-width:

        rate    bit-width   method   compression
        ------  ---------   ------   -----------
        0.5     FP16        fp16     2×
        0.25    INT8        int8     4×
        0.125   INT4        int4     8×
        0.0625  INT2        int2     16×

    Only these four rate values are accepted; any other value raises
    ValueError.  Smaller rate = more aggressive compression.
    """

    def __init__(self) -> None:
        self._fp16 = _QuanFP16()
        self._int8 = _QuanInt8()
        self._int4 = _QuanInt4()
        self._int2 = _QuanInt2()
        self._sub: dict[float, Any] = {
            0.5: self._fp16,
            0.25: self._int8,
            0.125: self._int4,
            0.0625: self._int2,
        }

    def _select(self, rate: float) -> Any:
        for valid_rate, sub in self._sub.items():
            if abs(rate - valid_rate) < 1e-9:
                return sub
        raise ValueError(
            f"Quantization rate must be one of "
            f"{sorted(_QUAN_VALID_RATES, reverse=True)}, got {rate}"
        )

    def _dispatch(self, method_tag: str) -> Any:
        return {
            "quan_fp16": self._fp16,
            "quan_int8": self._int8,
            "quan_int4": self._int4,
            "quan_int2": self._int2,
        }.get(method_tag, self._fp16)

    def compress(self, tensor: torch.Tensor, rate: float) -> bytes:
        return pickle.dumps(self._select(rate).compress(tensor))

    def decompress(self, data: bytes, device: str) -> torch.Tensor:
        payload = pickle.loads(data)
        return self._dispatch(payload["method"]).decompress(payload, device)


# ---------------------------------------------------------------------------
# LLMInt8
# ---------------------------------------------------------------------------


class LLMInt8(Compressor):
    """Hybrid mixed-precision compressor inspired by Dettmers et al. (2022).

    Splits activations into two groups based on a dynamic magnitude threshold:

    - **Outliers** — the top ``rate`` fraction of elements by absolute value.
      Stored at ``outlier_precision`` (``"fp16"`` or ``"int8"``).
    - **Regular values** — everything else.  Quantized row-wise using
      per-row AbsMax scaling to ``regular_precision``
      (``"fp16"``, ``"int8"``, ``"int4"``, or ``"int2"``).

    A 1-bit bitmask (numpy.packbits) separates the two groups.

    Args:
        outlier_precision: Precision for outlier storage. ``"fp16"`` or ``"int8"``.
        regular_precision: Precision for regular value storage.
            ``"fp16"``, ``"int8"``, ``"int4"``, or ``"int2"``.
    """

    _VALID_OUTLIER = {"fp16", "int8"}
    _VALID_REGULAR = {"fp16", "int8", "int4", "int2"}

    def __init__(
        self,
        outlier_precision: str = "fp16",
        regular_precision: str = "int8",
    ) -> None:
        if outlier_precision not in self._VALID_OUTLIER:
            raise ValueError(
                f"outlier_precision must be one of {self._VALID_OUTLIER}, "
                f"got '{outlier_precision}'"
            )
        if regular_precision not in self._VALID_REGULAR:
            raise ValueError(
                f"regular_precision must be one of {self._VALID_REGULAR}, "
                f"got '{regular_precision}'"
            )
        self._outlier_precision = outlier_precision
        self._regular_precision = regular_precision

    def compress(self, tensor: torch.Tensor, rate: float) -> bytes:
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        shape = tensor.shape
        numel = tensor.numel()

        nonfinite_mask = ~torch.isfinite(tensor)
        n_nonfinite = nonfinite_mask.sum().item()
        if n_nonfinite > 0:
            logger.warning(
                "LLMInt8: %d/%d non-finite values in tensor; "
                "filtering before quantile and including as outliers",
                n_nonfinite,
                numel,
            )

        abs_flat = tensor.abs().reshape(-1)
        if n_nonfinite > 0:
            finite_abs = abs_flat[torch.isfinite(abs_flat)]
            n_finite = finite_abs.numel()
            nonfinite_frac = n_nonfinite / numel
            if n_finite == 0 or rate <= nonfinite_frac:
                threshold = float("inf")
            else:
                n_extra = int(rate * numel) - n_nonfinite
                adjusted_rate = min(n_extra / n_finite, 1.0)
                threshold = torch.quantile(
                    finite_abs, float(1.0 - adjusted_rate)
                ).item()
        else:
            threshold = torch.quantile(abs_flat, float(1.0 - rate)).item()
        mask_bool = tensor.abs() > threshold
        if n_nonfinite > 0:
            mask_bool = mask_bool | nonfinite_mask
            tensor = tensor.clone()
            tensor.clamp_(-65504.0, 65504.0)

        outlier_raw = torch.masked_select(tensor, mask_bool)
        outlier_scale: float | None = None
        if self._outlier_precision == "fp16":
            outlier_values = outlier_raw.half()
        else:  # int8
            abs_max = outlier_raw.abs().max()
            outlier_scale = float(abs_max) / 127.0 if abs_max > 0 else 1.0
            outlier_values = (
                (outlier_raw / outlier_scale).round().clamp(-127, 127).to(torch.int8)
            )

        tensor_regular = tensor.clone()
        tensor_regular.masked_fill_(mask_bool, 0.0)
        flattened = tensor_regular.view(-1, shape[-1])
        row_abs_max = flattened.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)

        regular_padding = 0
        if self._regular_precision == "fp16":
            scales = row_abs_max
            quantized_regular = flattened.half()
        elif self._regular_precision == "int8":
            scales = row_abs_max / 127.0
            quantized_regular = (
                (flattened / scales).round().clamp(-127, 127).to(torch.int8)
            )
        elif self._regular_precision == "int4":
            scales = row_abs_max / 7.0
            x = flattened / scales
            floor_x = x.floor()
            quantized = (
                (floor_x + torch.bernoulli(x - floor_x).to(x.device))
                .clamp(-7, 7)
                .to(torch.int8)
            )
            shifted = (quantized + 7).to(torch.uint8).flatten()
            if shifted.numel() % 2 != 0:
                shifted = torch.cat(
                    [shifted, torch.zeros(1, dtype=torch.uint8, device=shifted.device)]
                )
                regular_padding = 1
            quantized_regular = (shifted[0::2] << 4) | (shifted[1::2] & 0x0F)
        else:  # int2
            scales = row_abs_max
            x = flattened / scales
            quantized = x.round().clamp(-1, 1).to(torch.int8)
            shifted = (quantized + 1).to(torch.uint8).flatten()
            regular_padding = (4 - shifted.numel() % 4) % 4
            if regular_padding:
                shifted = torch.cat(
                    [
                        shifted,
                        torch.zeros(
                            regular_padding, dtype=torch.uint8, device=shifted.device
                        ),
                    ]
                )
            quantized_regular = (
                (shifted[0::4] << 6)
                | (shifted[1::4] << 4)
                | (shifted[2::4] << 2)
                | shifted[3::4]
            )

        mask_np = mask_bool.reshape(-1).cpu().numpy().astype(np.uint8)
        packed_mask = np.packbits(mask_np)

        payload: dict[str, Any] = {
            "packed_mask": packed_mask,
            "outlier_values": outlier_values.cpu().numpy(),
            "outlier_scale": outlier_scale,
            "main_values": quantized_regular.cpu().numpy(),
            "main_scales": scales.cpu().numpy(),
            "regular_padding": regular_padding,
            "shape": shape,
            "numel": numel,
            "outlier_precision": self._outlier_precision,
            "regular_precision": self._regular_precision,
        }
        return pickle.dumps(payload)

    def decompress(self, data: bytes, device: str) -> torch.Tensor:
        payload = pickle.loads(data)
        shape = payload["shape"]
        numel = payload["numel"]
        reg_prec = payload["regular_precision"]
        o_prec = payload["outlier_precision"]

        mask_flat = np.unpackbits(payload["packed_mask"])[:numel]
        mask = torch.from_numpy(mask_flat.copy()).view(shape).to(device).bool()

        main_scales = torch.from_numpy(payload["main_scales"].copy()).to(device).float()

        if reg_prec == "fp16":
            restored = (
                torch.from_numpy(payload["main_values"].copy())
                .to(device)
                .float()
                .view(shape)
            )
        elif reg_prec == "int8":
            main_vals = (
                torch.from_numpy(payload["main_values"].copy()).to(device).float()
            )
            restored = (main_vals.view(-1, shape[-1]) * main_scales).view(shape)
        elif reg_prec == "int4":
            packed = torch.from_numpy(payload["main_values"].copy()).to(device)
            high = (packed >> 4) & 0x0F
            low = packed & 0x0F
            unpacked = torch.stack([high, low], dim=1).flatten()
            if payload["regular_padding"]:
                unpacked = unpacked[: -payload["regular_padding"]]
            quantized = unpacked.to(torch.int8) - 7
            restored = (quantized.float().view(-1, shape[-1]) * main_scales).view(shape)
        else:  # int2
            packed = torch.from_numpy(payload["main_values"].copy()).to(device)
            p0 = (packed >> 6) & 0x03
            p1 = (packed >> 4) & 0x03
            p2 = (packed >> 2) & 0x03
            p3 = packed & 0x03
            unpacked = torch.stack([p0, p1, p2, p3], dim=1).flatten()
            if payload["regular_padding"]:
                unpacked = unpacked[: -payload["regular_padding"]]
            quantized = unpacked.to(torch.int8) - 1
            restored = (quantized.float().view(-1, shape[-1]) * main_scales).view(shape)

        if payload["outlier_values"].size > 0:
            if o_prec == "fp16":
                outlier_vals = (
                    torch.from_numpy(payload["outlier_values"].copy())
                    .to(device)
                    .float()
                )
            else:  # int8
                outlier_vals = (
                    torch.from_numpy(payload["outlier_values"].copy())
                    .to(device)
                    .float()
                    * payload["outlier_scale"]
                )
            restored.masked_scatter_(mask, outlier_vals)

        return restored


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[CompressionMethod, type[Compressor]] = {
    CompressionMethod.NONE: NoCompression,
    CompressionMethod.TOPK: TopK,
    CompressionMethod.RANDOMK: RandomK,
    CompressionMethod.QUANTIZATION: Quantization,
}


def get_compressor(
    method: CompressionMethod,
    *,
    outlier_precision: str = "fp16",
    regular_precision: str = "int8",
) -> Compressor:
    """Instantiate a compressor for the given compression method.

    Args:
        method: The compression method to use.
        outlier_precision: Outlier precision for LLMInt8 (``"fp16"`` or ``"int8"``).
            Ignored for all other methods.
        regular_precision: Regular-value precision for LLMInt8
            (``"fp16"``, ``"int8"``, ``"int4"``, or ``"int2"``).
            Ignored for all other methods.

    Returns:
        A Compressor instance for the specified method.

    Raises:
        ValueError: If the method is not supported or precision values are invalid.
    """
    if method == CompressionMethod.LLMINT8:
        return LLMInt8(
            outlier_precision=outlier_precision,
            regular_precision=regular_precision,
        )
    cls = _REGISTRY.get(method)
    if cls is None:
        raise ValueError(f"Unsupported compression method: {method}")
    return cls()
