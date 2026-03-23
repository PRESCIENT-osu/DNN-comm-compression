# Compression

Compression is applied only at inter-node boundaries. The sending node compresses the outgoing activation tensor; the receiving node decompresses it using the same method. Within a node, activations pass between partitions uncompressed.

All compressors are implemented in `framework/node/compressor.py`.

## Interface

```python
class Compressor(ABC):
    def compress(self, tensor: torch.Tensor, rate: float) -> bytes: ...
    def decompress(self, data: bytes, device: str) -> torch.Tensor: ...
```

`compress` returns a byte payload ready for transmission. `decompress` reconstructs the tensor on the specified device (`"cpu"`, `"cuda:0"`, etc.).

Use `get_compressor(method)` to instantiate the correct implementation:

```python
from framework.node.compressor import get_compressor
from framework.config.experiment_schema import CompressionMethod

compressor = get_compressor(CompressionMethod.TOPK)
compressed = compressor.compress(tensor, rate=0.1)
reconstructed = compressor.decompress(compressed, device="cpu")

# LLMInt8 requires precision kwargs
compressor = get_compressor(
    CompressionMethod.LLMINT8,
    outlier_precision="fp16",
    regular_precision="int8",
)
```

## Methods

### NoCompression

Pickles the tensor without modification. Used as the distributed baseline — establishes that partitioning itself does not degrade accuracy.

- `rate`: ignored

### TopK

Retains the top-k% activations by absolute magnitude per sample in the batch; all other values are zeroed. Positions are encoded as a per-sample bit-packed mask (numpy.packbits) and the retained values are stored in position order.

- `rate`: fraction of elements to retain per sample, e.g. `0.1` keeps the 10% largest-magnitude values
- Compression ratio ≈ `rate + 1/8` (values + mask); more efficient than index-based sparse storage at low rates

### RandomK

Retains a uniformly random k% of activations; all others are zeroed. Indices and values of sampled elements are transmitted.

- `rate`: fraction of elements to retain, e.g. `0.1` samples 10% at random
- Compression ratio ≈ `rate` (plus index overhead)

### Quantization

Uniform quantization — the `rate` value selects the bit-width:

| rate | bit-width | compression |
|------|-----------|-------------|
| 0.5 | FP16 | 2× |
| 0.25 | INT8 (symmetric AbsMax) | 4× |
| 0.125 | INT4 (stochastic rounding) | 8× |
| 0.0625 | INT2 | 16× |

Only these four rate values are valid; any other value raises `ValueError`.

```yaml
# experiment.yaml
links:
  - from: A
    to: B
    compression: quantization
    rate: 0.25   # INT8
```

### LLMInt8

Hybrid mixed-precision compressor inspired by Dettmers et al. (2022). Splits activations into two groups using a dynamic magnitude threshold:

- **Outliers** — the top `rate` fraction of elements by absolute value. Stored at `outlier_precision`.
- **Regular values** — everything else. Quantized row-wise (per-row AbsMax scaling) to `regular_precision`.

A 1-bit bitmask (numpy.packbits) marks outlier positions.

**`rate`** — outlier ratio, e.g. `0.01` designates the top 1% of elements as outliers.

**`outlier_precision`** — precision for outlier storage:
- `"fp16"` (default) — 2 bytes per outlier, minimal precision loss
- `"int8"` — 1 byte per outlier, global AbsMax scale

**`regular_precision`** — precision for regular-value storage:
- `"fp16"` — 2 bytes per element
- `"int8"` (default) — 1 byte per element, row-wise AbsMax scale
- `"int4"` — 4 bits per element, stochastic rounding
- `"int2"` — 2 bits per element

```yaml
# experiment.yaml
links:
  - from: A
    to: B
    compression: llmint8
    rate: 0.01
    outlier_precision: fp16
    regular_precision: int8
```

Precision fields are optional and default to `fp16`/`int8`. They are ignored for all other compression methods.

## Compression Ratio Reference

Approximate ratios relative to raw float32 at `rate=0.1`:

| Method | ratio |
|--------|-------|
| none | ~1.0 (pickle overhead) |
| topk | ~22.5% |
| randomk | ~40% |
| quantization (fp16) | 50% |
| quantization (int8) | 25% |
| quantization (int4) | 12.5% |
| quantization (int2) | 6.25% |
| llmint8 (fp16+int8, 1% outliers) | ~26% |

## Adding a New Method

1. Add a value to `CompressionMethod` in `framework/config/experiment_schema.py`
2. Implement `Compressor` in `framework/node/compressor.py`
3. Register it in `_REGISTRY` (or handle in `get_compressor` for methods with constructor params)
