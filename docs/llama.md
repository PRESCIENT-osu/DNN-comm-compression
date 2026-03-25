# Llama Experiments

This document covers everything specific to running distributed inference experiments with Llama-3.1-8B: partitioning the model, building the node image, and running WikiText-2 perplexity and MMLU accuracy sweeps.

## Overview

Llama-3.1-8B is split into three sequential pipeline stages following the early/middle/late layer structure used in `comp_analysis.py`. Each stage runs on a separate node. Activations (hidden states) are compressed at the two inter-node boundaries.

```
[orchestrator] → early node (p1) → middle node (p2) → late node (p3) → [orchestrator]
                  embed + layers 0–10   layers 11–21    layers 22–31 + norm + lm_head
```

**Scope**: prefill-only. The full token sequence is processed in a single forward pass. Autoregressive generation (token-by-token with KV cache) is not supported.

**Inter-partition tensor**: `float [B, seq_len, 4096]` — the hidden state after each block of decoder layers. `p1` receives `int64 [B, seq_len]` token IDs from the orchestrator and produces the first hidden state.

**Result**: `p3` returns `float [B, seq_len, 128256]` logits to the orchestrator, which computes the metric (perplexity or accuracy) locally.

## Prerequisites

**HuggingFace access**: Llama-3.1-8B is a gated model. You must:
1. Accept Meta's license at `huggingface.co/meta-llama/Llama-3.1-8B`
2. Create a HuggingFace access token at `huggingface.co/settings/tokens`
3. Copy `.env.example` to `.env` and set `HF_TOKEN`:

```bash
cp .env.example .env
# edit .env: HF_TOKEN=hf_...
```

`.env` is gitignored and will never be committed.

**Memory**: at BF16 (default), each partition holds ~2.7 B parameters ≈ 5.4 GB. Each node needs at least 8 GB VRAM (accounting for activation buffers). The partition generation step loads the full model (all 32 layers) and requires ~18 GB RAM on the host.

## Step 1: Generate Partitions

Run once on the host before starting any nodes. Reads the model from HuggingFace hub (or a local path), splits it into three `nn.Module` stages, and writes the files that node containers will load at startup.

```bash
# Source HF_TOKEN from .env if not already exported
export $(grep -v '^#' .env | xargs)

python models/llama/partition_llama.py \
  --model meta-llama/Llama-3.1-8B \
  --output-dir models/llama/.partitions \
  --dtype bf16

# Optional: verify output matches the full model (top-1 token agreement ≥ 99%)
python models/llama/partition_llama.py \
  --model meta-llama/Llama-3.1-8B \
  --output-dir models/llama/.partitions \
  --dtype bf16 --verify
```

**Output** (`models/llama/.partitions/`, gitignored):
- `p1.pt` — embed_tokens + layers 0–10
- `p2.pt` — layers 11–21
- `p3.pt` — layers 22–31 + norm + lm_head
- `tokenizer/` — saved tokenizer (used by the data client at experiment runtime)

**Dtype options**: `bf16` (default, preferred on Ampere+), `fp16`, `fp32`. Use `fp32` only for debugging — it doubles memory use.

**Local model path**: pass a directory path instead of a HuggingFace repo ID if you have weights locally:
```bash
python models/llama/partition_llama.py --model /data/models/llama-3.1-8b ...
```

## Step 2: Build the Images

Llama nodes use `docker/Dockerfile.compute-llama`. The orchestrator and metrics server use their own images shared with ResNet experiments.

```bash
make build-llama         # dnn-compute-llama:latest  (CUDA + transformers)
make build-metrics       # dnn-metrics:latest
make build-orchestrator  # dnn-orchestrator:latest
```

The Llama compute image does not bake in model weights — partitions are mounted from the host at runtime via the `PARTITIONS_DIR` environment variable.

## Step 3: Run an Experiment

### WikiText-2 (perplexity)

The orchestrator tokenizes the WikiText-2 test set, sends `max_seq_len`-token chunks through the pipeline, receives logits, and computes per-token NLL → perplexity.

```bash
python -m framework.deploy \
  --experiment experiments/llama_topk_sweep \
  --target docker \
  --image dnn-compute-llama:latest \
  --partitions-dir models/llama/.partitions \
  --dataset-dir models/llama/.partitions/tokenizer \
  --apply
```

The runner logs perplexity after each sweep run:
```
Run 'e0_none_0.00' perplexity: 8.42
Run 'e1_topk_0.10' perplexity: 8.47
Run 'e1_topk_0.30' perplexity: 9.13
...
```

### MMLU (accuracy)

The orchestrator formats multiple-choice questions, sends tokenized prompts through the pipeline, receives logits, and picks the highest-scoring token among A/B/C/D.

The runner logs accuracy after each run:
```
Run 'e0_none_0.00' accuracy: 62.33% (187/300)
Run 'e1_topk_0.10' accuracy: 61.67% (185/300)
...
```

### Dry run (inspect sweep plan)

```bash
python -m framework.nodes.orchestrator.runner \
  experiments/llama_topk_sweep/experiment_wikitext.yaml --dry-run
```

### Monitor

```bash
docker compose -f experiments/llama_topk_sweep/deploy/docker-compose.yml logs -f orchestrator
```

## Experiment Config Fields

Llama experiment configs use all standard fields plus Llama-specific dataset fields:

```yaml
name: llama_topk_wikitext
model: llama-3.1-8b          # prefix "llama" triggers LlamaDataClient

dataset:
  name: wikitext2             # "wikitext2" or "mmlu"
  path: /partitions/tokenizer # used as tokenizer path if tokenizer_path not set
  tokenizer_path: /partitions/tokenizer  # directory produced by partition_llama.py
  batch_size: 2
  max_in_flight: 2
  max_seq_len: 512            # token sequence length per chunk (wikitext2 only)

  # MMLU-only fields:
  subjects:                   # list of MMLU subjects to evaluate
    - college_computer_science
    - high_school_mathematics
    - professional_law
    - global_facts
    - miscellaneous
    - business_ethics
  samples_per_subject: 50     # questions per subject
```

The `model` field must start with `"llama"` (case-insensitive) for the runner to select `LlamaDataClient`.

## Records Format

Llama records differ from ResNet records to accommodate both metric types. Each line of `records.jsonl`:

**WikiText-2 batch record:**
```json
{
  "request_id": "e1_topk_0.10_3_abc123",
  "batch_idx": 3,
  "metric_type": "perplexity",
  "experiment_id": "llama_topk_wikitext",
  "run_id": "e1_topk_0.10",
  "timestamp": 1234567890.0,
  "ground_truth": [],
  "predicted": [],
  "nll_sum": 142.7,
  "token_count": 1022
}
```

Perplexity for a run: `exp(sum(nll_sum) / sum(token_count))` across all records.

**MMLU batch record:**
```json
{
  "request_id": "e1_topk_0.10_0_def456",
  "batch_idx": 0,
  "metric_type": "accuracy",
  "experiment_id": "llama_topk_mmlu",
  "run_id": "e1_topk_0.10",
  "timestamp": 1234567890.0,
  "ground_truth": [1, 2, 0, 3],
  "predicted":    [1, 2, 0, 1],
  "nll_sum": 0.0,
  "token_count": 0
}
```

## Limitations

- **Prefill-only**: the pipeline processes each sequence in a single forward pass. Token-by-token generation with a growing KV cache is not supported.
- **Fixed sequence length**: WikiText-2 chunks are padded/truncated to `max_seq_len`. MMLU prompts are padded to the longest prompt in the batch (up to 512 tokens).
- **No CUDA quantization**: BitsAndBytes 4-bit/8-bit weight quantization (used in `comp_analysis.py`) requires CUDA and is not integrated into the node server. Use `--dtype fp16` or `bf16` for standard half-precision.
- **Single link topology**: the current experiment configs apply the same compression rate to both links. Different rates per link are supported by the sweep schema — create a custom experiment config using `sweep_mode: product`.

## Partition Details

| Partition | Layers | Parameters (approx) | BF16 size |
|-----------|--------|---------------------|-----------|
| p1 | embed_tokens + 0–10 | ~2.8 B | ~5.4 GB |
| p2 | 11–21 | ~2.6 B | ~5.1 GB |
| p3 | 22–31 + norm + lm_head | ~2.7 B | ~5.4 GB |

Partition boundaries follow the early/middle/late thirds from `comp_analysis.py` (layers 0–10 / 11–21 / 22–31 for a 32-layer model). The cut points are computed programmatically as `n//3` and `2*n//3`, so the script generalises to other model sizes.
