# Orchestrator

The orchestrator drives experiment sweeps from the host machine. It coordinates config pushes to nodes, sends dataset batches through the pipeline, collects results, and saves per-run records. It does not need to be containerised — it communicates with nodes via their exposed HTTP ports.

## Components

| Module | Role |
|--------|------|
| `framework/orchestrator/runner.py` | Top-level sweep loop and CLI entry point |
| `framework/orchestrator/controller.py` | Pushes compression configs to nodes |
| `framework/orchestrator/data_client.py` | Callback server, batch sending, result collection (image models) |
| `framework/orchestrator/llama_data_client.py` | Data client for Llama — tokenizes text, decodes logits as perplexity or accuracy |
| `framework/orchestrator/datasets.py` | Dataset loaders (image models) |

## Running

```bash
python -m framework.orchestrator.runner experiments/resnet56_topk_sweep

# With options
python -m framework.orchestrator.runner experiments/resnet56_topk_sweep \
  --callback-host host.docker.internal \
  --callback-port 8080 \
  --result-timeout 300 \
  --dry-run
```

**`--callback-host`**: the hostname pipeline nodes use to POST results back. Must be reachable from inside the node containers. For Docker Compose use `host.docker.internal`; for Kubernetes, run the orchestrator as a pod and use its service name. Defaults to `CALLBACK_HOST` env var or `localhost`.

**`--dry-run`**: prints the resolved sweep plan without sending any requests.

## Sweep Loop

For each resolved run:

1. **Push config** — controller sends `POST /config` to the sending and receiving node of each link concurrently. Each node drains in-flight requests before applying.
2. **Run data client** — sends all dataset batches to the first node, bounded by `max_in_flight` semaphore (held until result received).
3. **Collect results** — last node POSTs results to the orchestrator's callback server.
4. **Save records** — written to `experiments/<name>/results/<run_id>/records.jsonl`.
5. **Log metric** — top-1 accuracy for image models; perplexity or accuracy for Llama.

The runner selects the appropriate data client automatically based on `exp.model`: any model name starting with `"llama"` (case-insensitive) uses `LlamaDataClient`; all others use `DataClient`.

## Baseline Check

At startup the runner warns if referenced baseline results are not found:

```
WARNING: Baseline 'resnet56_distributed_baseline' has no results.
Run it before analysing this experiment for accurate comparisons.
```

## Controller

`push_run_config(exp, run)` sends configs for all links in a resolved run concurrently:

- Sending node receives `{"direction": "outgoing", "method": "...", "rate": ...}`
- Receiving node receives `{"direction": "incoming", "method": "...", "rate": ...}`

`wait_for_all_idle(exp)` polls all nodes' `GET /status` until `in_flight == 0` on every node. Used before pushing config when extra safety is needed.

## Data Client

`DataClient` manages the result callback server for the full session:

```python
client = DataClient(callback_host="host.docker.internal", callback_port=8080)
async with client.session():
    for run in runs:
        records = await client.run(exp, run_id, first_node_url, results_dir, emitter)
```

**Semaphore**: `max_in_flight` batches can be in the pipeline simultaneously. The semaphore is acquired before sending and released only after the result is received — ensuring true end-to-end backpressure.

**Callback server**: FastAPI app running on `callback_port`. The `POST /result` endpoint resolves the pending future for the matching `task_id`.

## Records

Per-run results are saved to `experiments/<name>/results/<run_id>/records.jsonl`.

**Image model record** (ResNet, etc.):
```json
{
  "request_id": "e1_topk_0.10_3_abc123",
  "batch_idx": 3,
  "ground_truth": [3, 7, 2, 1],
  "predicted": [3, 7, 2, 5],
  "experiment_id": "resnet56_topk_sweep",
  "run_id": "e1_topk_0.10",
  "timestamp": 1234567890.0
}
```

**Llama record** — includes `metric_type` and differs by dataset. See [docs/llama.md](llama.md#records-format) for the full schema.

Raw logits are not stored — only predicted indices or NLL sums.

To clean up records for an experiment, use the cleanup utility:

```bash
python -m framework.analysis.cleanup --experiment resnet56_topk_sweep --older-than 7
```

## Datasets

### Image models

| Name | Class | Notes |
|------|-------|-------|
| `cifar10` | `Cifar10Dataset` | Standard test split, downloads if not present |

To add a new image dataset, subclass `Dataset` in `framework/orchestrator/datasets.py` and register it in `_REGISTRY`.

### Llama

Llama datasets are handled by `LlamaDataClient` directly (not via `datasets.py`) and selected by `dataset.name` in the experiment config:

| Name | Metric | Source |
|------|--------|--------|
| `wikitext2` | Perplexity | WikiText-2 test split via HuggingFace `datasets` |
| `mmlu` | Accuracy (A/B/C/D) | `cais/mmlu` via HuggingFace `datasets` |

The data client tokenizes inputs using the tokenizer saved by `partition_llama.py` and computes metrics from the logits returned by the last node — no GPU is needed on the orchestrator side.

## Metrics

The runner creates its own `MetricsEmitter` and emits `ResultEvent` for each completed batch. These are streamed to the metrics server alongside node-side events and used during analysis to compute accuracy metrics.
