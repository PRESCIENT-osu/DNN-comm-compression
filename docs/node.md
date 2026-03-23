# Node Server

Each pipeline node runs a FastAPI server that handles inference requests, exposes a management API, and probes outgoing links when idle. Implemented in `framework/node/server.py`.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NODE_NAME` | yes | — | Node name matching experiment config (e.g. `A`) |
| `EXPERIMENT_CONFIG_PATH` | yes | — | Path to `experiment.yaml` |
| `PARTITIONS_DIR` | yes | — | Directory containing `<name>.pt` partition files |
| `METRICS_SERVER_URL` | no | `http://metrics:9100` | Metrics server base URL |
| `PROBE_INTERVAL_S` | no | `30` | Seconds between link probes |
| `METRICS_BUFFER_SIZE` | no | `1000` | Max events to buffer before dropping |
| `PORT` | no | `8000` | Bind port (overridden by `--port` CLI arg) |

## Running

```bash
NODE_NAME=A \
EXPERIMENT_CONFIG_PATH=experiments/resnet56_topk_sweep/experiment.yaml \
PARTITIONS_DIR=/app/partitions \
python -m framework.node.server --host 0.0.0.0 --port 8000
```

## Inference API

### `POST /infer`

Accepts an inference request and processes it in a background task. Returns `202 Accepted` immediately so the HTTP connection is not held open.

**Request body:**
```json
{
  "task_id": "abc123",
  "callback_url": "http://orchestrator:8080/result",
  "experiment_id": "resnet56_topk_sweep",
  "run_id": "e1_topk_0.10",
  "data": "<base64-encoded bytes>"
}
```

`data` carries:
- **First node**: pickled raw input tensor (from orchestrator, no compression)
- **Other nodes**: compressed activation bytes from previous node

**Processing pipeline (background task):**
1. Decode base64 → bytes
2. Decompress bytes → tensor (skipped on first node; emits `decompress` metric)
3. Run tensor through all assigned partitions sequentially (emits `forward_pass` metric)
4. If last node: pickle result and POST to `callback_url`
5. Otherwise: compress tensor (emits `compress` metric), POST to next node (emits `send` metric)

## Management API

### `GET /health`
Returns `{"status": "ok", "node": "<name>"}`. Used for readiness checks.

### `GET /status`
Returns current node status:
```json
{
  "node": "A",
  "in_flight": 3,
  "device": "cpu",
  "incoming": {"method": "none", "rate": 0.0},
  "outgoing": {"method": "topk", "rate": 0.1}
}
```
The orchestrator polls this before pushing a config change to confirm in-flight requests have drained.

### `POST /config`
Updates the compression config for the incoming or outgoing link.

**Request body:**
```json
{
  "direction": "outgoing",
  "method": "topk",
  "rate": 0.3,
  "drain_timeout_s": 60.0
}
```

Waits up to `drain_timeout_s` for in-flight requests to reach zero before applying. Returns `409` if the timeout is exceeded.

The orchestrator calls this on both the sending node (`direction: "outgoing"`) and the receiving node (`direction: "incoming"`) when changing a link's compression.

## Probe API

### `GET /probe`
Returns `{"status": "ok", "timestamp": <float>}`. Used by the previous node's prober to measure RTT.

### `POST /probe`
Accepts an arbitrary binary payload and returns its size. Used for throughput measurement.

## Link Probing

Each non-last node runs a background `_probe_loop` task that:
1. Waits for `idle_event` (in_flight == 0)
2. Pauses 0.5s to let the idle state settle
3. Measures RTT via `GET /probe` on the next node
4. Measures throughput via `POST /probe` with a 100 KB payload
5. Emits a `link_probe` metric event
6. Sleeps for `PROBE_INTERVAL_S` before trying again

Probing never happens during active inference — it only runs during idle windows between runs.

## Device Detection

At startup the server calls `torch.cuda.is_available()`. If a CUDA device is present it is used for all partition forward passes. Otherwise CPU is used. The detected device is reported in `GET /status` and in `forward_pass` metric events.

## Partition Loading

Partitions are loaded as TorchScript modules (`torch.jit.load`) from `PARTITIONS_DIR/<name>.pt` in the order specified by the node's `partitions` list in the experiment config. All models are set to eval mode.

## Llama Node Server

Llama experiments use a separate server implementation (`framework/node/llama_server.py`) because HuggingFace Llama models are not TorchScript-compatible. `llama_server.py` is identical to `server.py` except:

- Partitions are loaded via `torch.load(..., weights_only=False)` instead of `torch.jit.load`
- `models.llama.partition_llama` is imported at startup so that Python's pickle machinery can find the `LlamaPartition*` classes when deserialising the `.pt` files

All APIs (`/infer`, `/health`, `/status`, `/config`, `/probe`), the compression pipeline, metrics emission, and link probing are identical between the two servers.

**Running the Llama server:**
```bash
NODE_NAME=early \
EXPERIMENT_CONFIG_PATH=experiments/llama_topk_sweep/experiment_wikitext.yaml \
PARTITIONS_DIR=/partitions \
python -m framework.node.llama_server --host 0.0.0.0 --port 8000
```

**Building the image:**
```bash
docker build -f models/llama/Dockerfile -t llama-node:latest .
```

The Llama Dockerfile is at `models/llama/Dockerfile` and sets `PARTITIONS_DIR=/partitions` and `HF_HOME=/hf_cache` as defaults, both expected to be volume mounts. See [docs/llama.md](llama.md) for the full workflow.
