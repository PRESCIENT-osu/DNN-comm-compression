# Metrics

Metrics are collected on each node and streamed asynchronously to a central metrics server. The critical inference path (decompress → forward pass → compress → send) never waits on metrics I/O.

## Event Types

All events share common base fields:

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | string | Discriminator (see below) |
| `experiment_id` | string | Experiment name |
| `run_id` | string | Sweep run identifier |
| `timestamp` | float | Unix timestamp |
| `request_id` | string \| null | Per-request ID; null for link probe events |

### `forward_pass`
Emitted by a node after completing the forward pass through all its partitions.

| Field | Description |
|-------|-------------|
| `node` | Node name |
| `duration_ms` | Forward pass wall time |
| `device` | Device used (`cpu`, `cuda:0`, etc.) |

### `compress`
Emitted by a node after compressing an outgoing activation tensor.

| Field | Description |
|-------|-------------|
| `node` | Node name |
| `method` | Compression method |
| `rate` | Compression rate |
| `input_bytes` | Size of uncompressed tensor (bytes) |
| `output_bytes` | Size of compressed payload (bytes) |
| `duration_ms` | Compression wall time |

### `decompress`
Emitted by a node after decompressing a received activation tensor.

| Field | Description |
|-------|-------------|
| `node` | Node name |
| `method` | Compression method |
| `duration_ms` | Decompression wall time |

### `send`
Emitted by a node after sending an activation payload to the next node.

| Field | Description |
|-------|-------------|
| `from_node` | Sending node |
| `to_node` | Receiving node |
| `payload_bytes` | Compressed payload size |
| `duration_ms` | Network send wall time |

### `result`
Emitted by the orchestrator when a completed inference result is received.

| Field | Description |
|-------|-------------|
| `predicted` | Model output |
| `actual` | Ground truth label |

### `end_to_end`
Emitted by the orchestrator after a batch completes the full pipeline. Measures wall-clock time from when the batch was POSTed to the first node until the result was received at the orchestrator callback server. Emitted by both `DataClient` (image models) and `LlamaDataClient`.

| Field | Description |
|-------|-------------|
| `duration_ms` | Total round-trip time from orchestrator's perspective |
| `batch_size` | Number of samples in the batch |

This is the only orchestrator-side latency event. All other latency events (`forward_pass`, `compress`, `decompress`, `send`) are emitted by nodes and cover individual stages within the pipeline.

### `link_probe`
Emitted by the background link prober between experiment runs (never during active inference). See [Node Server](node.md) for probe timing.

| Field | Description |
|-------|-------------|
| `from_node` | Probing node |
| `to_node` | Target node |
| `rtt_ms` | Round-trip time |
| `throughput_mbps` | Observed throughput (null if not measured) |

## Emitter (`framework/node/metrics.py`)

Each node creates a `MetricsEmitter` instance at startup.

```python
emitter = MetricsEmitter(
    server_url="http://metrics:9100",
    buffer_size=1000,      # events before dropping
    batch_size=50,         # events per HTTP request
    flush_interval_s=1.0,  # max time before flush
    max_retries=3,         # attempts before dropping a batch
)
await emitter.start()

# Emit from anywhere in the node server (non-blocking)
emitter.emit(ForwardPassEvent(
    experiment_id="resnet56_topk_sweep",
    run_id="e1_topk_0.10",
    request_id="abc123",
    node="A",
    duration_ms=8.4,
    device="cpu",
))

await emitter.stop()  # flushes remaining events before shutdown
```

**Buffer full**: if the queue reaches `buffer_size`, the incoming event is dropped and a warning is logged. The inference path is unaffected.

**Server unavailable**: failed sends are retried with exponential backoff (1s, 2s, 4s, …) up to `max_retries`. After that, the batch is dropped with a warning.

## Metrics Server (`framework/metrics_server/server.py`)

Accepts batches of events via HTTP and appends them to per-experiment NDJSON files.

```
metrics_data/
└── resnet56_topk_sweep/
    ├── forward_pass.ndjson
    ├── compress.ndjson
    ├── decompress.ndjson
    ├── send.ndjson
    ├── end_to_end.ndjson
    ├── result.ndjson
    └── link_probe.ndjson
```

Each line in an NDJSON file is a complete JSON event object.

### Running the metrics server

```bash
python -m framework.metrics_server.server --host 0.0.0.0 --port 9100 --storage-dir metrics_data
```

Storage directory can also be set via `METRICS_STORAGE_DIR` environment variable.

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | POST | Ingest a JSON array of events |
| `/experiments` | GET | List experiments with stored metrics |
| `/experiments/{id}` | GET | List available event types for an experiment |
