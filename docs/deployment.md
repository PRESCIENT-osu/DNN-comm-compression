# Deployment

The deployment tool generates Docker Compose or Kubernetes manifests from an experiment directory and optionally applies them. Implemented in `framework/deploy/`.

## Docker Images

The framework uses four separate images — one per deployment role — to avoid shipping CUDA and PyTorch into containers that don't need them.

| Image | Dockerfile | Base | Purpose |
|-------|-----------|------|---------|
| `dnn-compute-resnet` | `docker/Dockerfile.compute-resnet` | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` | ResNet pipeline nodes |
| `dnn-compute-llama` | `docker/Dockerfile.compute-llama` | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` | Llama pipeline nodes |
| `dnn-metrics` | `docker/Dockerfile.metrics` | `python:3.11-slim` | Metrics ingestion server |
| `dnn-orchestrator` | `docker/Dockerfile.orchestrator` | `python:3.11-slim` | Experiment orchestrator (CPU only) |

### Building images

```bash
# All four images at once
make build

# Individual images
make build-resnet
make build-llama
make build-metrics
make build-orchestrator

# Custom tag
make build IMAGE_TAG=v0.2
make build-llama IMAGE_TAG=v0.2
```

All `make build-*` targets use the repo root as the Docker build context. The equivalent direct commands are:

```bash
docker build -f docker/Dockerfile.compute-resnet -t dnn-compute-resnet:latest .
docker build -f docker/Dockerfile.compute-llama  -t dnn-compute-llama:latest  .
docker build -f docker/Dockerfile.metrics        -t dnn-metrics:latest        .
docker build -f docker/Dockerfile.orchestrator   -t dnn-orchestrator:latest   .
```

### Traffic shaping

The compute node images include `iproute2` and copy `entrypoint.sh`. When the deploy tool injects `TC_LINK_<N>_*` environment variables into a compute node service (because the corresponding infra link has bandwidth/delay/loss parameters), `entrypoint.sh` applies per-link HTB qdiscs and netem rules before starting the server.

The metrics and orchestrator images do not include `entrypoint.sh` — they are never traffic-shaped.

## Generating Manifests

Generate the experiment directory first using `tools/generate.py` (see [docs/config.md](config.md#generating-experiments)), then deploy it:

```bash
# 1. Generate the experiment
python tools/generate.py \
    --spec specs/resnet56/equal-split \
    --profile profiles/linear-3/100mbps.yaml

# 2. Deploy the generated experiment
python -m framework.deploy \
  --experiment experiments/resnet56_equal-split_linear-3_100mbps \
  --target docker|k8s \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10 \
  [--image dnn-compute-resnet:latest] \
  [--metrics-dir metrics_data] \
  [--namespace default] \
  [--apply]
```

`--dataset-dir` is the host path to the dataset directory. It is mounted into the orchestrator container at the path configured in `exp.dataset.path`.

Manifests are written to `experiments/<name>/deploy/`:
- Docker: `docker-compose.yml`
- k8s: `manifests.yaml`

Without `--apply`, manifests are written for inspection only. With `--apply`, the tool runs `docker compose up -d` or `kubectl apply -f`.

## Docker Compose

Generates one service per pipeline node, a metrics server service, and an orchestrator service, all on a shared `pipeline` bridge network.

**Image selection**: the `--image` flag sets the compute node image. The metrics and orchestrator services always use `dnn-metrics` and `dnn-orchestrator` respectively (hardcoded in the backend). Pass the appropriate compute image for the model being deployed.

**Port mapping**: each node must have a unique port in the experiment config (e.g. A:8000, B:8001, C:8002).

**Per-link traffic shaping**: nodes with outgoing link parameters in the infra config receive `TC_LINK_<N>_*` environment variables and `cap_add: [NET_ADMIN]`. At startup, `entrypoint.sh` resolves each downstream node's hostname to an IP via Docker DNS and installs an HTB qdisc with per-destination classes:

```
tc htb class  →  rate limit per link
tc netem leaf →  delay / jitter / loss per link (if set)
tc u32 filter →  match destination IP → class
```

Each outgoing link gets its own HTB class and filter, so traffic to different downstream nodes is shaped independently.

**Orchestrator**: runs as a service in the compose network. It connects to nodes and the metrics server by their hostnames. Results are streamed to the metrics server as `ResultEvent`s — no local file writes. Monitor with:

```bash
docker compose -f experiments/resnet56_equal-split_linear-3_100mbps/deploy/docker-compose.yml logs -f orchestrator
```

**Running**:
```bash
# From the experiment deploy directory
cd experiments/resnet56_equal-split_linear-3_100mbps/deploy
docker compose up -d

# Or via the deploy tool with --apply
python -m framework.deploy \
  --experiment experiments/resnet56_equal-split_linear-3_100mbps \
  --target docker \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10 \
  --apply
```

## Kubernetes (KinD)

Generates a single `manifests.yaml` containing:
- `ConfigMap` — experiment YAML mounted into node pods at `/app/config/experiment.yaml`
- One `Pod` + `Service` per pipeline node
- Metrics server `Pod` + `Service`
- Orchestrator `Job`

**Per-link traffic shaping**: the same `TC_LINK_<N>_*` env var scheme as Docker is used. Node pod specs receive `securityContext.capabilities.add: [NET_ADMIN]` when any link has traffic shaping parameters. Node Services are **headless** (`clusterIP: None`) so DNS resolves directly to pod IPs — this is required because tc u32 filters match on the packet's destination IP, which in Kubernetes is the pod IP (after kube-proxy DNAT), not the ClusterIP.

**Orchestrator Job**: runs the sweep inside the cluster, connects to nodes and the metrics server via cluster DNS, and exits when complete. `CALLBACK_HOST` is injected from the pod's own IP via the Downward API so node pods can POST results back without a Service. Monitor with:

```bash
kubectl logs -f job/<experiment-name>-orchestrator
```

**Volumes**: partitions and the dataset are mounted as `hostPath` volumes — paths must be accessible on the kind cluster nodes. For paths outside `/tmp`, add `extraMounts` to the kind cluster config.

**NodePorts**: to expose the metrics server outside the cluster (e.g. to query it from your local machine), set `metrics_node_port` in `infra.yaml`. Node services do not need NodePorts since the orchestrator runs in-cluster.

**Applying**:
```bash
python -m framework.deploy \
  --experiment experiments/resnet56_equal-split_linear-3_100mbps \
  --target k8s \
  --partitions-dir /path/on/server/to/partitions \
  --dataset-dir /path/on/server/to/datasets/cifar10 \
  --apply

# Or manually
kubectl apply -f experiments/resnet56_equal-split_linear-3_100mbps/deploy/manifests.yaml
```

## Infra Config Inheritance

The `infra.yaml` in a generated experiment directory is the profile verbatim. For hand-written legacy experiments, infra configs can still inherit from a base and override specific nodes or links:

```yaml
# experiments/my_legacy_experiment/infra.yaml
inherits: ../base/resnet56_infra.yaml

links:
  - from: A
    to: B
    bandwidth_mbps: 10
    delay_ms: 20
  - from: B
    to: C
    bandwidth_mbps: 10
```

In the new workflow, network and resource variations are expressed as separate profiles (e.g. `profiles/linear-3/100mbps.yaml`, `profiles/linear-3/wan.yaml`) and selected at generation time via `tools/generate.py --profile`. The deploy tool resolves infra inheritance before generating manifests. Run `make validate` to check configs before deploying.

## Node Ports Convention

For multi-node experiments, assign sequential ports to avoid host port collisions:

| Node | Port |
|------|------|
| A | 8000 |
| B | 8001 |
| C | 8002 |
| metrics | 9100 |
| orchestrator callback | 8080 |
