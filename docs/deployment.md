# Deployment

The deployment tool generates Docker Compose or Kubernetes manifests from an experiment directory and optionally applies them. Implemented in `framework/deploy/`.

## Building the Image

```bash
docker build -t dnn-compression:latest .
```

The image includes `iproute2` for `tc` traffic shaping. `entrypoint.sh` applies per-link bandwidth, delay, and loss rules before starting the node server or orchestrator.

## Generating Manifests

```bash
python -m framework.deploy.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target docker|k8s \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10 \
  [--image dnn-compression:latest] \
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
docker compose -f experiments/resnet56_topk_sweep/deploy/docker-compose.yml logs -f orchestrator
```

**Running**:
```bash
# From the experiment deploy directory
cd experiments/resnet56_topk_sweep/deploy
docker compose up -d

# Or via the deploy tool with --apply
python -m framework.deploy.deploy \
  --experiment experiments/resnet56_topk_sweep \
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
python -m framework.deploy.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target k8s \
  --partitions-dir /path/on/server/to/partitions \
  --dataset-dir /path/on/server/to/datasets/cifar10 \
  --apply

# Or manually
kubectl apply -f experiments/resnet56_topk_sweep/deploy/manifests.yaml
```

## Infra Config Inheritance

Child infra configs can inherit from a base and override specific nodes or links:

```yaml
# experiments/resnet56_bandwidth_test/infra.yaml
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

The deploy tool resolves inheritance before generating manifests. Run `make validate` to check configs before deploying.

## Node Ports Convention

For multi-node experiments, assign sequential ports to avoid host port collisions:

| Node | Port |
|------|------|
| A | 8000 |
| B | 8001 |
| C | 8002 |
| metrics | 9100 |
| orchestrator callback | 8080 |
