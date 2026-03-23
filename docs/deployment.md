# Deployment

The deployment tool generates Docker Compose or Kubernetes manifests from an experiment directory and optionally applies them. Implemented in `framework/deploy/`.

## Building the Image

```bash
docker build -t dnn-compression:latest .
```

The image includes `iproute2` for `tc netem` bandwidth simulation and uses `entrypoint.sh` to apply bandwidth limits before starting the node server.

## Generating Manifests

```bash
python -m framework.deploy.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target docker|k8s \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir /path/to/data \
  [--image dnn-compression:latest] \
  [--metrics-dir metrics_data] \
  [--namespace default] \
  [--apply]
```

Manifests are written to `experiments/<name>/deploy/`:
- Docker: `docker-compose.yml`
- k8s: `manifests.yaml`

Without `--apply`, manifests are written for inspection only. With `--apply`, the tool runs `docker compose up -d` or `kubectl apply -f`.

## Docker Compose

One service per pipeline node plus a metrics server service, all on a shared `pipeline` bridge network.

**Port mapping**: each node must have a unique port in the experiment config (e.g. A:8000, B:8001, C:8002) so all nodes are reachable from the host orchestrator at `localhost:<port>`.

**Bandwidth simulation**: nodes with outgoing bandwidth limits in the infra config receive:
- `TC_BANDWIDTH_MBPS` environment variable (read by `entrypoint.sh`)
- `cap_add: [NET_ADMIN]` to allow `tc` to run

The `tc netem rate` command limits egress at the pod/container level (all outgoing traffic, not per-link). For link-specific control, use a CNI plugin.

**Running**:
```bash
# From the experiment deploy directory
cd experiments/resnet56_topk_sweep/deploy
docker compose up -d

# Or via the deploy tool
python -m framework.deploy.deploy --experiment experiments/resnet56_topk_sweep \
  --target docker --partitions-dir models/resnet/.partitions --dataset-dir /data --apply
```

**Orchestrator callback**: the orchestrator runs on the host and nodes must reach its callback server. Pass `--callback-host host.docker.internal` to the runner so nodes can POST results back:

```bash
python -m framework.orchestrator.runner experiments/resnet56_topk_sweep \
  --callback-host host.docker.internal --callback-port 8080
```

## Kubernetes (KinD)

Generates a single `manifests.yaml` containing:
- `ConfigMap` — experiment YAML mounted into pods at `/app/config/experiment.yaml`
- One `Pod` + `Service` per pipeline node
- Metrics server `Pod` + `Service`

**Bandwidth limits**: applied via Cilium `kubernetes.io/egress-bandwidth` pod annotation. Requires Cilium CNI with bandwidth manager enabled in the KinD cluster.

**Volumes**: partitions and dataset are mounted as `hostPath` volumes — the paths must be accessible to KinD nodes. For paths outside `/tmp`, add `extraMounts` to the KinD cluster config.

**Applying**:
```bash
python -m framework.deploy.deploy --experiment experiments/resnet56_topk_sweep \
  --target k8s --partitions-dir models/resnet/.partitions --dataset-dir /data --apply

# Or manually
kubectl apply -f experiments/resnet56_topk_sweep/deploy/manifests.yaml
```

**Orchestrator against k8s**: the orchestrator runs on the host and needs to reach node services. Use `kubectl port-forward` for each node:
```bash
kubectl port-forward svc/node-a 8000:8000 &
kubectl port-forward svc/node-b 8001:8001 &
kubectl port-forward svc/node-c 8002:8002 &

python -m framework.orchestrator.runner experiments/resnet56_topk_sweep \
  --callback-host <host-ip-reachable-from-pods> --callback-port 8080
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
