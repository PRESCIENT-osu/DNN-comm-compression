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

# Or via the deploy tool with --apply
python -m framework.deploy.deploy --experiment experiments/resnet56_topk_sweep \
  --target docker --partitions-dir models/resnet/.partitions --apply
```

**Orchestrator**: runs on the host. Use `--node-host localhost` and `--metrics-host localhost` since all services are published to host ports. `--callback-host` must be a host IP reachable from inside the containers (not `127.0.0.1`):

```bash
python -m framework.orchestrator.runner experiments/resnet56_topk_sweep \
  --node-host localhost \
  --metrics-host localhost \
  --callback-host <your-machine-ip>
```

## Kubernetes (KinD)

Generates a single `manifests.yaml` containing:
- `ConfigMap` — experiment YAML mounted into pods at `/app/config/experiment.yaml`
- One `Pod` + `Service` per pipeline node
- Metrics server `Pod` + `Service`

**Bandwidth limits**: applied via Cilium `kubernetes.io/egress-bandwidth` pod annotation. Requires Cilium CNI with bandwidth manager enabled in the KinD cluster.

**Volumes**: partitions are mounted as `hostPath` volumes — the paths must be accessible to kind nodes. For paths outside `/tmp`, add `extraMounts` to the kind cluster config. The dataset is not mounted into pods; it is loaded directly by the orchestrator on the host.

**NodePorts**: to expose node and metrics services outside the cluster, set `node_port` per node and `metrics_node_port` in the experiment's `infra.yaml`. If omitted, services are `ClusterIP` only (unreachable from the host orchestrator without `kubectl port-forward`).

**Applying**:
```bash
python -m framework.deploy.deploy --experiment experiments/resnet56_topk_sweep \
  --target k8s --partitions-dir /path/to/partitions --apply

# Or manually
kubectl apply -f experiments/resnet56_topk_sweep/deploy/manifests.yaml
```

**Orchestrator**: runs on the host and reaches pods via NodePort. Pass the cluster node's IP (e.g. Tailscale IP of the server) and your local machine's IP for the callback:

```bash
python -m framework.orchestrator.runner experiments/resnet56_topk_sweep \
  --node-host <cluster-node-ip> \
  --metrics-host <cluster-node-ip> \
  --callback-host <local-machine-ip>
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
