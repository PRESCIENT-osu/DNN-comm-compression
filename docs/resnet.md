# ResNet-56 Experiments

End-to-end guide for running ResNet-56 distributed inference experiments.

## Partitioning

ResNet-56 is split into 5 sequential TorchScript partitions. This is a one-time offline step — partitions are reused across all experiments.

```bash
python models/resnet/partition_resnet56.py --verify
```

Output is written to `models/resnet/.partitions/` (gitignored). The `--verify` flag runs a forward pass to confirm partition outputs match the original model.

## Dataset

CIFAR-10 is loaded by the orchestrator at runtime. Place it (or let torchvision download it) at `.datasets/cifar10/` relative to repo root:

```bash
mkdir -p .datasets/cifar10
# torchvision will auto-download on first run if the directory is empty
```

## Experiment Configs

| Experiment | Description |
|------------|-------------|
| `resnet56_single_node_baseline` | All 5 partitions on one node — no inter-node communication |
| `resnet56_distributed_baseline` | 3 nodes, no compression — measures raw distribution overhead |
| `resnet56_topk_sweep` | 3 nodes, TopK compression, paired sweep over rates [0.1, 0.3, 0.5] |
| `resnet56_topk_sweep_product` | 3 nodes, TopK compression, product sweep over rates [0.1, 0.25, 0.5, 0.75, 1.0] |

Run baselines before sweep experiments so analysis comparisons are available.

## Running on Docker

### 1. Build the image

```bash
docker build -t dnn-compression:latest .
```

### 2. Generate the Docker Compose manifest

```bash
python -m framework.deploy.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target docker \
  --partitions-dir models/resnet/.partitions \
  --metrics-dir metrics_data
```

Manifest is written to `experiments/resnet56_topk_sweep/deploy/docker-compose.yml`.

### 3. Start containers

```bash
docker compose -f experiments/resnet56_topk_sweep/deploy/docker-compose.yml up -d
```

### 4. Run the orchestrator

```bash
python -m framework.orchestrator.runner experiments/resnet56_topk_sweep \
  --node-host localhost \
  --metrics-host localhost \
  --callback-host <your-machine-ip>
```

`--callback-host` must be an IP reachable from inside the containers — use your machine's local network IP, not `127.0.0.1`. Find it with `ifconfig` or `ip addr`.

### 5. Tear down

```bash
docker compose -f experiments/resnet56_topk_sweep/deploy/docker-compose.yml down
```

Repeat steps 2–5 for each experiment, substituting the experiment directory.

---

## Running on Kubernetes (kind)

These steps assume a kind cluster running on a remote server, with both your local machine and the server connected via Tailscale.

### Prerequisites

- kind cluster running on the server
- `kubectl` configured on your local machine to reach the cluster
- Partitions copied to the server at a known path (e.g. `~/partitions/resnet56`)
- CIFAR-10 on your **local machine** at `.datasets/cifar10/` (the orchestrator runs locally)

### 1. Build the image and load into kind

On the server:

```bash
docker build -t dnn-compression:latest .
kind load docker-image dnn-compression:latest
```

### 2. Add NodePorts to the infra config

NodePorts must be set in the experiment's `infra.yaml` to expose services outside the cluster. They are already set for `resnet56_topk_sweep`. For other experiments, add them similarly:

```yaml
# experiments/resnet56_topk_sweep/infra.yaml
metrics_node_port: 30900

nodes:
  - name: A
    node_port: 30800
  - name: B
    node_port: 30801
  - name: C
    node_port: 30802
```

### 3. Generate and apply manifests

From your local machine (with `kubectl` access to the cluster):

```bash
python -m framework.deploy.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target k8s \
  --partitions-dir /path/on/server/to/resnet56/partitions \
  --metrics-dir /path/on/server/to/metrics_data \
  --apply
```

`--partitions-dir` and `--metrics-dir` are hostPath values written into the pod specs — they must be paths accessible on the kind cluster nodes (i.e. paths on the server). If using paths outside `/tmp`, add `extraMounts` to your kind cluster config.

Or generate without applying and inspect first:

```bash
python -m framework.deploy.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target k8s \
  --partitions-dir /path/on/server/to/resnet56/partitions \
  --metrics-dir /path/on/server/to/metrics_data

kubectl apply -f experiments/resnet56_topk_sweep/deploy/manifests.yaml
```

### 4. Run the orchestrator

```bash
python -m framework.orchestrator.runner experiments/resnet56_topk_sweep \
  --node-host <server-tailscale-ip> \
  --metrics-host <server-tailscale-ip> \
  --callback-host <local-machine-tailscale-ip>
```

Node services are exposed via NodePort on the server. The orchestrator reaches them at `<server-tailscale-ip>:<nodeport>`. The callback server runs on your local machine, reachable by pods through the server's host network and Tailscale routing.

### 5. Tear down

```bash
kubectl delete -f experiments/resnet56_topk_sweep/deploy/manifests.yaml
```
