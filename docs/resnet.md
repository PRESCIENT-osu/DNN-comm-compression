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

### 1. Build the images

```bash
make build-resnet        # dnn-compute-resnet:latest
make build-metrics       # dnn-metrics:latest
make build-orchestrator  # dnn-orchestrator:latest

# or all at once
make build
```

### 2. Validate the config

```bash
make validate EXPERIMENT=experiments/resnet56_topk_sweep SHOW_RUNS=1
```

### 3. Generate and apply the Docker Compose manifest

```bash
python -m framework.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target docker \
  --image dnn-compute-resnet:latest \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10 \
  --apply
```

`--dataset-dir` is the host path to the dataset directory. It is mounted into the orchestrator container at the path configured in `experiment.yaml`.

Or generate without applying and inspect first:

```bash
python -m framework.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target docker \
  --image dnn-compute-resnet:latest \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10

docker compose -f experiments/resnet56_topk_sweep/deploy/docker-compose.yml up -d
```

The orchestrator starts automatically as part of the compose stack and runs the full sweep.

### 4. Monitor

```bash
docker compose -f experiments/resnet56_topk_sweep/deploy/docker-compose.yml logs -f orchestrator
```

### 5. Tear down

```bash
docker compose -f experiments/resnet56_topk_sweep/deploy/docker-compose.yml down
```

Repeat steps 3–5 for each experiment, substituting the experiment directory.

---

## Running on Kubernetes (kind)

These steps assume a kind cluster running on a remote server, with both your local machine and the server connected via Tailscale.

### Prerequisites

- kind cluster running on the server
- `kubectl` configured on your local machine to reach the cluster
- Docker images built and loaded into kind on the server
- Partitions copied to the server at a known path (e.g. `~/partitions/resnet56`)
- CIFAR-10 on the **server** at a known path (e.g. `~/datasets/cifar10`) — the orchestrator Job runs inside the cluster

### 1. Build images and load into kind

On the server:

```bash
make build-resnet
make build-metrics
make build-orchestrator

kind load docker-image dnn-compute-resnet:latest
kind load docker-image dnn-metrics:latest
kind load docker-image dnn-orchestrator:latest
```

### 2. Validate the config

```bash
make validate EXPERIMENT=experiments/resnet56_topk_sweep
```

### 3. Generate and apply manifests

From your local machine (with `kubectl` access to the cluster):

```bash
python -m framework.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target k8s \
  --image dnn-compute-resnet:latest \
  --partitions-dir /path/on/server/to/resnet56/partitions \
  --dataset-dir /path/on/server/to/datasets/cifar10 \
  --metrics-dir /path/on/server/to/metrics_data \
  --apply
```

`--partitions-dir`, `--dataset-dir`, and `--metrics-dir` are hostPath values written into pod specs — they must be paths accessible on the kind cluster nodes (i.e. paths on the server). If using paths outside `/tmp`, add `extraMounts` to your kind cluster config.

Or generate without applying and inspect first:

```bash
python -m framework.deploy \
  --experiment experiments/resnet56_topk_sweep \
  --target k8s \
  --image dnn-compute-resnet:latest \
  --partitions-dir /path/on/server/to/resnet56/partitions \
  --dataset-dir /path/on/server/to/datasets/cifar10 \
  --metrics-dir /path/on/server/to/metrics_data

kubectl apply -f experiments/resnet56_topk_sweep/deploy/manifests.yaml
```

The orchestrator runs as a Kubernetes Job inside the cluster. It connects to nodes and the metrics server via cluster DNS and exits when the sweep is complete.

### 4. Monitor

```bash
kubectl logs -f job/resnet56-topk-sweep-orchestrator
```

### 5. Tear down

```bash
kubectl delete -f experiments/resnet56_topk_sweep/deploy/manifests.yaml
```
