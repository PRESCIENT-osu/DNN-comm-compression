# Deployment

The deployment tool generates Docker Compose or Kubernetes manifests from an experiment directory and optionally applies them. Implemented in `framework/deploy/`.

## Docker Images

The framework uses four separate images — one per deployment role — to avoid shipping CUDA and PyTorch into containers that don't need them.

| Image | Dockerfile | Base | Purpose |
|-------|-----------|------|---------|
| `dnn-compute-resnet` | `docker/Dockerfile.compute-resnet` | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` | ResNet pipeline nodes |
| `dnn-compute-llama` | `docker/Dockerfile.compute-llama` | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` | Llama pipeline nodes |
| `dnn-compute-multi` | `docker/Dockerfile.compute-multi` | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` | Multi-pipeline nodes (FIFO queue, ResNet + Llama) |
| `dnn-metrics` | `docker/Dockerfile.metrics` | `python:3.11-slim` | Metrics ingestion server |
| `dnn-orchestrator` | `docker/Dockerfile.orchestrator` | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` | Experiment orchestrator (CUDA — Stein oracle needs GPU) |

### Building images

```bash
# All images at once
make build

# Individual images
make build-resnet
make build-llama
make build-multi
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
docker build -f docker/Dockerfile.compute-multi  -t dnn-compute-multi:latest  .
docker build -f docker/Dockerfile.metrics        -t dnn-metrics:latest        .
docker build -f docker/Dockerfile.orchestrator   -t dnn-orchestrator:latest   .
```

### Orchestrator CUDA requirement

The orchestrator image is based on the CUDA PyTorch runtime (same base as the compute nodes). This is required because optimizer sub-experiments (see [docs/optimizer.md](optimizer.md)) run the Stein gradient oracle locally — evaluating full ResNet and Llama simulation pipelines on GPU to estimate ∇A_k(η). Sweep-only experiments (no optimizer sub-experiments) do not use the GPU, but the same image is used regardless to avoid maintaining two orchestrator images.

Mount an HuggingFace model cache at `/hf_cache` inside the orchestrator container to avoid re-downloading Llama weights on each run.

### Traffic shaping

The compute node images include `iproute2` and copy `entrypoint.sh`. When the deploy tool injects `TC_LINK_<N>_*` environment variables into a compute node service (because the corresponding infra link has bandwidth/delay/loss parameters), `entrypoint.sh` applies per-link HTB qdiscs and netem rules before starting the server.

The metrics and orchestrator images do not include `entrypoint.sh` — they are never traffic-shaped.

## Generating Manifests

Generate the experiment directory first using `tools/generate.py` (see [docs/config.md](config.md#generating-experiments)), then deploy it.

Manifests are written to `experiments/<name>/deploy/`:
- Docker: `docker-compose.yml`
- k8s: `manifests.yaml`

Without `--apply`, manifests are written for inspection only. With `--apply`, the tool runs `docker compose up -d` or `kubectl apply -f`.

### Single-model experiment

```bash
python tools/generate.py \
    --spec specs/resnet56/equal-split \
    --profile profiles/linear-3/100mbps.yaml

python -m framework.deploy \
  --experiment experiments/resnet56_equal-split_linear-3_100mbps \
  --target docker \
  --image dnn-compute-resnet:latest \
  --partitions-dir models/resnet/.partitions \
  --dataset-dir .datasets/cifar10 \
  [--metrics-dir metrics_data] \
  [--namespace default] \
  [--apply]
```

### Multi-model experiment

```bash
python tools/generate.py --multi \
    --spec multispecs/resnet56_llama_mmlu \
    --profile profiles/linear-3-multi/100mbps.yaml

python -m framework.deploy --multi \
  --experiment experiments/multi/resnet56_llama_mmlu_linear-3-multi_100mbps \
  --target docker \
  --image dnn-compute-multi:latest \
  --partitions-dir models \
  --dataset-dir .datasets \
  [--metrics-dir metrics_data] \
  [--apply]
```

`--partitions-dir` for multi-model is the base directory containing per-model subdirectories (e.g. `models/resnet/.partitions/` and `models/llama/.partitions/` under `models/`). `--dataset-dir` is the base `.datasets/` directory containing all dataset subdirectories.

### Optimizer experiment

```bash
python tools/generate.py --opt \
    --spec optspecs/resnet56_llama_mmlu \
    --profile profiles/linear-3-multi/100mbps.yaml

python -m framework.deploy --opt \
  --experiment experiments/opt/resnet56_llama_mmlu_linear-3-multi_100mbps \
  --target docker \
  --image dnn-compute-multi:latest \
  --partitions-dir models \
  --dataset-dir .datasets \
  [--artifacts-dir artifacts] \
  [--metrics-dir metrics_data] \
  [--apply]
```

`--artifacts-dir` is the host path for optimizer artifact storage (profiling results, estimator state, accuracy models). Defaults to `artifacts/`.

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

## Kubernetes (k3s single-node) — end-to-end optimization experiment

Complete runbook for running an **optimizer experiment** on a single-node k3s cluster on a GPU server (reference hardware: 8× NVIDIA L40S). Every command runs **on the server, from the repo root**, unless noted. The example uses the `opt_multi_resnet_lwiki` optspec with the `linear-3-opt/1Gbps` profile — substitute your own.

### What gets deployed

`python -m framework.deploy --opt --target k8s` writes a single `manifests.yaml` (into `experiments/opt/<name>/deploy/`) containing:

- One **Pod + headless Service** per pipeline node (`node-a/b/c`, Services `opt-a/opt-b/opt-c`), each running `framework.nodes.compute.multi.server`.
- A **metrics** Pod + Service (`metrics:9100`).
- An **orchestrator Job** running `framework.nodes.orchestrator.opt_runner /app/experiments/<name>`.

Manifest properties (from `framework/deploy/k8s_backend.py`):

- **Headless Services** (`clusterIP: None`) so DNS resolves to pod IPs — required because the per-link `tc u32` filters match the packet's destination pod IP, not a ClusterIP.
- **Traffic shaping**: node pods receive `TC_LINK_<N>_*` env vars + `securityContext.capabilities.add: [NET_ADMIN]`; `entrypoint.sh` installs HTB + netem per link at startup.
- **Callback**: the orchestrator's `CALLBACK_HOST` is set to its own pod IP via the Downward API; nodes POST results back to it directly (no Service needed).
- **Config & data** are `hostPath` mounts, not ConfigMaps. On single-node k3s every hostPath resolves to the server's own filesystem, so there is no KinD-style `extraMounts` step — absolute host paths just work.

### GPU usage

With the `linear-3-opt` profile (`gpu: 1` per node) and the orchestrator's hardcoded request:

| Pod | GPUs |
|-----|------|
| node-a / node-b / node-c | 1 each |
| orchestrator (Stein oracle runs full ResNet + Llama on GPU) | 1 |
| metrics | 0 |

**Total: 4 GPUs**, leaving 4 of the 8 L40S free (e.g. for a second concurrent experiment). Each L40S (48 GB) comfortably holds a Llama-3.1-8B partition (~16 GB fp16) or the orchestrator's full-model simulations.

### 0. Cluster & GPU prerequisites (one-time)

```bash
# NVIDIA driver already installed on the host (nvidia-smi lists 8× L40S).

# Single-node k3s
curl -sfL https://get.k3s.io | sh -
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml          # add to your shell profile

# GPU support: expose nvidia.com/gpu to the scheduler. Recommended: install the
# NVIDIA GPU Operator (sets up the container runtime, device plugin, and
# RuntimeClass); or install nvidia-container-toolkit + the k8s-device-plugin
# manually. Verify the node advertises 8 GPUs:
kubectl get node -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}{"\n"}'   # -> 8
```

> **`runtimeClassName` caveat.** The generated manifests do **not** set `runtimeClassName: nvidia`. This works when the NVIDIA runtime is the containerd default (the GPU Operator configures that). If your setup needs an explicit RuntimeClass, pods won't get GPUs until you either make `nvidia` the default runtime or patch `runtimeClassName: nvidia` into `manifests.yaml` after generating it — the deploy backend does not emit it.

### 1. Repo & Python environment (one-time)

```bash
git clone <repo-url> dnn-comm-compression && cd dnn-comm-compression
git submodule update --init --recursive              # pulls external/Inference_Optimizer
make install-dev                                     # creates .venv, installs deps
source .venv/bin/activate
```

### 2. Prepare models and datasets on the host (one-time)

Everything lands under the repo root and is `hostPath`-mounted into the pods.

```bash
# Model partitions (loaded by the node pods)
python models/resnet/partition_resnet56.py --verify
#   -> models/resnet/.partitions/{p1..p5}.pt

huggingface-cli login                                # Llama-3.1-8B is gated
python models/llama/partition_llama.py \
  --model meta-llama/Llama-3.1-8B \
  --output-dir models/llama/.partitions \
  --dtype fp16
#   -> models/llama/.partitions/{p1,p2,p3}.pt and models/llama/.partitions/tokenizer/

# Full models for the Stein simulation oracle (the `simulation_path` in the optspec).
#   ResNet checkpoint already ships in the repo: models/resnet/resnet56-4bfd9763.th
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir models/llama/llama-3.1-8b
#   -> models/llama/llama-3.1-8b/  (loaded by SimulatedLlamaPipeline)

# Datasets (.datasets is mounted read-write, so downloads persist to the host)
mkdir -p .datasets
python -c "import torchvision; torchvision.datasets.CIFAR10('.datasets/cifar10', train=False, download=True)"
#   -> .datasets/cifar10
#   WikiText-2 (and MMLU for other optspecs) are pulled from HuggingFace at runtime.
```

> The image sets `HF_HOME=/hf_cache` but the manifest does **not** mount it, so HuggingFace `datasets` downloads (WikiText-2 / MMLU) happen inside the orchestrator pod on each run and require network egress. Pre-seed the datasets or an HF cache if the pod is offline. Known gap.

### 3. Build and import images

Optimizer runs use three images: `dnn-compute-multi` (nodes), `dnn-metrics`, `dnn-orchestrator`.

```bash
make build-multi build-metrics build-orchestrator    # or `make build` for all five

# Import into k3s containerd. Pods use imagePullPolicy: Never and there is no
# registry, so the images must live in k3s's k8s.io containerd namespace, which
# `k3s ctr images import` targets by default.
for img in dnn-compute-multi:latest dnn-metrics:latest dnn-orchestrator:latest; do
  docker save "$img" | sudo k3s ctr images import -
done
sudo k3s ctr images ls | grep dnn-                   # verify all three imported
```

> **Re-import after every rebuild.** Because the pull policy is `Never`, rebuilding an image does not update the cluster until you re-run the import — otherwise pods silently run stale code.

### 4. Generate the experiment

```bash
python tools/generate.py --opt \
  --spec optspecs/opt_multi_resnet_lwiki \
  --profile profiles/linear-3-opt/1Gbps.yaml
#   -> experiments/opt/opt_multi_resnet_lwiki_linear-3-opt_1Gbps/{experiment.yaml, infra.yaml}
```

### 5. Deploy to the cluster

Run from the repo root so the relative host paths (`models`, `.datasets`, `artifacts`, `metrics_data`) resolve to `<repo>/...` on the server — the generator bakes absolute, resolved paths into the manifest.

```bash
kubectl create namespace dnn        # once

python -m framework.deploy --opt \
  --experiment experiments/opt/opt_multi_resnet_lwiki_linear-3-opt_1Gbps \
  --target k8s \
  --image dnn-compute-multi:latest \
  --partitions-dir models \
  --dataset-dir .datasets \
  --artifacts-dir artifacts \
  --metrics-dir metrics_data \
  --namespace dnn \
  --apply
```

`--apply` runs `kubectl apply -f manifests.yaml -n dnn`. To inspect first, omit it and apply manually:

```bash
kubectl apply -f experiments/opt/opt_multi_resnet_lwiki_linear-3-opt_1Gbps/deploy/manifests.yaml -n dnn
```

Flag notes (verified against `framework/deploy/__main__.py`): `--image` **must** be overridden to `dnn-compute-multi:latest` (default is `dnn-compression:latest`); `--partitions-dir models` is the base holding `resnet/.partitions` and `llama/.partitions`; `--dataset-dir .datasets` is the base of all dataset subdirs; `--artifacts-dir` (writable) holds profiling / estimator / accuracy-model outputs.

### 6. Monitor

```bash
kubectl get pods -n dnn -w
kubectl get jobs -n dnn

# Orchestrator log stream. Job name = experiment name, lowercased, '_' -> '-':
kubectl logs -f job/opt-multi-resnet-lwiki-linear-3-opt-1gbps-orchestrator -n dnn
```

The orchestrator waits for all node pods to report healthy, then runs the sub-experiments sequentially (profiling → accuracy models → optimizer / baseline slot loops) and the Job exits `Completed`.

### 7. Collect results

The metrics server writes NDJSON to its `hostPath` mount, i.e. directly on the server:

```
metrics_data/            # <repo>/metrics_data on the host — per-experiment NDJSON event stream
artifacts/               # profiling results, estimator state, accuracy-model pickles
```

To query the metrics HTTP API instead of reading files:

```bash
kubectl port-forward -n dnn svc/metrics 9100:9100
curl 'http://localhost:9100/metrics/query?event_type=opt_slot&experiment_id=opt_multi_resnet_lwiki_linear-3-opt_1Gbps'
```

(Alternatively set `metrics_node_port` in the profile / `infra.yaml` before generating to expose the metrics Service as a NodePort.)

For the Assumption-3 estimation-error analysis (`Δ_k`, `δ`, `δ⁺` from `opt_slot.d_hat_per_task`, `throughput_constraint`, and `channel_estimate_quality`), see [assumption3_validation.md](assumption3_validation.md); the opt-specific plotting pass is not yet implemented.

### 8. Teardown

```bash
kubectl delete -f experiments/opt/opt_multi_resnet_lwiki_linear-3-opt_1Gbps/deploy/manifests.yaml -n dnn
# results in metrics_data/ and artifacts/ persist on the host
```

### Known gaps / caveats

- **`runtimeClassName: nvidia` is not emitted** — see the GPU prerequisites note.
- **No HF cache mount** — WikiText-2 / MMLU download inside the orchestrator pod each run unless pre-seeded on the host.
- **Re-import images after every rebuild** (`imagePullPolicy: Never`).
- **Redeploy after code changes** — runner/adapter changes need only a re-imported `dnn-orchestrator` image and a re-applied Job; node-side changes need `dnn-compute-multi` rebuilt and re-imported.

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
