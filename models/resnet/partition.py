import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

# -------------------------
# Minimal CIFAR ResNet56 parts
# (You can swap in your own ResNet56 implementation; the key is stage boundaries.)
# -------------------------

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            # CIFAR ResNet often uses "option A" downsample; projection also works for inference.
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = F.relu(out + identity, inplace=True)
        return out

def make_layer(in_ch, out_ch, blocks, stride):
    layers = [BasicBlock(in_ch, out_ch, stride=stride)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)

class Stage0(nn.Module):
    """stem + layer1 (16ch)"""
    def __init__(self, n=9):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(16)
        self.layer1 = make_layer(16, 16, blocks=n, stride=1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.layer1(x)
        return x

class Stage1(nn.Module):
    """layer2 (32ch, downsample)"""
    def __init__(self, n=9):
        super().__init__()
        self.layer2 = make_layer(16, 32, blocks=n, stride=2)

    def forward(self, x):
        return self.layer2(x)

class Stage2(nn.Module):
    """layer3 (64ch, downsample) + head"""
    def __init__(self, n=9, num_classes=10):
        super().__init__()
        self.layer3 = make_layer(32, 64, blocks=n, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x

def build_stage(stage_idx: int):
    if stage_idx == 0:
        return Stage0(n=9)
    if stage_idx == 1:
        return Stage1(n=9)
    if stage_idx == 2:
        return Stage2(n=9, num_classes=10)
    raise ValueError("stage_idx must be 0,1,2")

# -------------------------
# Distributed + Pipeline runtime
# -------------------------
def init_dist():
    dist.init_process_group(backend="nccl")  # GPUs
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank

@torch.no_grad()
def main():
    rank, world, local_rank = init_dist()
    assert world in (2, 3), "Use 2 or 3 pipeline stages/containers for this example."

    device = torch.device("cuda", local_rank)
    stage_idx = rank  # one stage per rank/container

    mod = build_stage(stage_idx).to(device).eval()

    # Wrap in PipelineStage: handles send/recv buffers between stages. [1](https://www.mathworks.com/help/deeplearning/ug/train-residual-network-for-image-classification.html)
    stage = PipelineStage(
        mod,
        stage_index=stage_idx,
        num_stages=world,
        device=device,
        group=None
    )

    # GPipe schedule runs microbatches through the pipeline. [1](https://www.mathworks.com/help/deeplearning/ug/train-residual-network-for-image-classification.html)[7](https://deepwiki.com/pytorch/PiPPy/3.1-gpipe-schedule)
    # For inference throughput, set n_microbatches > 1. For low latency, use 1 (less overlap).
    n_microbatches = int(os.environ.get("MICROBATCHES", "8"))
    schedule = ScheduleGPipe(stage, n_microbatches)

    # CIFAR input shape fixed => easier for PP buffer shapes. [1](https://www.mathworks.com/help/deeplearning/ug/train-residual-network-for-image-classification.html)
    batch = int(os.environ.get("BATCH", "256"))

    if stage_idx == 0:
        x = torch.randn(batch, 3, 32, 32, device=device)
        # Rank0 provides input; others call step() without args. [1](https://www.mathworks.com/help/deeplearning/ug/train-residual-network-for-image-classification.html)
        schedule.step(x)
        print(f"[rank {rank}] sent batch through pipeline")
    else:
        out = schedule.step()
        if stage_idx == world - 1:
            print(f"[rank {rank}] final logits shape: {tuple(out.shape)}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()