"""Partition a pre-trained ResNet56 checkpoint into 5 sequential TorchScript modules.

Partition layout:
  p1 — conv1 + bn1 + relu
  p2 — layer1 (9 BasicBlocks, 16 channels)
  p3 — layer2 (9 BasicBlocks, 32 channels, stride=2 at entry)
  p4 — layer3 (9 BasicBlocks, 64 channels, stride=2 at entry)
  p5 — adaptive avgpool + flatten + linear

Usage:
  python models/resnet/partition_resnet56.py \\
    --checkpoint archive/models/resnet/resnet56-4bfd9763.th \\
    --output-dir models/resnet/partitions \\
    [--verify]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model definition (must match the checkpoint architecture)
# ---------------------------------------------------------------------------


class BasicBlock(nn.Module):
    """ResNet basic block with zero-padding shortcut (Option A)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.use_shortcut_pad: bool = stride != 1 or in_planes != planes
        self.shortcut_stride: int = stride
        self.shortcut_pad: int = planes // 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_shortcut_pad:
            shortcut = F.pad(
                x[:, :, :: self.shortcut_stride, :: self.shortcut_stride],
                (0, 0, 0, 0, self.shortcut_pad, self.shortcut_pad),
                "constant",
                0.0,
            )
        else:
            shortcut = x
        return F.relu(out + shortcut)


class ResNet56(nn.Module):
    """ResNet56 for CIFAR-10 matching the checkpoint architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, n_blocks=9, stride=1)
        self.layer2 = self._make_layer(16, 32, n_blocks=9, stride=2)
        self.layer3 = self._make_layer(32, 64, n_blocks=9, stride=2)
        self.linear = nn.Linear(64, 10)

    @staticmethod
    def _make_layer(
        in_planes: int, planes: int, n_blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlock(in_planes, planes, stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ---------------------------------------------------------------------------
# Partition modules
# ---------------------------------------------------------------------------


class Partition1(nn.Module):
    """conv1 + bn1 + relu."""

    def __init__(self, conv1: nn.Conv2d, bn1: nn.BatchNorm2d) -> None:
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn1(self.conv1(x)))


class Partition2(nn.Module):
    """layer1: 9 BasicBlocks at 16 channels."""

    def __init__(self, layer1: nn.Sequential) -> None:
        super().__init__()
        self.layer1 = layer1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer1(x)


class Partition3(nn.Module):
    """layer2: 9 BasicBlocks at 32 channels (stride=2 at entry)."""

    def __init__(self, layer2: nn.Sequential) -> None:
        super().__init__()
        self.layer2 = layer2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(x)


class Partition4(nn.Module):
    """layer3: 9 BasicBlocks at 64 channels (stride=2 at entry)."""

    def __init__(self, layer3: nn.Sequential) -> None:
        super().__init__()
        self.layer3 = layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(x)


class Partition5(nn.Module):
    """Adaptive avgpool + flatten + linear classifier."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: Path) -> ResNet56:
    """Load a ResNet56 model from a checkpoint file.

    Strips the ``module.`` DataParallel prefix from state dict keys if present.

    Args:
        checkpoint_path: Path to the ``.th`` checkpoint file.

    Returns:
        ResNet56 model in eval mode with loaded weights.
    """
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = raw["state_dict"] if "state_dict" in raw else raw

    # Strip DataParallel 'module.' prefix
    cleaned = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    model = ResNet56()
    model.load_state_dict(cleaned)
    model.eval()
    logger.info("Loaded checkpoint from %s", checkpoint_path)
    return model


# ---------------------------------------------------------------------------
# Partitioning and saving
# ---------------------------------------------------------------------------


def partition_and_save(model: ResNet56, output_dir: Path) -> dict[str, Path]:
    """Split model into 5 TorchScript partitions and save to disk.

    Args:
        model: Loaded ResNet56 model in eval mode.
        output_dir: Directory to write partition files into.

    Returns:
        Dict mapping partition name to saved file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parts: dict[str, nn.Module] = {
        "p1": Partition1(model.conv1, model.bn1),
        "p2": Partition2(model.layer1),
        "p3": Partition3(model.layer2),
        "p4": Partition4(model.layer3),
        "p5": Partition5(model.linear),
    }

    saved: dict[str, Path] = {}
    for name, module in parts.items():
        module.eval()
        scripted = torch.jit.script(module)
        path = output_dir / f"{name}.pt"
        scripted.save(str(path))
        logger.info("Saved %s → %s", name, path)
        saved[name] = path

    return saved


def verify_partitions(
    model: ResNet56,
    saved: dict[str, Path],
    device: str = "cpu",
) -> None:
    """Run a dummy batch through both the full model and the chained partitions.

    Asserts that outputs match to validate partitioning correctness.

    Args:
        model: Original full model for reference output.
        saved: Dict mapping partition name to .pt file path.
        device: Device to run verification on.
    """
    dummy = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        expected = model(dummy)

    parts = [
        torch.jit.load(str(saved[f"p{i}"]), map_location=device) for i in range(1, 6)
    ]
    x = dummy.to(device)
    with torch.no_grad():
        for part in parts:
            x = part(x)
    actual = x

    max_diff = (expected.to(device) - actual).abs().max().item()
    if max_diff < 1e-4:
        logger.info("Verification passed — max output diff: %.2e", max_diff)
    else:
        raise RuntimeError(
            f"Verification failed — max output diff {max_diff:.2e} exceeds threshold"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ResNet56 partition script."""
    parser = argparse.ArgumentParser(
        description="Partition ResNet56 into 5 TorchScript modules."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("archive/models/resnet/resnet56-4bfd9763.th"),
        help="Path to the .th checkpoint file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/resnet/.partitions"),
        help="Directory to save partition .pt files (default: models/resnet/.partitions)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify partitions by comparing output with the full model",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    saved = partition_and_save(model, args.output_dir)

    if args.verify:
        verify_partitions(model, saved)

    logger.info("Done. %d partitions saved to %s", len(saved), args.output_dir)


if __name__ == "__main__":
    main()
