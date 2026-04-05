"""Simulated ResNet56 pipeline for Stein gradient oracle accuracy evaluation.

Loads the full ResNet56 from a checkpoint and installs activation compression
hooks at the exact inter-node partition boundaries defined by the pipeline
config.  No partition .pt files are required — hooks operate directly on the
original nn.Module, completely bypassing TorchScript.

Partition → hook target mapping in the full ResNet56:
  p1  →  pre-forward on model.layer1   (captures relu(bn1(conv1(x))) output)
  p2  →  pre-forward on model.layer2   (captures layer1 output)
  p3  →  pre-forward on model.layer3   (captures layer2 output)
  p4  →  post-forward on model.layer3  (captures layer3 output, before functional avgpool)

The eta vector has one element per inter-node link (len(flow) - 1).
Compression is applied via injected ``compress_fns`` callables, one per link,
matching the scheme used by the deployed compressors on the nodes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from framework.optimizer.compression_simulator import topk_sparsify_per_sample
from models.resnet.partition_resnet56 import ResNet56
from models.resnet.partition_resnet56 import load_model as _load_resnet56

logger = logging.getLogger(__name__)

# Maps the last partition name on a sending node to the hook target in the
# full ResNet56.  'pre' = hook fires on the input to the named submodule
# (= output of that partition); 'post' = hook fires on the output.
_BOUNDARY: dict[str, tuple[str, str]] = {
    "p1": ("layer1", "pre"),
    "p2": ("layer2", "pre"),
    "p3": ("layer3", "pre"),
    "p4": ("layer3", "post"),  # p5 starts with functional avgpool; hook layer3 output
}


class SimulatedResNetPipeline:
    """Full-model ResNet56 simulation with inter-node compression hooks.

    Loads the full ResNet56 from a checkpoint and installs activation
    compression hooks at the exact positions where partition boundaries cross
    node boundaries.  This mirrors what the deployed system does: the sender
    compresses the activation before transmitting, the receiver decompresses
    and feeds the result into its first partition.

    The eta vector has one element per inter-node link (= len(flow) - 1).
    Element eta[i] is the compression ratio for the link flow[i] → flow[i+1].

    Args:
        checkpoint_path: Path to the ResNet56 .th checkpoint file.
        partitions: Mapping of node_id → list of partition names assigned to
            that node, e.g. ``{"A": ["p1"], "B": ["p2", "p3"], "C": ["p4", "p5"]}``.
        flow: Ordered list of node IDs for this pipeline, e.g. ``["A", "B", "C"]``.
        test_loader: DataLoader yielding (image_tensor, label_tensor) batches
            for CIFAR-10 accuracy evaluation.
        compress_fns: One callable per inter-node link.  Each callable has
            signature ``(tensor: Tensor, eta: float) -> Tensor`` and simulates
            the deployed compressor's round-trip information loss for that link.
            If ``None``, defaults to ``topk_sparsify_per_sample`` for all links.
        device: Compute device.  Defaults to CPU.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        partitions: dict[str, list[str]],
        flow: list[str],
        test_loader: DataLoader,
        compress_fns: list[Callable[[torch.Tensor, float], torch.Tensor]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.test_loader = test_loader

        self.model: ResNet56 = _load_resnet56(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        self._n_links: int = len(flow) - 1
        self._eta: list[float] = [1.0] * self._n_links
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        if compress_fns is None:
            self._compress_fns: list[Callable[[torch.Tensor, float], torch.Tensor]] = [
                topk_sparsify_per_sample for _ in range(self._n_links)
            ]
        else:
            if len(compress_fns) != self._n_links:
                raise ValueError(
                    f"Expected {self._n_links} compress_fns (one per link), "
                    f"got {len(compress_fns)}"
                )
            self._compress_fns = compress_fns

        for link_idx in range(self._n_links):
            sender = flow[link_idx]
            last_partition = partitions[sender][-1]
            if last_partition not in _BOUNDARY:
                raise ValueError(
                    f"No boundary hook defined for partition '{last_partition}'. "
                    f"Supported: {list(_BOUNDARY)}"
                )
            attr_name, hook_type = _BOUNDARY[last_partition]
            submodule: nn.Module = getattr(self.model, attr_name)

            if hook_type == "pre":
                handle = submodule.register_forward_pre_hook(
                    self._make_pre_hook(link_idx)
                )
            else:
                handle = submodule.register_forward_hook(self._make_post_hook(link_idx))
            self._handles.append(handle)
            logger.debug(
                "Installed %s hook on model.%s for link %s→%s (eta[%d])",
                hook_type,
                attr_name,
                flow[link_idx],
                flow[link_idx + 1],
                link_idx,
            )

    @property
    def n_links(self) -> int:
        """Number of inter-node links (= dimension of eta)."""
        return self._n_links

    def set_eta(self, eta: torch.Tensor) -> None:
        """Update compression ratios in-place.

        Args:
            eta: Tensor of shape (n_links,) with values in [0, 1].

        Raises:
            ValueError: If eta length does not match the number of links.
        """
        if eta.numel() != self._n_links:
            raise ValueError(
                f"Expected eta of length {self._n_links}, got {eta.numel()}"
            )
        self._eta = [float(torch.clamp(v, 0.0, 1.0).item()) for v in eta]

    def accuracy(self, eta: torch.Tensor, *, n_samples: int | None = 512) -> float:
        """Evaluate CIFAR-10 top-1 accuracy with given compression ratios.

        Args:
            eta: Compression ratios of shape (n_links,).
            n_samples: Random subset size for fast evaluation.  Pass ``None``
                to run the full test loader (used for accuracy_callable_true).

        Returns:
            Top-1 accuracy in [0, 1].
        """
        self.set_eta(eta)
        self.model.eval()

        if n_samples is not None:
            if not hasattr(self, "_pool_x"):
                xs, ys = zip(*list(self.test_loader), strict=False)
                self._pool_x = torch.cat(xs)
                self._pool_y = torch.cat(ys)
            idx = torch.randperm(len(self._pool_x))[:n_samples]
            x = self._pool_x[idx].to(self.device)
            y = self._pool_y[idx].to(self.device)
            with torch.no_grad():
                correct = (self.model(x).argmax(1) == y).sum().item()
            return correct / len(y)

        correct = total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                correct += (self.model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / total

    def remove_hooks(self) -> None:
        """Remove all installed forward hooks."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def _make_pre_hook(self, link_idx: int):
        def hook(module: nn.Module, args: tuple) -> tuple:
            compressed = self._compress_fns[link_idx](args[0], self._eta[link_idx])
            return (compressed,) + args[1:]

        return hook

    def _make_post_hook(self, link_idx: int):
        def hook(module: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
            return self._compress_fns[link_idx](output, self._eta[link_idx])

        return hook
