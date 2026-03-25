from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from framework.nodes.compute.common.server import build_app

logger = logging.getLogger(__name__)


def _load_partitions(
    partition_names: list[str],
    partitions_dir: Path,
    device: str,
) -> list[torch.jit.ScriptModule]:
    """Load TorchScript partition models from disk.

    Args:
        partition_names: Ordered list of partition identifiers (e.g. ``["p2", "p3"]``).
        partitions_dir: Directory containing ``<name>.pt`` files.
        device: Device to map the loaded models to.

    Returns:
        List of loaded TorchScript modules in partition order.

    Raises:
        FileNotFoundError: If a partition file does not exist.
    """
    partitions = []
    for name in partition_names:
        path = partitions_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Partition file not found: {path}")
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        partitions.append(model)
        logger.info("Loaded partition '%s' from %s", name, path)
    return partitions


app = build_app(_load_partitions)


def main() -> None:
    """Entry point for running the ResNet node server."""
    import argparse

    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="ResNet DNN inference node server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
