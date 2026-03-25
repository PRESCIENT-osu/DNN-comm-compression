from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

import models.llama.partition_llama as _llama_partitions  # noqa: F401
from framework.nodes.compute.common.server import build_app

logger = logging.getLogger(__name__)


def _load_partitions(
    partition_names: list[str],
    partitions_dir: Path,
    device: str,
) -> list[Any]:
    """Load Llama partition modules from disk via torch.load.

    Args:
        partition_names: Ordered list of partition identifiers (e.g. ``["p2", "p3"]``).
        partitions_dir: Directory containing ``<name>.pt`` files.
        device: Device to map the loaded models to.

    Returns:
        List of loaded nn.Module partitions in partition order.

    Raises:
        FileNotFoundError: If a partition file does not exist.
    """
    partitions = []
    for name in partition_names:
        path = partitions_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Partition file not found: {path}")
        model = torch.load(str(path), map_location=device, weights_only=False)
        model.eval()
        partitions.append(model)
        logger.info("Loaded partition '%s' from %s", name, path)
    return partitions


app = build_app(_load_partitions)


def main() -> None:
    """Entry point for running the Llama node server."""
    import argparse

    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Llama DNN inference node server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
