from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch
import torchvision
import torchvision.transforms as transforms

from framework.datamodels.experiment import DatasetConfig

logger = logging.getLogger(__name__)


class Dataset(ABC):
    """Abstract base class for datasets used by the data client."""

    @abstractmethod
    def batches(self) -> Iterator[tuple[int, torch.Tensor, list[int]]]:
        """Yield batches of data.

        Yields:
            Tuples of (batch_idx, input_tensor, ground_truth_labels).
        """

    @abstractmethod
    def num_batches(self) -> int:
        """Return the total number of batches.

        Returns:
            Total batch count.
        """


class Cifar10Dataset(Dataset):
    """CIFAR-10 test set loader.

    Loads the standard CIFAR-10 test split and yields normalised batches
    suitable for ResNet56 inference.

    Args:
        path: Directory containing (or where to download) the CIFAR-10 data.
        batch_size: Number of images per batch.
        max_samples: If set, cap the number of images used. Applied after
            optional shuffling so the subset is consistent across runs.
        seed: If set, shuffle the test set with this seed before subsetting.
            Must be set together with max_samples to have any effect; a seed
            alone (without max_samples) still shuffles, giving a randomised
            order that is stable across sweep runs.
    """

    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD = (0.2023, 0.1994, 0.2010)

    def __init__(
        self,
        path: str,
        batch_size: int,
        max_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD),
            ]
        )
        base = torchvision.datasets.CIFAR10(
            root=path, train=False, download=True, transform=transform
        )
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(base), generator=generator).tolist()
        else:
            indices = list(range(len(base)))
        if max_samples is not None:
            indices = indices[:max_samples]
        self._dataset = torch.utils.data.Subset(base, indices)
        self._batch_size = batch_size
        self._loader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    def batches(self) -> Iterator[tuple[int, torch.Tensor, list[int]]]:
        """Yield CIFAR-10 test batches.

        Yields:
            Tuples of (batch_idx, image_tensor [B, 3, 32, 32], labels).
        """
        for idx, (images, labels) in enumerate(self._loader):
            yield idx, images, labels.tolist()

    def num_batches(self) -> int:
        """Return the total number of batches.

        Returns:
            Number of batches in the test set.
        """
        return len(self._loader)


_REGISTRY: dict[str, type[Dataset]] = {
    "cifar10": Cifar10Dataset,
}


def get_dataset(config: DatasetConfig) -> Dataset:
    """Instantiate the dataset specified in the config.

    Args:
        config: Dataset configuration from the experiment config.

    Returns:
        A Dataset instance ready to yield batches.

    Raises:
        ValueError: If the dataset name is not supported.
    """
    cls = _REGISTRY.get(config.name.lower())
    if cls is None:
        raise ValueError(
            f"Unsupported dataset '{config.name}'. Supported: {list(_REGISTRY.keys())}"
        )
    return cls(
        path=config.path,
        batch_size=config.batch_size,
        max_samples=config.max_samples,
        seed=config.seed,
    )
