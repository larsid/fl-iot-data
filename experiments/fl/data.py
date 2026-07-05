from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class PartitionSpec:
    num_partitions: int = 64
    num_clients: int = 8


def build_transform(train: bool) -> transforms.Compose:
    ops: list = []
    if train:
        ops += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(ops)


def iid_partition_indices(
    dataset_size: int,
    spec: PartitionSpec,
    seed: int,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_size)
    rng.shuffle(indices)
    partition_size = dataset_size // spec.num_partitions
    return [
        indices[i * partition_size : (i + 1) * partition_size].tolist()
        for i in range(spec.num_partitions)
    ]


def select_client_partitions(
    partitions: Sequence[Sequence[int]],
    spec: PartitionSpec,
    seed: int,
) -> List[List[int]]:
    rng = np.random.default_rng(seed + 1)
    chosen = rng.choice(spec.num_partitions, size=spec.num_clients, replace=False)
    return [list(partitions[idx]) for idx in chosen]


def load_cifar10_train(root: str) -> Dataset:
    return datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=build_transform(train=True),
    )


def load_cifar10_test(root: str) -> Dataset:
    return datasets.CIFAR10(
        root=root,
        train=False,
        download=False,
        transform=build_transform(train=False),
    )


def build_client_partitions(
    train_set: Dataset,
    seed: int,
    spec: PartitionSpec = PartitionSpec(),
) -> List[List[int]]:
    partitions = iid_partition_indices(len(train_set), spec, seed)
    return select_client_partitions(partitions, spec, seed)


def build_client_loader(
    train_set: Dataset,
    indices: List[int],
    batch_size: int,
) -> DataLoader:
    subset = Subset(train_set, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)


def build_test_loader(test_set: Dataset, batch_size: int = 256) -> DataLoader:
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
