"""Training samplers for deliberate molecule-size rebalancing."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch.utils.data import DistributedSampler
from torch_geometric.data import InMemoryDataset


class SizeTailSampler(DistributedSampler):
    """Repeat tiny, small and large molecules by exact integer factors.

    Unlike weighted sampling with replacement, every ordinary molecule appears
    once per epoch and every tail molecule appears its requested number of
    times before distributed padding and MiDi's adaptive batch selection.
    Subclassing DistributedSampler prevents Lightning from sharding it twice.
    Validation and test loaders should not use this sampler.
    When configured, the tiny tier overrides the small tier's factor.
    The optional diatomic factor overrides every tier for exactly two nodes,
    including explicit hydrogen atoms. None preserves the existing behavior.
    """

    def __init__(
        self,
        dataset,
        *,
        small_max_nodes: int = 10,
        small_factor: int = 1,
        tiny_max_nodes: int | None = None,
        tiny_factor: int = 1,
        diatomic_factor: int | None = None,
        large_min_nodes: int = 110,
        large_factor: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
    ) -> None:
        if small_max_nodes < 0 or large_min_nodes < 0:
            raise ValueError("size thresholds must be non-negative")
        if small_max_nodes >= large_min_nodes:
            raise ValueError("small_max_nodes must be less than large_min_nodes")
        if not isinstance(small_factor, int) or small_factor < 1:
            raise ValueError("small_factor must be an integer >= 1")
        if not isinstance(large_factor, int) or large_factor < 1:
            raise ValueError("large_factor must be an integer >= 1")
        if not isinstance(tiny_factor, int) or tiny_factor < 1:
            raise ValueError("tiny_factor must be an integer >= 1")
        if tiny_max_nodes is not None and (
            not isinstance(tiny_max_nodes, int) or not 0 <= tiny_max_nodes <= small_max_nodes
        ):
            raise ValueError("tiny_max_nodes must be an integer between 0 and small_max_nodes")
        if tiny_max_nodes is None and tiny_factor != 1:
            raise ValueError("tiny_factor requires tiny_max_nodes")
        if diatomic_factor is not None and (type(diatomic_factor) is not int or diatomic_factor < 1):
            raise ValueError("diatomic_factor must be an integer >= 1")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        # Iterating a PyG InMemoryDataset reconstructs and caches every graph.
        # The collated node offsets already contain exactly the needed sizes.
        if isinstance(dataset, InMemoryDataset) and dataset.transform is None:
            if dataset.slices is not None and "x" in dataset.slices:
                sizes = dataset.slices["x"].diff().to(dtype=torch.long, device="cpu")
                if dataset._indices is not None:
                    sizes = sizes[torch.as_tensor(list(dataset.indices()), dtype=torch.long)]
            else:
                sizes = torch.tensor([int(data.num_nodes) for data in dataset])
        else:
            sizes = torch.tensor([int(data.num_nodes) for data in dataset])
        factors = torch.ones(len(sizes), dtype=torch.long)
        factors[sizes <= small_max_nodes] = small_factor
        # The tiny tier overrides the small factor; it does not multiply it.
        if tiny_max_nodes is not None:
            factors[sizes <= tiny_max_nodes] = tiny_factor
        factors[sizes >= large_min_nodes] = large_factor
        # Exactly two total atoms, including hydrogens. Override, never multiply.
        if diatomic_factor is not None:
            factors[sizes == 2] = diatomic_factor
        expanded = torch.repeat_interleave(torch.arange(len(sizes)), factors)
        self.expanded_indices = expanded

        if drop_last:
            self.total_size = len(expanded) - (len(expanded) % num_replicas)
        else:
            self.total_size = math.ceil(len(expanded) / num_replicas) * num_replicas
        self.num_samples = self.total_size // num_replicas

    def __iter__(self) -> Iterator[int]:
        indices = self.expanded_indices
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(indices), generator=generator)
            indices = indices[order]

        if self.drop_last:
            indices = indices[: self.total_size]
        elif len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            indices = torch.cat((indices, indices.repeat(math.ceil(padding / len(indices)))[:padding]))

        return iter(indices[self.rank : self.total_size : self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic shuffle for a new training epoch."""
        self.epoch = epoch


__all__ = ["SizeTailSampler"]
