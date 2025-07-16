# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import Iterator, List, Optional, Set

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader, DynamicBatchSampler


def effective_batch_size(
    max_size: int, reference_batch_size: int, reference_size: int = 20, sampling: bool = False
) -> int:
    """Calculates the effective batch size based on the maximum graph size.

    Args:
        max_size (int): The maximum number of nodes in the graphs.
        reference_batch_size (int): The reference batch size.
        reference_size (int): The reference size. Defaults to 20.
        sampling (bool): Whether to apply a sampling factor. Defaults to False.

    Returns:
        int: The calculated effective batch size.
    """
    x = reference_batch_size * (reference_size / max_size) ** 2
    return math.floor(1.8 * x) if sampling else math.floor(x)


class AdaptiveBatchSampler(DynamicBatchSampler):
    """A batch sampler that adapts batch size based on graph sizes.

    Args:
        dataset (Dataset): The dataset containing the graphs.
        reference_batch_size (int): The reference batch size.
        shuffle (bool): Whether to shuffle the dataset before sampling. Defaults to False.
        max_num_nodes (Optional[int]): Maximum number of nodes in a graph. Defaults to None.
        num_steps (Optional[int]): The number of steps for sampling. Defaults to None.
        reference_size (Optional[int]): The reference size. Defaults to None.
        lookahead_size (int): The lookahead size for batching. Defaults to 800.
    """

    def __init__(
        self,
        dataset: Dataset,
        reference_batch_size: int,
        shuffle: bool = False,
        max_num_nodes: Optional[int] = None,
        num_steps: Optional[int] = None,
        reference_size: Optional[int] = None,
        lookahead_size: int = 800,
    ):
        self.dataset = dataset
        self.shuffle = shuffle

        # Determine the maximum number of nodes in the dataset if not provided
        if max_num_nodes is None:
            max_num_nodes = np.max([data.num_nodes for data in dataset])
        self.max_num_nodes = max_num_nodes

        self.num_steps = num_steps
        self.max_steps = num_steps or len(dataset)
        self.lookahead_size = lookahead_size
        self.reference_batch_size = reference_batch_size

        # Calculate the reference size based on the median number of nodes in graphs
        if reference_size is None:
            reference_size = math.floor(
                np.median([data.num_nodes for data in dataset if data.num_nodes <= self.max_num_nodes])
            )
        self.reference_size = reference_size

    def __iter__(self) -> Iterator[List[int]]:
        """Generates batches of indices for the dataset.

        Yields:
            Iterator[List[int]]: A list of indices for each batch.
        """
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        samples: List[int] = []
        num_steps: int = 0
        skipped: Set[int] = set()
        pointer = 0

        while (pointer != len(self.dataset) or len(skipped) > 0) and num_steps < self.max_steps:
            # Determine the current lookahead window size and indices
            window_indices = list(skipped)
            new_pointer = pointer

            # Collect indices within the lookahead window that satisfy the max_num_nodes condition
            while (len(window_indices) < self.lookahead_size) and (new_pointer < len(self.dataset)):
                if self.dataset[indices[new_pointer]].num_nodes <= self.max_num_nodes:
                    window_indices.append(indices[new_pointer])
                new_pointer += 1

            # If no indices are collected, continue to the next iteration
            if len(window_indices) == 0:
                continue

            # Get the sizes of graphs in the window
            graph_sizes = torch.tensor([self.dataset[idx].num_nodes for idx in window_indices])

            # Randomly choose a size to determine the max size for this batch
            random = torch.randint(0, len(graph_sizes), size=(1,)).item()
            srted, argsort = torch.sort(graph_sizes)
            max_size = min(max(graph_sizes), self.max_num_nodes, graph_sizes[random] + 5)
            max_size = max(max_size, 9)

            # Calculate effective batch size
            ebs = effective_batch_size(max_size, self.reference_batch_size, reference_size=self.reference_size)
            ebs = max(ebs, 1)  # Ensure that batch size is at least 1

            # Determine which indices to keep based on the max size
            max_index = torch.nonzero(srted <= max_size).max().item()
            min_index = max(0, max_index - ebs)
            indices_to_keep = set(argsort[min_index : max_index + 1].tolist())

            # Add additional indices if the effective batch size is not reached
            if max_index < ebs:
                for index in range(max_index + 1, len(srted)):
                    size = srted[index].item()
                    if size <= self.max_num_nodes:
                        potential_ebs = effective_batch_size(
                            size, self.reference_batch_size, reference_size=self.reference_size
                        )
                        if len(indices_to_keep) < potential_ebs:
                            indices_to_keep.add(argsort[index].item())

            samples = [window_indices[i] for i in indices_to_keep]
            for idx in samples:
                if idx in skipped:
                    skipped.remove(idx)

            # Update the pointer and skipped indices for the next iteration
            for i in range(pointer, new_pointer):
                if indices[i] not in samples and self.dataset[indices[i]].num_nodes <= self.max_num_nodes:
                    skipped.add(indices[i])

            pointer = new_pointer
            if len(samples) > 0:
                yield samples
                samples = []
                num_steps += 1

    def __len__(self):
        if self.num_steps is not None:
            return self.num_steps
        else:
            return len(self.dataset)


def test_adaptive_data_loader():
    """Tests the AdaptiveBatchSampler with a MoleculeDataset."""
    from molecule_datamodule import MoleculeDataset

    dataset = MoleculeDataset(
        root="/home/fnikitin/data/pyg_geom_drug", processed_folder="processed", split="test", only_stats=False
    )
    batch_size = 800
    batch_sampler = AdaptiveBatchSampler(dataset, reference_batch_size=batch_size, shuffle=True, reference_size=20)

    # Create the DataLoader with the custom BatchSampler
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    import time

    # Measure the average batching time
    t0 = time.time()
    num_batches = 0

    for batch in data_loader:
        num_batches += 1
    t1 = time.time()

    print(f"Average batching time is {(t1 - t0)/num_batches:.6f} s per batch, number of batches - {num_batches}")

    smiles = []
    batch_sizes = []
    for batch in data_loader:
        batch_sizes.append(len(batch))
        smiles.extend(batch.smiles)

    dataset_smiles = [d.smiles for d in dataset]

    # Check that all smiles strings are included and batch sizes are correct
    assert set(dataset_smiles) == set(smiles)
    assert len(dataset_smiles) == len(smiles)

    print(f"The median batch size is {np.median(batch_sizes)}")

    # Test with a maximum number of nodes constraint
    batch_size = 256
    max_num_nodes = 40
    batch_sampler = AdaptiveBatchSampler(
        dataset, reference_batch_size=batch_size, shuffle=True, lookahead_size=800, max_num_nodes=max_num_nodes
    )

    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    batch_sizes = []
    num_nodes = []

    for batch in data_loader:
        batch_sizes.append(len(batch))
        for mol in batch.mol:
            num_nodes.append(len(mol.GetAtoms()))

    assert max(num_nodes) <= max_num_nodes
    print(f"The median batch size is {np.median(batch_sizes)} with maximum of {max_num_nodes} nodes")


if __name__ == "__main__":
    test_adaptive_data_loader()
