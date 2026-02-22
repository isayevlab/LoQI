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

"""
Atom-count-aware batching utilities for LoQI inference.

Problem: molecules have highly variable sizes (10–100+ atoms with Hs).
PyG's DataLoader batches by graph count, which can lead to:
- Batches with many small molecules (wasted parallelism) OR
- Batches with one huge molecule and many idle cores

Solution: First-Fit-Decreasing (FFD) bin-packing by atom count.
Sort molecules largest-first, then greedily fill bins up to max_atoms_per_batch.
This minimises the number of batches and reduces wasted GPU memory.
"""

from typing import List

from torch_geometric.data import Batch, Data


def ffd_pack_indices(atom_counts: List[int], max_atoms_per_batch: int) -> List[List[int]]:
    """
    First-Fit-Decreasing bin-packing by atom count.

    Args:
        atom_counts:         Atom count for each Data item.
        max_atoms_per_batch: Soft upper bound on total atoms per batch.
            A single molecule whose atom count already exceeds this value
            is placed in its own bin (the bound is never enforced across
            bin boundaries, only during placement decisions).

    Returns:
        List of index lists, one per bin.
        Each inner list contains indices into the original data_list.
        Indices within each bin are sorted ascending for reproducibility.
    """
    if not atom_counts:
        return []

    # Sort descending by atom count
    sorted_indices = sorted(range(len(atom_counts)), key=lambda i: atom_counts[i], reverse=True)

    bins: List[List] = []    # each entry: [current_total_atoms, [idx, ...]]

    for idx in sorted_indices:
        n_atoms = atom_counts[idx]
        placed = False
        for b in bins:
            if b[0] + n_atoms <= max_atoms_per_batch:
                b[0] += n_atoms
                b[1].append(idx)
                placed = True
                break
        if not placed:
            bins.append([n_atoms, [idx]])

    return [sorted(b[1]) for b in bins]


def pack_batches(data_list: List[Data], max_atoms_per_batch: int) -> List[Batch]:
    """
    Pack data_list into Batch objects using FFD atom-count bin-packing.

    Args:
        data_list:           List of PyG Data objects to batch.
        max_atoms_per_batch: Maximum total atoms per Batch.

    Returns:
        List of Batch objects, one per bin.
        Items within each Batch appear in ascending original-index order.
    """
    if not data_list:
        return []
    atom_counts = [data.x.size(0) for data in data_list]
    bin_indices = ffd_pack_indices(atom_counts, max_atoms_per_batch)
    return [Batch.from_data_list([data_list[i] for i in indices]) for indices in bin_indices]
