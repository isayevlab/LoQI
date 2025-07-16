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

import torch

from megalodon.data.random_rotations import random_rotations
from torch_geometric.utils import coalesce
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean


def make_graph_fully_connected(edge_index, edge_attr, batch):
    # Load bond information from the dataloader
    bond_edge_index, bond_edge_attr = sort_edge_index(
        edge_index=edge_index, edge_attr=edge_attr, sort_by_row=False
    )
    bond_edge_index, bond_edge_attr = coalesce(bond_edge_index, bond_edge_attr,
                                               reduce="min")

    # Create Fully Connected Graph instead
    edge_index_global = (
        torch.eq(batch.unsqueeze(0),
                 batch.unsqueeze(-1)).int().fill_diagonal_(0)
    )
    edge_index_global, _ = dense_to_sparse(edge_index_global)
    edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
    edge_attr_tmp = torch.full(
        size=(edge_index_global.size(-1),),
        fill_value=0,
        device=edge_index_global.device,
        dtype=torch.long,
    )
    edge_index_global = torch.cat([edge_index_global, bond_edge_index], dim=-1)
    edge_attr_tmp = torch.cat([edge_attr_tmp, bond_edge_attr], dim=0)

    edge_index_global, edge_attr_global = coalesce(edge_index_global, edge_attr_tmp,
                                                   reduce="max")

    edge_index_global, edge_attr_global = sort_edge_index(
        edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
    )
    return edge_index_global, edge_attr_global



class BatchPreProcessor:
    def __init__(self, 
                 aug_rotations=False,
                 scale_coords=1.0):
        
        self.aug_rotations = aug_rotations
        self.scale_coords = scale_coords

    def __call__(self, batch):
        """Custom collate function to apply augmentations."""
        batch_size = torch.max(batch.batch) + 1
        if self.aug_rotations and batch.pos is not None:
            rotations = random_rotations(batch_size, batch.pos.dtype, batch.pos.device)
            rotations = rotations[batch.batch]
            batch.pos = torch.bmm(rotations, batch.pos.unsqueeze(-1)).squeeze(-1)

        if self.scale_coords and batch.pos is not None:
            batch.pos = batch.pos / self.scale_coords

        batch.h = batch.x
        batch.x = batch.pos
        batch.pos = None

        for key in ["charges", "edge_attr", "h"]:
            batch[key] = batch[key].long()

        batch['edge_index'], batch['edge_attr'] = make_graph_fully_connected(edge_index=batch["edge_index"],
                                                                             edge_attr=batch["edge_attr"],
                                                                             batch=batch["batch"])
        batch['x'] = (
                batch['x'] -
                scatter_mean(batch['x'], index=batch.batch, dim=0, dim_size=batch_size)[
                    batch.batch]
        )
        return batch