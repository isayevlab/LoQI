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

inner_to_atomic_number = {
    0: 1,    # H
    1: 5,    # B
    2: 6,    # C
    3: 7,    # N
    4: 8,    # O
    5: 9,    # F
    6: 13,   # Al
    7: 14,   # Si
    8: 15,   # P
    9: 16,   # S
    10: 17,  # Cl
    11: 33,  # As
    12: 35,  # Br
    13: 53,  # I
    14: 80,  # Hg
    15: 83,  # Bi
    16: 34   # Se
}

import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor

class Forces(nn.Module):
    def __init__(self, module: nn.Module, x: str = 'coord', y: str = 'energy', key_out: str = 'forces'):
        super().__init__()
        self.module = module
        self.x = x
        self.y = y
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        data[self.x].requires_grad_(True)
        data = self.module(data)
        y = data[self.y]
        g = torch.autograd.grad([y.sum()], [data[self.x]], create_graph=self.training)[0]
        assert g is not None
        data[self.key_out] = -g
        torch.set_grad_enabled(prev)
        return data

class AIMNet2ForcesLoss:
    def __init__(self, model_path, charge="charges", max_forces=0.05, min_time=0.9, atomics="h", coord="x", weight=1.0):
        super().__init__()
        self.nnip = Forces(torch.load(model_path))
        self.max_forces = max_forces
        self.min_time = min_time
        self.atomics = atomics
        self.coord = coord
        self.charge = charge
        self.weight = weight

    def __call__(self, batch, out, time, ws_t, stage="train"): 
        coord = out[f"{self.coord}_hat"]
        atomics = batch[self.atomics]
        charge = batch[self.charge]

        device = coord.device

        self.nnip.to(device)

        # Prepare AIMNet2 input batch
        aimnet2_batch = self.prepare_aimnet2batch(coord, atomics, charge, batch["batch"])

        # Calculate forces using AIMNet2
        forces = self.nnip(aimnet2_batch)["forces"]

        # Compute force loss
        loss_forces = torch.sum(torch.sum(torch.square(forces), dim=-1), dim=-1)

        # Normalize loss by the number of atoms per molecule
        num_atoms = (aimnet2_batch["numbers"] > 0).sum(dim=-1)
        loss_forces = loss_forces / num_atoms

        # Apply constraints and weighting
        loss_forces[torch.isnan(loss_forces)] = 0.0
        loss_forces[loss_forces > self.max_forces] = 0.0
        loss_forces[time < self.min_time] = 0.0

        # Final weighted loss
        loss = (loss_forces * ws_t).mean()*self.weight
        return loss

    @staticmethod
    def prepare_aimnet2batch(coord, atomics, charge, batch_idx):
        """
        Prepare data for AIMNet2 input format.

        Args:
            coord (Tensor): Tensor of atomic coordinates [n_atoms, 3].
            atomics (Tensor): Tensor of atom types in source format.
            charge (Tensor): Tensor of atomic charges.
            batch_idx (Tensor): Batch indices for the atoms.

        Returns:
            Dict[str, Tensor]: AIMNet2-compatible batch.
        """
        device = coord.device

        # Convert atomics to atomic numbers
        atomic_numbers = torch.tensor([inner_to_atomic_number[a.item()] for a in atomics], device=device)

        # Create batch tensors
        n_molecules = batch_idx.max().item() + 1
        max_n_atoms = torch.bincount(batch_idx).max().item()

        batch_coord = torch.zeros((n_molecules, max_n_atoms, 3), device=device)
        batch_atomics = torch.zeros((n_molecules, max_n_atoms), device=device).long()
        batch_charge = torch.zeros(n_molecules, device=device).long()

        for i in range(n_molecules):
            mask = batch_idx == i
            n_atoms = mask.sum().item()
            batch_coord[i, :n_atoms] = coord[mask]
            batch_atomics[i, :n_atoms] = atomic_numbers[mask]
            batch_charge[i] = charge[mask].sum() - 2 * n_atoms

        return {"coord": batch_coord, "numbers": batch_atomics, "charge": batch_charge}
