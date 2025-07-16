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
import torch.nn.functional as F
from torch import nn

from omegaconf import OmegaConf
from megalodon.dynamics.megaflow_semla_ckpt.semla import SemlaGenerator, EquiInvDynamics


class MimicOriginalSemlaWrapper(SemlaGenerator):
    def __init__(self, args_dict):
        args_dict = OmegaConf.to_container(args_dict, resolve=True) 
        dynamics_args = args_dict["dynamics_args"]
        dynamics = EquiInvDynamics(**dynamics_args)
        args_dict["dynamics"] = dynamics
        del args_dict["dynamics_args"]
        super().__init__(**args_dict)

        self.full_atom_encoder = {
            "H": 0, "B": 1, "C": 2, "N": 3, "O": 4, "F": 5,
            "Al": 6, "Si": 7, "P": 8, "S": 9, "Cl": 10,
            "As": 11, "Br": 12, "I": 13, "Hg": 14, "Bi": 15
        }

        self.reordered_atom_encoder = {
            "<PAD>": 0, "<MASK>": 1,
            "H": 2, "C": 3, "N": 4, "O": 5, "F": 6, "P": 7, "S": 8, "Cl": 9,
            "Br": 10, "B": 11, "Al": 12, "Si": 13, "As": 14, "I": 15, "Hg": 16, "Bi": 17
        }

        self.mapping = {v: self.reordered_atom_encoder[k] for k, v in
                        self.full_atom_encoder.items()}
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

        self.charge_order = [-2, -1, 0, 1, 2, 3]
        self.int_charge_order = [0, 1, 2, 3, -1, -2, -3]
        self.charge_reorder = [5, 4, 0, 1, 2, 3]

    def remap_atomic_features(self, h_t):
        # Separate atom types and charges
        atom_features = h_t[:, :16]  

        # Convert one-hot to indices
        atom_indices = atom_features.argmax(dim=-1)

        # Remap atom indices to the new order
        remapped_atoms = torch.tensor([
            self.mapping[idx.item()] for idx in atom_indices
        ], device=atom_indices.device)

        # Convert back to one-hot
        remapped_atom_one_hot = F.one_hot(remapped_atoms, num_classes=18).float()

        return remapped_atom_one_hot

    def reorder_logits_back(self, node_logits, charge_logits):

        node_logits = node_logits[:, list(self.mapping.values())]
        charge_logits = charge_logits[:, self.charge_reorder]

        return node_logits, charge_logits

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        out_h = torch.zeros_like(batch["h_t"])
        batch = batch.copy()

        batch_counts = torch.bincount(batch["batch"])
        max_len = batch_counts.max().item()
        batch_size = torch.max(batch["batch"]) + 1
        batch['h_t'] = self.remap_atomic_features(batch['h_t'])

        batch["h_t"] = torch.cat((time[batch["batch"]].view(-1, 1), batch['h_t']), dim=-1)

        E = torch.zeros((batch_size, max_len, max_len, batch["edge_attr_t"].shape[-1]),
                        device=batch["edge_attr_t"].device, dtype=batch["edge_attr_t"].dtype)
        H = torch.zeros((batch_size, max_len, batch["h_t"].shape[-1]),
                        device=batch["h_t"].device, dtype=batch["h_t"].dtype)
        X = torch.zeros((batch_size, max_len, batch["x_t"].shape[-1]),
                        device=batch["x_t"].device, dtype=batch["x_t"].dtype)

        M = torch.zeros((batch_size, max_len),
                         device=batch["h_t"].device, dtype=batch["h_t"].dtype)

        if conditional_batch is None or len(conditional_batch) == 0:
            cond_E = torch.zeros_like(E)
            cond_X = torch.zeros_like(X)
            cond_H = torch.zeros((batch_size, max_len, batch["h_t"].shape[-1] - 1),
                        device=batch["h_t"].device, dtype=batch["h_t"].dtype)
        else:
            cond_E = conditional_batch["cond_E"]
            cond_X = conditional_batch["cond_X"]
            cond_H = conditional_batch["cond_H"]


        b_index = batch["batch"]
        e_index = batch["edge_index"]
        e_batch = b_index[e_index[0]]
        for i in range(batch_size):
            mask = (b_index == i)
            e_mask = (e_batch == i)
            local_e_index = e_index[:, e_mask] - e_index[:, e_mask].min()

            E[i, local_e_index[0], local_e_index[1]] = batch["edge_attr_t"][e_mask]
            H[i, :mask.sum()] = batch["h_t"][mask]
            X[i, :mask.sum()] = batch["x_t"][mask]
            M[i, :mask.sum()] = 1.

        M = M.long()
        pred_coords, type_logits, edge_logits, charge_logits = super().forward(X, H, E, cond_X, cond_H, cond_E, M)

        out = {"cond_X": pred_coords, "cond_H":  F.softmax(type_logits, dim=-1), "cond_E": F.softmax(edge_logits, dim=-1)}

        out_x = torch.zeros_like(batch["x_t"])
        out_edge_attr = torch.zeros_like(batch["edge_attr_t"])

        for i in range(batch_size):
            mask = (b_index == i)
            e_mask = (e_batch == i)
            local_e_index = e_index[:, e_mask] - e_index[:, e_mask].min()

            out_x[mask] = pred_coords[i, :mask.sum()]
            h_i, c_i = self.reorder_logits_back(type_logits[i, :mask.sum()], charge_logits[i, :mask.sum()])
            out_h[mask] = torch.cat([h_i, c_i], dim=-1)
            out_edge_attr[e_mask] = edge_logits[i, local_e_index[0], local_e_index[1]]

        out["x_hat"] = out_x
        out['h_logits'] = out_h
        out['edge_attr_logits'] = out_edge_attr

        return out


    def load_checkpoint_to_model(self, ckpt_path, strict=True, ema=False):

        if ema:
            prefix = "ema_gen"
        else:
            prefix = "gen"
        # Load the state dictionary from the checkpoint
        checkpoint = torch.load(ckpt_path)["state_dict"]

        state_dict = {k.replace(f"{prefix}.module.", ""): v for k, v in checkpoint.items() if k.startswith(f"{prefix}.module")}

        self.load_state_dict(state_dict)

        print("Checkpoint loaded successfully.")
