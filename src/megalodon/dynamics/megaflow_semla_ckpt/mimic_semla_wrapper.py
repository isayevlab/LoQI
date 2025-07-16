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

from megalodon.dynamics.megaflow_semla_ckpt.mega_pyg import MegalodonDotFN


class MimicSemlaWrapper(MegalodonDotFN):
    def __init__(self, args_dict):
        super().__init__()

        self.dynamics = MegalodonDotFN(**args_dict)
        invariant_node_feat_dim = 256
        invariant_edge_feat_dim = 256
        N = 18

        self.edge_out_proj = nn.Sequential(
            nn.Linear(invariant_edge_feat_dim, invariant_edge_feat_dim),
            nn.SiLU(inplace=False),
            nn.Linear(invariant_edge_feat_dim, 5)
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(invariant_node_feat_dim, invariant_node_feat_dim),
            nn.SiLU(inplace=False),
            nn.Linear(invariant_node_feat_dim, N)
        )

        self.charge_head = nn.Sequential(
            nn.Linear(invariant_node_feat_dim, invariant_node_feat_dim),
            nn.SiLU(inplace=False),
            nn.Linear(invariant_node_feat_dim, 7)
        )

        self.x_cond = nn.Sequential(
            nn.Linear(2, 32, bias=False), nn.Identity(), nn.Linear(32, 1, bias=False))
        self.x_clamps = (-100, 100)
        self.h_cond = nn.Sequential(
            nn.Linear(2 * N, 64, bias=True), nn.SiLU(), nn.Linear(64, N))
        self.b_cond = nn.Sequential(
            nn.Linear(2 * 5, 64, bias=True), nn.SiLU(), nn.Linear(64, 5))

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
        atom_features = h_t[:, :16]  # First 16 are atom types

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

    def symmetrize_edge_attr(self, edge_index, edge_attr, num_nodes):
        # Step 1: Create a dense adjacency matrix
        adj_matrix = torch.zeros((num_nodes, num_nodes, edge_attr.shape[-1]), dtype=edge_attr.dtype,
                                 device=edge_attr.device)
        adj_matrix[edge_index[0], edge_index[1]] = edge_attr

        # Step 2: Symmetrize the adjacency matrix
        sym_adj_matrix = adj_matrix + adj_matrix.transpose(0, 1)
        updated_edge_attr = sym_adj_matrix[edge_index[0], edge_index[1]]
        return updated_edge_attr

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        # Make edges fully connected
        batch = batch.copy()

        # Remap atomic features
        batch['h_t'] = self.remap_atomic_features(batch['h_t'])
        if conditional_batch is None or len(conditional_batch) == 0:
            cond_coords = torch.zeros_like(batch["x_t"])
            cond_atomics = torch.zeros_like(batch["h_t"])
            cond_bonds = torch.zeros_like(batch["edge_attr_t"])
        else:
            cond_coords = conditional_batch["x_hat"]
            cond_atomics = conditional_batch["atomics"]
            cond_bonds = conditional_batch["bonds"]

        x = torch.stack([batch["x_t"], torch.clamp(cond_coords, min=-100, max=100)], dim=-1)
        batch["x_t"] = self.x_cond(x)[..., 0] + batch["x_t"]

        h = torch.cat((batch['h_t'], cond_atomics), dim=-1)
        batch['h_t'] = self.h_cond(h) + batch['h_t']

        b = torch.cat((batch["edge_attr_t"], cond_bonds), dim=-1)
        batch["edge_attr_t"] = self.b_cond(b) + batch["edge_attr_t"]

        out = self.dynamics.forward(
            batch=batch["batch"],
            X=batch["x_t"],
            H=batch["h_t"],
            E=batch["edge_attr_t"],
            E_idx=batch["edge_index"],
            t=time,
        )

        # Separate outputs and process through heads
        h_logits = out['h_logits']
        edge_attr_logits = out['edge_attr_logits']

        node_logits = self.classifier_head(h_logits)
        charge_logits = self.charge_head(h_logits)
        edge_index = batch["edge_index"]

        out["atomics"] = torch.softmax(node_logits, dim=-1)

        pred_edges = self.symmetrize_edge_attr(edge_index, edge_attr_logits, len(batch["h_t"]))

        edge_logits = self.edge_out_proj(pred_edges)

        out["bonds"] = torch.softmax(edge_logits, dim=-1)

        # Reorder logits back to the original order
        node_logits, charge_logits = self.reorder_logits_back(node_logits, charge_logits)

        # Concatenate outputs
        out['h_logits'] = torch.cat([node_logits, charge_logits], dim=-1)
        out['edge_attr_logits'] = edge_logits

        return out

    def load_checkpoint_to_model(self, ckpt_path, strict=True, ema=False):

        if ema:
            prefix = "ema_gen"
        else:
            prefix = "gen"
        # Load the state dictionary from the checkpoint
        checkpoint = torch.load(ckpt_path)["state_dict"]

        # Extract the dynamics submodule state dictionary
        dynamics_state_dict = {k.replace(f"{prefix}.dynamics.", ""): v for k, v in
                               checkpoint.items() if k.startswith(f"{prefix}.dynamics.")}

        # Load the dynamics submodule weights into model.dynamics.dynamics
        self.dynamics.load_state_dict(dynamics_state_dict, strict=strict)

        # Load edge_out_proj weights
        edge_out_proj_state_dict = {k.replace(f"{prefix}.edge_out_proj.", ""): v for k, v in
                                    checkpoint.items() if k.startswith(f"{prefix}.edge_out_proj.")}
        self.edge_out_proj.load_state_dict(edge_out_proj_state_dict, strict=strict)

        # Load classifier_head weights
        classifier_head_state_dict = {k.replace(f"{prefix}.classifier_head.", ""): v for k, v in
                                      checkpoint.items() if
                                      k.startswith(f"{prefix}.classifier_head.")}
        self.classifier_head.load_state_dict(classifier_head_state_dict, strict=strict)

        # Load charge_head weights
        charge_head_state_dict = {k.replace(f"{prefix}.charge_head.", ""): v for k, v in
                                  checkpoint.items() if k.startswith(f"{prefix}.charge_head.")}
        self.charge_head.load_state_dict(charge_head_state_dict, strict=strict)

        # Load edge_out_proj weights
        x_cond_dict = {k.replace(f"{prefix}.x_cond.", ""): v for k, v in checkpoint.items() if
                       k.startswith(f"{prefix}.x_cond.")}
        self.x_cond.load_state_dict(x_cond_dict, strict=strict)

        # Load classifier_head weights
        h_cond_dict = {k.replace(f"{prefix}.h_cond.", ""): v for k, v in checkpoint.items() if
                       k.startswith(f"{prefix}.h_cond.")}
        self.h_cond.load_state_dict(h_cond_dict, strict=strict)

        # Load charge_head weights
        b_cond_dict = {k.replace(f"{prefix}.b_cond.", ""): v for k, v in checkpoint.items() if
                       k.startswith("gen.b_cond.")}
        self.b_cond.load_state_dict(b_cond_dict, strict=strict)
        print("Checkpoint loaded successfully.")
