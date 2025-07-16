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
from copy import deepcopy
from omegaconf import OmegaConf
from megalodon.dynamics.megaflow_semla_ckpt.semla import SemlaGenerator, EquiInvDynamics


class SemlaWrapper(SemlaGenerator):
    def __init__(self, args_dict):
        args_dict = OmegaConf.to_container(args_dict, resolve=True)
        dynamics_args = args_dict["dynamics_args"]
        dynamics = EquiInvDynamics(**dynamics_args)
        args_dict["dynamics"] = dynamics
        del args_dict["dynamics_args"]
        self.vocab_size = args_dict["vocab_size"]
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        out_h = torch.zeros_like(batch["h_t"])
        batch = deepcopy(batch)

        batch_counts = torch.bincount(batch["batch"])
        max_len = batch_counts.max().item()
        batch_size = torch.max(batch["batch"]) + 1
        batch["h_t"] = batch["h_t"][:, :self.vocab_size]

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

        if self.training and torch.rand(1).item() > 0.5:
            with torch.no_grad():
                cond_X, cond_H, cond_E, _ = super().forward(X, H, E,
                                                               cond_X,
                                                               cond_H,
                                                               cond_E, M)
                cond_H = torch.softmax(cond_H, dim=-1).detach()
                cond_E = torch.softmax(cond_E, dim=-1).detach()
                cond_X = cond_X.detach()

        pred_coords, type_logits, edge_logits, charge_logits = super().forward(X, H, E, cond_X, cond_H, cond_E, M)

        out = {"cond_X": pred_coords.detach(),
               "cond_H":  F.softmax(type_logits, dim=-1).detach(),
               "cond_E": F.softmax(edge_logits, dim=-1).detach()}

        out_x = torch.zeros_like(batch["x_t"])
        out_edge_attr = torch.zeros_like(batch["edge_attr_t"])

        for i in range(batch_size):
            mask = (b_index == i)
            e_mask = (e_batch == i)
            local_e_index = e_index[:, e_mask] - e_index[:, e_mask].min()

            out_x[mask] = pred_coords[i, :mask.sum()]
            out_h[mask] = torch.cat([type_logits[i, :mask.sum()], charge_logits[i, :mask.sum()]], dim=-1)
            out_edge_attr[e_mask] = edge_logits[i, local_e_index[0], local_e_index[1]]

        out["x_hat"] = out_x
        out['h_logits'] = out_h
        out['edge_attr_logits'] = out_edge_attr

        return out
