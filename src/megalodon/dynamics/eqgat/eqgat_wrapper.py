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

from megalodon.dynamics.eqgat.eqgat_denoising_model import DenoisingEdgeNetwork

class EQGATWrapper(DenoisingEdgeNetwork):
    """A wrapper class for the EQGAT model."""

    def __init__(self, args_dict, time_type="continuous", timesteps=None, random_learning=False):
        """
        Initializes the EQGATWrapper.

        Args:
            args_dict (dict): A dictionary of arguments for initializing the EQGAT model.
        """
        self.args = args_dict
        self.time_type = time_type
        self.timesteps = timesteps
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        """
        Forward pass of the EQGAT model.

        Args:
            batch (torch_geometric.data.Batch): The input batch.
            time (Tensor): The time tensor.

        Returns:
            dict: The output of the EQGAT model.
        """
        timesteps = timesteps if timesteps is not None else self.timesteps
        if self.time_type == "discrete" and timesteps is not None:
            time = (timesteps - time.float()) / timesteps
        temb = time.clamp(min=0.001)
        temb = temb.unsqueeze(dim=1)
        x, pos, edge_attr_global = batch["h_t"].clone(), batch["x_t"].clone(), batch["edge_attr_t"].clone()

        _out = super().forward(
            x=x,
            t=temb,  # should be in [0, 1]?
            pos=pos,
            edge_index_global=batch["edge_index"],
            edge_attr_global=edge_attr_global,
            batch=batch["batch"],
            batch_edge_global=batch["batch"][batch["edge_index"][0]],
        )

        out = {
            "x_hat": _out["coords_pred"],
            "h_logits": _out["atoms_pred"],
            "edge_attr_logits": _out["bonds_pred"],
        }
        return out