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


from typing import Dict, List

import torch
import torch.nn as nn


class BaseSelfConditioningModule(nn.Module):
    """
    A base module for self-conditioning. This module handles the fusion of original and conditional features.

    Attributes:
        modules_dict (nn.ModuleDict): A dictionary of submodules for each feature.
        keys (List[str]): A list of feature names.
        vector_mask (List[bool]): A list indicating if a feature is a vector.
        fuse_softmax (List[bool]): A list indicating if softmax should be applied during fusion.
        clamps (List[Tuple[Optional[float], Optional[float]]]): A list of clamping ranges for the features.
    """

    def __init__(self, variables: List[Dict[str, any]]):
        """
        Initializes the BaseSelfConditioningModule.

        Args:
            variables (List[Dict[str, any]]): A list of dictionaries containing variable information.
        """
        super().__init__()
        self.modules_dict = nn.ModuleDict()
        self.keys = []
        self.vector_mask = []
        self.fuse_softmax = []
        self.clamps = []

        for var in variables:
            self.keys.append(var["variable_name"])
            self.vector_mask.append(var["vector"])
            self.fuse_softmax.append(var["fuse_softmax"])
            self.clamps.append((var["clamp_min"], var["clamp_max"]))

            if not self.vector_mask[-1]:
                inp, out = 2 * var["inp_dim"], var["inp_dim"]
                activation = nn.SiLU() if var["fuse_softmax"] else nn.Identity()
                bias = True
            else:
                inp, out = 2, 1  # for 3D coordinates
                activation = nn.Identity()
                bias = False
            self.modules_dict[self.keys[-1]] = nn.Sequential(
                nn.Linear(inp, var["hidden_dims"], bias=bias), activation, nn.Linear(var["hidden_dims"], out)
            )

    def forward(self, batch: Dict[str, torch.Tensor], cond_batch: Dict[str, torch.Tensor]):
        """
        Forward pass for the self-conditioning module.

        Args:
            batch (Dict[str, torch.Tensor]): The original batch of features.
            cond_batch (Dict[str, torch.Tensor]): The conditional batch of features.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: The updated batch and non-fused features.
        """
        non_fused_variables = {}
        for key, vec, fuse, clamp in zip(self.keys, self.vector_mask, self.fuse_softmax, self.clamps):
            non_fused_variables[f"{key}_t"] = batch[f"{key}_t"].clone()
            x = batch[f"{key}_t"]

            if f"{key}_logits" in cond_batch and not fuse:
                x_cond = cond_batch[f"{key}_logits"]
            else:
                x_cond = cond_batch[f"{key}_hat"]

            if clamp[0] is not None or clamp[1] is not None:
                x_cond = torch.clamp(x_cond, min=clamp[0], max=clamp[1])

            if not vec:
                x = torch.cat([x, x_cond], dim=-1)
                x = self.modules_dict[key](x)
            else:
                x = torch.stack([x, x_cond], dim=-1)
                x = self.modules_dict[key](x)[..., 0]
            batch[f"{key}_t"] += x

        return batch, non_fused_variables


class SelfConditioningBuilder:
    """
    A builder class for creating self-conditioning modules based on configuration.

    Attributes:
        self_cond_classes (Dict[str, nn.Module]): A dictionary mapping config names to self-conditioning classes.
    """

    def __init__(self):
        """Initializes the SelfConditioningBuilder with default self-conditioning classes."""
        self.self_cond_classes = {"base": BaseSelfConditioningModule}

    def create_self_cond(self, config: Dict[str, any]) -> nn.Module:
        """
        Creates a self-conditioning module based on the provided configuration.

        Args:
            config (Dict[str, any]): A dictionary containing the configuration.

        Returns:
            nn.Module: An instance of the self-conditioning module.

        Raises:
            ValueError: If the config name is not recognized.
        """
        config_name = config["name"]
        variables = config["variables"]
        self_cond_class = self.self_cond_classes.get(config_name.lower())
        if self_cond_class is None:
            raise ValueError(f"Unknown self_cond config name: {config_name}")

        return self_cond_class(variables)

    def register_self_cond(self, config_name: str, self_cond_class: nn.Module):
        """
        Registers a new self-conditioning class with the builder.

        Args:
            config_name (str): The name of the configuration.
            self_cond_class (nn.Module): The class of the self-conditioning module.
        """
        self.self_cond_classes[config_name.lower()] = self_cond_class


# Example test function
def test_self_conditioning():
    # Example configuration
    config = {
        "name": "base",
        "variables": [
            {
                "variable_name": "h",
                "inp_dim": 256,
                "hidden_dims": 64,
                "fuse_softmax": True,
                "vector": False,
                "clamp_min": None,
                "clamp_max": None,
            },
            {
                "variable_name": "charges",
                "inp_dim": 16,
                "hidden_dims": 64,
                "fuse_softmax": True,
                "vector": False,
                "clamp_min": None,
                "clamp_max": None,
            },
            {
                "variable_name": "edge_attr",
                "inp_dim": 128,
                "hidden_dims": 64,
                "fuse_softmax": True,
                "vector": False,
                "clamp_min": None,
                "clamp_max": None,
            },
            {
                "variable_name": "x",
                "inp_dim": 3,
                "hidden_dims": 64,
                "fuse_softmax": False,
                "vector": True,
                "clamp_min": -100,
                "clamp_max": 100,
            },
        ],
    }

    # Instantiate the builder
    builder = SelfConditioningBuilder()

    # Create the self-conditioning module
    self_cond = builder.create_self_cond(config)

    # Example batch data
    batch = {
        "x_t": torch.randn(10, 3),
        "h_t": torch.randn(10, 256),
        "charges_t": torch.randn(10, 16),
        "edge_attr_t": torch.randn(10, 128),
        "batch": torch.randint(0, 2, (10,)),
        "time": torch.randn(10),
    }
    cond_batch = {
        "x_hat": torch.randn(10, 3),
        "h_hat": torch.randn(10, 256),
        "charges_hat": torch.randn(10, 16),
        "edge_attr_hat": torch.randn(10, 128),
        "batch": torch.randint(0, 2, (10,)),
        "time": torch.randn(10),
    }
    batch, non_fused_variables = self_cond(batch, cond_batch)
    print(batch)


# Run the test function
if __name__ == "__main__":
    test_self_conditioning()
