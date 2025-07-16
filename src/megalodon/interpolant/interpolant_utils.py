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


def float_time_to_index(time: torch.Tensor, num_time_steps: int) -> torch.Tensor:
    """
    Convert a float time value to a time index.

    Args:
        time (torch.Tensor): A tensor of float time values in the range [0, 1].
        num_time_steps (int): The number of discrete time steps.

    Returns:
        torch.Tensor: A tensor of time indices corresponding to the input float time values.
    """
    # Ensure time values are in the range [0, 1]
    time = torch.clamp(time, 0.0, 1.0)

    # Scale to the index range and round
    indices = torch.round(time * (num_time_steps - 1)).to(torch.int64)

    return indices


def log_1_min_a(a):
    return torch.log(1 - torch.exp(a) + 1e-40)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    return sample_index


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))