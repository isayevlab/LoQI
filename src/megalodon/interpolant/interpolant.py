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
from torch import nn

from megalodon.interpolant.interpolant_utils import float_time_to_index


class Interpolant(nn.Module):
    def __init__(
        self,
        prior_type: str,
        solver_type: str = "sde",
        timesteps: int = 500,
        time_type: str = 'discrete',
    ):
        super(Interpolant, self).__init__()
        self.prior_type = prior_type
        self.timesteps = timesteps
        self.solver_type = solver_type
        self.time_type = time_type

    def sample_time(self, num_samples, method='uniform', device='cpu', mean=0, scale=0.81, min_t=0):
        if self.time_type == "continuous":
            return self.sample_time_continuous(num_samples, method, device, mean, scale, min_t)
        else:
            return self.sample_time_idx(num_samples, method, device, mean, scale)

    def sample_time_idx(self, num_samples, method, device='cpu', mean=0, scale=0.81):
        if method == 'symmetric':
            time_step = torch.randint(0, self.timesteps, size=(num_samples // 2 + 1,))
            time_step = torch.cat([time_step, self.timesteps - time_step - 1], dim=0)[:num_samples]

        elif method == 'uniform':
            time_step = torch.randint(0, self.timesteps, size=(num_samples,))

        elif method == "stab_mode":  #! converts uniform to Stability AI mode distribution

            def fmode(u: torch.Tensor, s: float) -> torch.Tensor:
                return 1 - u - s * (torch.cos((torch.pi / 2) * u) ** 2 - 1 + u)

            time_step = float_time_to_index(fmode(torch.rand(num_samples), scale), self.timesteps)
        elif (
            method == 'logit_normal'
        ):  # see Figure 11 https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf
            time_step = float_time_to_index(
                torch.sigmoid(torch.normal(mean=mean, std=scale, size=(num_samples,))), self.timesteps
            )
        elif method == "beta":
            dist = torch.distributions.Beta(2.0, 1.0)
            time_step = float_time_to_index(dist.sample([num_samples]), self.timesteps)
        else:
            raise ValueError
        return time_step.to(device)

    def sample_time_continuous(self, num_samples, method, device='cpu', mean=0, scale=0.81, min_t=0):
        if method == 'symmetric':
            time_step = torch.rand(num_samples // 2 + 1)
            time_step = torch.cat([time_step, 1 - time_step], dim=0)[:num_samples]

        elif method == 'uniform':
            time_step = torch.rand(num_samples)
        elif method == "stab_mode":  #! converts uniform to Stability AI mode distribution

            def fmode(u: torch.Tensor, s: float) -> torch.Tensor:
                return 1 - u - s * (torch.cos((torch.pi / 2) * u) ** 2 - 1 + u)

            time_step = fmode(torch.rand(num_samples), scale)
        elif (
            method == 'logit_normal'
        ):  # see Figure 11 https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf
            time_step = torch.sigmoid(torch.normal(mean=mean, std=scale, size=(num_samples,)))
        elif method == "beta":  # From MolFlow
            dist = torch.distributions.Beta(2.0, 1.0)
            time_step = dist.sample([num_samples])
        else:
            raise ValueError
        if min_t > 0:
            time_step = time_step * (1 - 2 * min_t) + min_t
        return time_step.to(device)

    def loss_weight_t(self, time):
        return torch.clamp(self.snr(time), min=0.05, max=1.5)