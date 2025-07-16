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


import numpy as np
import torch


def cosine_beta_schedule(params, num_diffusion_timesteps, s=0.008, nu=1.0, sqrt=False, return_alpha=False):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = num_diffusion_timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos((((x / steps) ** nu) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.clip(alphas, a_min=0.001, a_max=1.0)  #! should this be 0.999 like EQ
    if sqrt:
        alphas = np.sqrt(alphas)
    if return_alpha:
        return torch.tensor(alphas), torch.tensor(1 - alphas)
    return 1 - alphas


def cosine_beta_schedule_fm(params, num_diffusion_timesteps, s=0.008, nu=1.0):
    """
    cosine schedule
    as proposed in FlowMol
    """
    steps = num_diffusion_timesteps + 1
    x = np.linspace(0, steps, steps)
    t = x / steps
    alphas = 1 - np.cos((t**nu + s) / (1 + s) * np.pi * 0.5) ** 2
    t = torch.clamp_(t, min=1e-9)
    alpha_prime = np.sin(np.pi * (t + s) ** nu / (1 + s)) * (np.pi / 2) * (nu * (t + s) ** (nu - 1)) / (1 + s)
    return alphas, alpha_prime


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule_eq(params, num_diffusion_timesteps, s=0.008, nu=1.0):
    steps = num_diffusion_timesteps + 2
    x = torch.linspace(0, num_diffusion_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    ### new included
    alphas_cumprod = torch.from_numpy(clip_noise_schedule(alphas_cumprod, clip_value=0.05))
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = alphas.clip(min=0.001)
    betas = 1 - alphas
    betas = betas[1:]  #! Cut the extra piece that EQ skipped so we can iterate [0, 499] instead of [1, 500]
    betas = torch.clip(betas, 0.0, 0.999).float()
    return 1 - betas, betas
