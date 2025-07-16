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


from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def build_scheduler(
    scheduler_type: str,
    num_diffusion_timesteps: int,
    s: float = 0.008,
    sqrt: bool = False,
    nu: float = 1.0,
    clip: bool = True,
    cut: bool = True,
):
    """
    Factory function to build a scheduler based on the provided parameters.

    Args:
        scheduler_params (dict): Parameters for building the scheduler, must include "schedule_type".

    Returns:
        InterpolantDiffusionScheduler: An instance of a scheduler.

    Raises:
        NotImplementedError: If the scheduler type is not implemented.
    """
    if scheduler_type == "cosine_adaptive":
        return CosineSchedule(num_diffusion_timesteps, s, sqrt, nu, clip, cut)
    elif scheduler_type == "linear":
        return LinearSchedule(num_diffusion_timesteps)
    else:
        raise NotImplementedError(f"Scheduler '{scheduler_type}' is not implemented")


class InterpolantDiffusionSchedule(nn.Module, ABC):
    """
    Abstract base class for diffusion schedulers that compute alpha and beta values for diffusion processes.

    Methods:
        get_alphas_and_betas() -> Tuple[torch.Tensor, torch.Tensor]:
            Returns the alpha and beta values for the diffusion process.

        compute_alphas() -> torch.Tensor:
            Abstract method to compute the alpha values. Subclasses must implement this method.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('alphas', self.compute_alphas())
        log_alphas = torch.log(self.alphas)
        log_alphas_bar = torch.cumsum(log_alphas, dim=0)
        self.register_buffer('alphas_bar', torch.exp(log_alphas_bar))

    def get_alphas_and_betas(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the alpha and beta values for the diffusion process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Alpha and beta values.
        """
        return self.alphas, 1 - self.alphas

    def get_alphas_bar(self):
        return self.alphas_bar

    @abstractmethod
    def compute_alphas(self) -> torch.Tensor:
        """
        Abstract method to compute the alpha values. Subclasses must implement this method.

        Returns:
            torch.Tensor: Alpha values.
        """
        pass


class LinearSchedule(InterpolantDiffusionSchedule):
    """
    Linear scheduler for diffusion processes. This class computes the alpha values linearly.

    Args:
        num_diffusion_timesteps (int): Number of diffusion time steps.
    """

    def __init__(self, num_diffusion_timesteps: int, **kwargs):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        super().__init__()

    def compute_alphas(self) -> torch.Tensor:
        """
        Computes the alpha values linearly.

        Returns:
            torch.Tensor: Alpha values.
        """
        alphas = torch.linspace(1, 0, self.num_diffusion_timesteps + 1)[:-1]
        return alphas.clip(min=0.001, max=1.0)


class CosineSchedule(InterpolantDiffusionSchedule):
    """
    Cosine scheduler for diffusion processes. This class computes the alpha values using a cosine function.

    Args:
        num_diffusion_timesteps (int): Number of diffusion time steps.
        s (float): Smoothing parameter for the cosine function. Default is 0.008.
        sqrt (bool): Whether to take the square root of the alpha values. Default is False.
        nu (float): Exponent for the cosine function. Default is 1.0.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        num_diffusion_timesteps: int,
        s: float = 0.008,
        sqrt: bool = False,
        nu: float = 1.0,
        clip=True,
        cut=False,
        **kwargs,
    ):
        self.s = s
        self.nu = nu
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.sqrt = sqrt
        self.clip = clip
        self.cut = cut
        super().__init__()

    def clip_noise_schedule(self, alphas2, clip_value=0.01) -> torch.Tensor:
        """
        For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
        sampling.
        """
        alphas2 = torch.concat([torch.ones(1), alphas2], dim=0)
        alphas_step = alphas2[1:] / alphas2[:-1]
        alphas_step = alphas_step.clip(min=clip_value, max=1.0)
        alphas2 = torch.cumprod(alphas_step, dim=0)
        return alphas2

    def clip_noise_schedule_np(self, alphas2, clip_value=0.001):
        """
        For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
        sampling.
        """
        alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

        alphas_step = alphas2[1:] / alphas2[:-1]

        alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
        alphas2 = np.cumprod(alphas_step, axis=0)

        return alphas2

    # def compute_alphas(self) -> torch.Tensor:
    #     """
    #     Computes the alpha values using a cosine function.

    #     Returns:
    #         torch.Tensor: Alpha values.
    #     """
    #     steps = self.num_diffusion_timesteps + 2
    #     x = torch.linspace(0, self.num_diffusion_timesteps, steps)
    #     alphas_cumprod = (
    #         torch.cos(((x / self.num_diffusion_timesteps) ** self.nu + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2
    #     )
    #     if self.clip:
    #         alphas_cumprod = self.clip_noise_schedule(alphas_cumprod, clip_value=0.05)
    #     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    #     alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    #     if self.sqrt:
    #         alphas = torch.sqrt(alphas)
    #     alphas = alphas.clip(min=0.001) #, max=1.0)
    #     betas = 1.0 - alphas
    #     betas = torch.clip(betas, 0.0, 0.999).float()
    #     alphas = 1.0 - betas
    #     if self.cut:
    #         return alphas[1:]  # Removing the extra piece
    #     else:
    #         return alphas

    def compute_alphas(self) -> torch.Tensor:
        """
        Computes the alpha values using a cosine function.

        Returns:
            torch.Tensor: Alpha values.
        """
        steps = self.num_diffusion_timesteps + 2
        x = np.linspace(0, steps, steps)
        # x = np.expand_dims(x, 0)  # ((1, steps))
        # nu_arr = np.array(nu)  # (components, )  # X, charges, E, y, pos
        _steps = steps
        # _steps = num_diffusion_timesteps
        alphas_cumprod = (
            np.cos(0.5 * np.pi * (((x / _steps) ** self.nu) + self.s) / (1 + self.s)) ** 2
        )  # ((components, steps))
        # divide every element of alphas_cumprod by the first element of alphas_cumprod
        alphas_cumprod_new = alphas_cumprod / alphas_cumprod[0]
        ### new included
        alphas_cumprod_new = self.clip_noise_schedule_np(alphas_cumprod_new, clip_value=0.05)  # [None, ...]
        # remove the first element of alphas_cumprod and then multiply every element by the one before it
        alphas = alphas_cumprod_new[1:] / alphas_cumprod_new[:-1]
        alphas = alphas.clip(min=0.001)
        betas = 1 - alphas
        betas = torch.clip(torch.from_numpy(betas), 0.0, 0.999).squeeze().float()
        if self.cut:
            return 1.0 - betas[1:]
        else:
            return 1.0 - betas


def test_scheduler():
    """
    Test function to verify the scheduler implementation.

    Asserts that the length of the computed alpha values matches the expected number of diffusion time steps.
    Prints the computed alpha values.
    """
    scheduler_params = {
        "schedule_type": "cosine_adaptive",
        "num_diffusion_timesteps": 10,
        "nu": 1.5,
        "s": 0.008,
        "sqrt": False,
    }
    scheduler = build_scheduler(scheduler_params=scheduler_params)
    assert len(scheduler.alphas) == scheduler_params["num_diffusion_timesteps"]

    alphas, betas = scheduler.get_alphas_and_betas()
    alphas_bar = scheduler.get_alphas_bar()
    assert torch.abs(alphas[0] * alphas[1] - alphas_bar[1]) < 1e-7
    print("Cosine Scheduler Alphas:", alphas)

    scheduler_params["schedule_type"] = "linear"
    scheduler = build_scheduler(scheduler_params=scheduler_params)
    assert len(scheduler.alphas) == scheduler_params["num_diffusion_timesteps"]

    alphas, betas = scheduler.get_alphas_and_betas()
    print("Linear Scheduler Alphas:", alphas)


if __name__ == "__main__":
    test_scheduler()