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
from torch_scatter import scatter_mean

from megalodon.interpolant.interpolant import Interpolant
from megalodon.interpolant.interpolant_scheduler import build_scheduler


class ContinuousDiffusionInterpolant(Interpolant):
    """
    Class for continuous interpolation.

    Attributes:
        prior_type (str): Type of prior.
        vector_field_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        prior_type: str = 'gaussian',
        diffusion_type: str = 'vdm',
        solver_type: str = "sde",
        timesteps: int = 500,
        time_type: str = 'discrete',
        num_classes: int = 3,
        scheduler_type='cosine_adaptive',
        s: float = 0.008,
        sqrt: bool = False,
        nu: float = 1.0,
        clip: bool = True,
        com_free: bool = True,
        cut: bool = False,
    ):
        super(ContinuousDiffusionInterpolant, self).__init__(prior_type, solver_type, timesteps, time_type)
        self.num_classes = num_classes
        self.diffusion_type = diffusion_type
        self.com_free = com_free
        self.init_schedulers(timesteps, scheduler_type, s, sqrt, nu, clip, cut)

    def init_schedulers(self, timesteps, scheduler_type, s, sqrt, nu, clip, cut):
        self.scheduler = build_scheduler(scheduler_type, timesteps, s, sqrt, nu, clip, cut)
        alphas, betas = self.scheduler.get_alphas_and_betas()
        if self.diffusion_type == "ddpm":
            log_alpha = torch.log(alphas)
            log_alpha_bar = torch.cumsum(log_alpha, dim=0)
            alpha_bar = alphas_cumprod = torch.exp(log_alpha_bar)
            alpha_bar_prev = alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            posterior_mean_c0_coef = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            posterior_mean_ct_coef = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
            posterior_logvar = torch.log(
                torch.nn.functional.pad(posterior_variance[:-1], (1, 0), value=posterior_variance[0])
            )

            self.register_buffer('alphas', alphas)
            self.register_buffer('betas', betas)
            self.register_buffer('alpha_bar', alpha_bar)
            self.register_buffer('alpha_bar_prev', alpha_bar_prev)
            self.register_buffer('posterior_variance', posterior_variance)
            self.register_buffer('posterior_mean_c0_coef', posterior_mean_c0_coef)
            self.register_buffer('posterior_mean_ct_coef', posterior_mean_ct_coef)
            self.register_buffer('posterior_logvar', posterior_logvar)

            self.register_buffer('forward_data_schedule', torch.sqrt(alpha_bar))
            self.register_buffer('forward_noise_schedule', torch.sqrt(1 - alpha_bar))
            self.register_buffer('reverse_data_schedule', posterior_mean_c0_coef)
            self.register_buffer('reverse_noise_schedule', posterior_mean_ct_coef)
            self.register_buffer('log_var', posterior_logvar)
            # import pickle
            # temp = {
            #     'forward_data_schedule':  torch.sqrt(alpha_bar),
            #     'forward_noise_schedule': torch.sqrt(1 - alpha_bar),
            #     'reverse_data_schedule': posterior_mean_c0_coef,
            #     'reverse_noise_schedule': posterior_mean_ct_coef,
            #     'log_var': posterior_logvar
            # }
            # with open("/workspace/bionemo/megalodon/models/ddpm.pickle", "wb") as f:
            #     pickle.dump(temp, f)

        elif self.diffusion_type == "vdm":
            assert not cut  # alphas.shape = T + 1
            log_alpha = torch.log(alphas)
            log_alpha_bar = torch.cumsum(log_alpha, dim=0)
            alphas_cumprod = torch.exp(log_alpha_bar)
            self.register_buffer('alphas', alphas[1:])
            self.register_buffer('betas', betas[1:])
            # self._log_alpha_bar = log_alpha_bar
            alpha_bar = torch.exp(log_alpha_bar)
            self.register_buffer('alpha_bar', torch.exp(log_alpha_bar)[1:])
            sigma2_bar = -torch.expm1(2 * log_alpha_bar)
            sigma_bar = torch.sqrt(sigma2_bar)
            self.register_buffer('sigma_bar', sigma_bar[1:])

            self.register_buffer('forward_data_schedule', alpha_bar[1:])
            self.register_buffer('forward_noise_schedule', sigma_bar[1:])
            # Diffusion s < t to t is closer to noise T = 0 data
            s_time = list(range(self.timesteps))  # [i for i in range(self.timesteps - 1, -1, -1)]
            t_time = list(range(1, 1 + self.timesteps))

            # sigma_sq_ratio = self.get_sigma_pos_sq_ratio(s_int=s, t_int=t)
            s2_s = -torch.expm1(2 * log_alpha_bar[s_time])
            s2_t = -torch.expm1(2 * log_alpha_bar[t_time])
            sigma_sq_ratio = torch.exp(torch.log(s2_s) - torch.log(s2_t)).float()
            self.register_buffer('sigma_sq_ratio', sigma_sq_ratio)

            prefactor1 = sigma2_bar[t_time]
            alpha_pos_ts_sq = torch.exp(2 * log_alpha_bar[t_time] - 2 * log_alpha_bar[s_time])
            prefactor2 = sigma2_bar[s_time] * alpha_pos_ts_sq
            sigma2_t_s = prefactor1 - prefactor2
            noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
            noise_prefactor = torch.sqrt(noise_prefactor_sq)  # .unsqueeze(-1)

            # z_t_prefactor = alpha_pos_ts_sq * sigma_sq_ratio  # .unsqueeze(-1)
            z_t_prefactor = torch.exp(log_alpha_bar[t_time] - log_alpha_bar[s_time]).float() * sigma_sq_ratio
            a_s = alpha_bar[s_time]  # self.get_alpha_bar(t_int=s_int)
            alpha_ratio_sq = alpha_pos_ts_sq  # self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int)
            sigma_ratio_sq = sigma_sq_ratio  # self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
            x_prefactor = (a_s * (1 - alpha_ratio_sq * sigma_ratio_sq)).float()

            self.register_buffer('reverse_data_schedule', x_prefactor)
            self.register_buffer('reverse_noise_schedule', z_t_prefactor)
            # self.register_buffer('std', noise_prefactor)
            self.register_buffer('log_var', 2 * torch.log(noise_prefactor))  # we do this since its exp(0.5*log_var)
            # import pickle
            # temp = {
            #     'forward_data_schedule': alpha_bar[1:],
            #     'forward_noise_schedule': sigma_bar[1:],
            #     'reverse_data_schedule': x_prefactor,
            #     'reverse_noise_schedule': z_t_prefactor,
            #     'log_var': noise_prefactor
            # }
            # with open("/workspace/bionemo/megalodon/models/vdm.pickle", "wb") as f:
            #     pickle.dump(temp, f)

    def forward_schedule(self, batch, time):
        t_idx = self.timesteps - 1 - time
        return (
            self.forward_data_schedule[t_idx].unsqueeze(1)[batch],
            self.forward_noise_schedule[t_idx].unsqueeze(1)[batch],
        )

    def reverse_schedule(self, batch, time):
        t_idx = self.timesteps - 1 - time
        return (
            self.reverse_data_schedule[t_idx].unsqueeze(1)[batch],
            self.reverse_noise_schedule[t_idx].unsqueeze(1)[batch],
            self.log_var[t_idx].unsqueeze(1)[batch],
        )

    def interpolate(self, batch, x1, time):
        """
        Interpolate using continuous diffusion.
        """
        x0 = self.prior(batch, x1.shape, x1.device)
        data_scale, noise_scale = self.forward_schedule(batch, time)
        return x1, data_scale * x1 + noise_scale * x0, x0

    def prior(self, batch, shape, device, x1=None):
        if self.prior_type == "gaussian" or self.prior_type == "normal":
            x0 = torch.randn(shape).to(device)
            if self.com_free:
                x0 = x0 - scatter_mean(x0, batch, dim=0)[batch]
        else:
            raise ValueError("Only Gaussian is supported")
        return x0.to(device)

    def step(self, batch, xt, x_hat, x0, time, dt=None):
        """
        Perform a euler step.
        """
        if self.solver_type == "sde":
            data_scale, noise_scale, log_var = self.reverse_schedule(batch, time)
            mean = data_scale * x_hat + noise_scale * xt
            if self.diffusion_type == "ddpm":
                # no noise when diffusion t == 0 so flow matching t == 1
                nonzero_mask = (1 - (time == (self.timesteps - 1)).float())[batch].unsqueeze(-1)
                x_next = mean + nonzero_mask * (0.5 * log_var).exp() * self.prior(batch, xt.shape, device=xt.device)
            elif self.diffusion_type == "vdm":
                x_next = mean + (0.5 * log_var).exp() * self.prior(batch, xt.shape, device=xt.device)
            # TODO: can add guidance module here
        else:
            raise ValueError("Only SDE Implemented")

        if self.com_free:
            x_next = x_next - scatter_mean(x_next, batch, dim=0)[batch]
        return x_next

    def snr(self, time):
        t_idx = self.timesteps - 1 - time
        abar = self.alpha_bar[t_idx]
        return abar / (1 - abar)