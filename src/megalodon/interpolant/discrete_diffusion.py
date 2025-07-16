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
import torch.nn.functional as F
from torch_geometric.utils import sort_edge_index

from megalodon.interpolant.interpolant import Interpolant
from megalodon.interpolant.interpolant_scheduler import build_scheduler
from megalodon.interpolant.interpolant_utils import (
    index_to_log_onehot,
    log_1_min_a,
    log_add_exp,
    log_sample_categorical,
)


class DiscreteDiffusionInterpolant(Interpolant):
    """
    Class for continuous interpolation.
    Note the udnerlying D3PM only works for discrete time.
    Can look into MultiFlow Precurssor that uses CTMC for continuous time discrete diffusion (https://arxiv.org/pdf/2205.14987, https://github.com/andrew-cr/tauLDR/blob/main/lib/models/models.py)
    Argmax Flow also operate over discrete time
    Continuous time can work for cintuous gaussian representations found in DiffSBDD but we are not doing this.

    Attributes:
        prior_type (str): Type of prior.
        vector_field_type (str): Type of interpolant update weight.
        solver_type (str): ODE or SDE
        timesteps (int): Number of interpolant steps
    """

    def __init__(
        self,
        prior_type: str = "uniform",
        diffusion_type: str = 'd3pm',
        solver_type: str = "sde",
        timesteps: int = 500,
        time_type: str = 'discrete',
        num_classes: int = 12,
        custom_prior: torch.Tensor = None,
        scheduler_type='cosine_adaptive',
        s: float = 0.008,
        sqrt: bool = False,
        nu: float = 1.0,
        clip: bool = True,
        cut: bool = True,
    ):
        super(DiscreteDiffusionInterpolant, self).__init__(prior_type, solver_type, timesteps, time_type)
        self.num_classes = num_classes
        self.custom_prior = custom_prior
        self.diffusion_type = diffusion_type
        self.init_schedulers(timesteps, scheduler_type, s, sqrt, nu, clip, cut)

    def get_Qt(self, terminal_distribution: torch.Tensor):
        #! If terminal distriubtion is [0 ... 0, 1] then we get masking state
        #! If terminal is [1/k ... 1/k] we get uniform
        # See Appendix A.2 D3PM https://arxiv.org/pdf/2107.03006
        QT = []
        for alpha_t in self.alphas:
            stay_prob = torch.eye(len(terminal_distribution)) * alpha_t
            diffuse_prob = (1.0 - alpha_t) * (
                torch.ones(1, len(terminal_distribution)) * (terminal_distribution.unsqueeze(0))
            )
            QT.append(stay_prob + diffuse_prob)
        return torch.stack(QT, dim=0)

    def d3pm_setup(self):
        self.discrete_time_only = True
        if self.prior_type == "uniform":
            prior_dist = torch.ones((self.num_classes)) * 1 / self.num_classes
        elif self.prior_type in ["absorb", "mask"]:
            prior_dist = torch.zeros((self.num_classes))
            prior_dist[-1] = 1.0
        elif self.prior_type in ["custom", "data"]:
            prior_dist = self.custom_prior
        assert torch.sum(prior_dist).item() - 1.0 < 1e-5
        Qt = self.get_Qt(prior_dist)
        Qt_prev = torch.eye(self.num_classes)
        Qt_bar = []
        for i in range(len(self.alphas)):
            Qtb = Qt_prev @ Qt[i]
            Qt_bar.append(Qtb)
            Qt_prev = Qtb
        Qt_bar = torch.stack(Qt_bar)
        Qt_bar_prev = Qt_bar[:-1]
        Qt_prev_pad = torch.eye(self.num_classes)
        Qt_bar_prev = torch.concat([Qt_prev_pad.unsqueeze(0), Qt_bar_prev], dim=0)

        return Qt, Qt_bar, Qt_bar_prev

    def init_schedulers(self, timesteps, scheduler_type, s, sqrt, nu, clip, cut):
        self.schedule_type = scheduler_type
        self.scheduler = build_scheduler(scheduler_type, timesteps, s, sqrt, nu, clip, cut)
        alphas, betas = self.scheduler.get_alphas_and_betas()
        log_alpha = torch.log(alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        alpha_bar = torch.exp(log_alpha_bar)
        # alpha_bar_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        if self.diffusion_type == "d3pm":  # MIDI and EQ
            Qt, Qt_bar, Qt_prev_bar = self.d3pm_setup()
            self.register_buffer('alphas', alphas[-self.timesteps :])
            self.register_buffer('alpha_bar', alpha_bar[-self.timesteps :])
            self.register_buffer('forward_data_schedule', Qt_bar[-self.timesteps :])
            self.register_buffer('forward_noise_schedule', Qt_bar[-self.timesteps :])
            self.register_buffer('Qt', Qt[-self.timesteps :])
            self.register_buffer('Qt_bar', Qt_bar[-self.timesteps :])
            self.register_buffer('Qt_prev_bar', Qt_prev_bar[-self.timesteps :])
        elif self.diffusion_type == "argmax":  # TargetDiff
            self.register_buffer('forward_data_schedule', log_alpha_bar)
            self.register_buffer('forward_noise_schedule', log_1_min_a(log_alpha_bar))

    def forward_schedule(self, batch, time):
        t_idx = self.timesteps - 1 - time
        return self.forward_data_schedule[t_idx][batch], self.forward_noise_schedule[t_idx][batch]

    def reverse_schedule(self, time):
        t_idx = self.timesteps - 1 - time
        return (
            self.reverse_data_schedule[t_idx].unsqueeze(1),
            self.reverse_noise_schedule[t_idx].unsqueeze(1),
            self.log_var[t_idx].unsqueeze(1),
        )

    def interpolate(self, batch, x1, time):
        """
        Interpolate using discrete interpolation method.
        """
        if len(x1.shape) == 1:
            x1_hot = F.one_hot(x1, self.num_classes)
        else:
            x1_hot = x1
        if self.diffusion_type == "d3pm":
            assert self.discrete_time_only
            ford = self.forward_schedule(batch, time)[0]
            probs = torch.einsum(
                "nj, nji -> ni", [x1_hot.float(), ford]
            )  #! Eqn 3 of D3PM https://arxiv.org/pdf/2107.03006
            assert torch.all((probs.sum(-1) - 1.0).abs() < 1e-4)
            xt = probs.multinomial(
                1,
            ).squeeze()
            return x1, xt, probs

        elif self.diffusion_type == "argmax":
            log_x0 = index_to_log_onehot(x1)
            data_scale, noise_scale = self.forward_schedule(time)
            log_probs = log_add_exp(log_x0 + data_scale, noise_scale - np.log(self.num_classes))
            xt = log_sample_categorical(log_probs)

        return x1, xt, None

    def interpolate_edges(self, batch, x1, x1_index, time):
        """
        Interpolate using discrete interpolation method.
        Similar to sample_edges_categorical https://github.com/tuanle618/eqgat-diff/blob/68aea80691a8ba82e00816c82875347cbda2c2e5/eqgat_diff/experiments/diffusion/categorical.py#L242
        """
        j, i = x1_index
        mask = j < i
        mask_i = i[mask]
        edge_attr_triu = x1[mask]
        _, edge_attr_t, upper_probs = self.interpolate(batch[mask_i], edge_attr_triu, time)
        edge_index_global_perturbed, edge_attr_global_perturbed = self.clean_edges(x1_index, edge_attr_t)
        return x1, edge_attr_global_perturbed, _

    def prior(self, batch, shape, device, one_hot=False):
        """
        Returns discrete index (num_samples,) or one hot if True (num_samples, num_classes)
        """
        num_samples = shape[0]
        if self.prior_type in ["absorb", "mask"]:
            x0 = torch.ones((num_samples,)).to(torch.int64) * (self.num_classes - 1)
        elif self.prior_type == "uniform":
            x0 = torch.randint(0, self.num_classes, (num_samples,)).to(torch.int64)
        elif self.prior_type in ["custom", "data"]:
            x0 = torch.multinomial(self.custom_prior, num_samples, replacement=True).to(torch.int64)
        else:
            raise ValueError("Only uniform and mask are supported")
        if one_hot:
            x0 = F.one_hot(x0, num_classes=self.num_classes)
        return x0.to(device)

    def prior_edges(self, batch, shape, index, device, one_hot=False, return_masks=False):
        """
        Returns discrete index (num_samples,) or one hot if True (num_samples, num_classes)
        similar to initialize_edge_attrs_reverse https://github.com/tuanle618/eqgat-diff/blob/68aea80691a8ba82e00816c82875347cbda2c2e5/eqgat_diff/experiments/diffusion/utils.py#L18
        """
        num_samples = shape[0]
        j, i = index
        mask = j < i
        mask_i = i[mask]
        num_upper_E = len(mask_i)
        num_samples = num_upper_E
        edge_attr_triu = self.prior(batch, (num_samples, self.num_classes), device)
        edge_index_global, edge_attr_global, mask, mask_i = self.clean_edges(index, edge_attr_triu, return_masks=True)
        if one_hot:
            edge_attr_global = F.one_hot(edge_attr_global, num_classes=self.num_classes).float()
        if return_masks:
            return edge_attr_global.to(device), edge_index_global.to(device), mask, mask_i
        else:
            return edge_attr_global.to(device), edge_index_global.to(device)

    def step(self, batch, xt, x_hat, time, x0=None, dt=None):
        """
        Perform a euler step in the discrete interpolant method.
        xt is the discrete variable at time t (or one hot)
        x_hat is the softmax of the logits
        """
        if self.solver_type == "sde" and self.diffusion_type == "d3pm":
            assert self.discrete_time_only
            if len(xt.shape) <= 1:
                xt = F.one_hot(xt, num_classes=self.num_classes).float()
            t_idx = self.timesteps - 1 - time[batch]

            A = torch.einsum("nj, nji -> ni", [xt, self.Qt[t_idx].permute(0, 2, 1)]).unsqueeze(1)  # [A, 1, 16]
            B = self.Qt_prev_bar[t_idx]  # [A, 16, 16]
            p0 = A * B
            p1 = torch.einsum("nij, nj -> ni", [self.Qt_bar[t_idx], xt]).unsqueeze(-1)
            probs = p0 / (p1.clamp(min=1e-5))
            unweighted_probs = (probs * x_hat.unsqueeze(-1)).sum(1)
            unweighted_probs[unweighted_probs.sum(dim=-1) == 0] = 1e-5
            # (N, a_t-1)
            probs = unweighted_probs / (unweighted_probs.sum(-1, keepdims=True) + 1.0e-5)
            x_next = probs.multinomial(
                1,
            ).squeeze()
        else:
            raise ValueError("Only SDE Implemented for D3PM")

        return x_next

    def step_edges(
        self, batch, edge_index, edge_attr_t, edge_attr_hat, time, mask=None, mask_i=None, return_masks=False, dt=None
    ):
        if mask is None or mask_i is None:
            j, i = edge_index
            mask = j < i
            mask_i = i[mask]
        edge_attr_t = edge_attr_t[mask]
        edge_attr_hat = edge_attr_hat[mask]
        edge_attr_next = self.step(batch[mask_i], edge_attr_t, edge_attr_hat, time)
        return self.clean_edges(edge_index, edge_attr_next, return_masks=return_masks)

    def clean_edges(self, edge_index, edge_attr_next, one_hot=False, return_masks=False):
        j, i = edge_index
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global = torch.stack([j, i], dim=0)
        edges_triu = F.one_hot(edge_attr_next, self.num_classes).float()
        edge_attr_global = torch.concat([edges_triu, edges_triu], dim=0)
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            sort_by_row=False,
        )
        if not one_hot:
            edge_attr_global = edge_attr_global.argmax(1)
        if return_masks:
            return edge_index_global, edge_attr_global, mask, mask_i
        else:
            return edge_index_global, edge_attr_global

    def snr(self, time):
        t_idx = self.timesteps - 1 - time
        abar = self.alpha_bar[t_idx]
        return abar / (1 - abar)