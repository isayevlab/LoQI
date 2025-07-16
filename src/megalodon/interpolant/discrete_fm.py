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
from torch_geometric.utils import sort_edge_index

from megalodon.interpolant.interpolant import Interpolant
from megalodon.interpolant.interpolant_scheduler import build_scheduler


class DiscreteFlowMatchingInterpolant(Interpolant):
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
        prior_type: str = "uniform",
        vector_field_type: str = "standard",
        solver_type: str = "ode",
        timesteps: int = 500,
        min_t: float = 1e-2,
        time_type: str = 'continuous',
        num_classes: int = 10,
        custom_prior: torch.Tensor = None,
        scheduler_type='linear',
        s: float = 0.008,
        sqrt: bool = False,
        nu: float = 1.0,
        clip: bool = True,
        temp: float = 1.0,  # 0.1,
        stochasticity: float = 1.0,  # 10 #! Current run has too much noise lets tone it down
    ):
        super(DiscreteFlowMatchingInterpolant, self).__init__(prior_type, solver_type, timesteps, time_type)
        self.num_classes = num_classes
        self.vector_field_type = vector_field_type
        self.min_t = min_t
        self.max_t = 1 - min_t
        self.custom_prior = custom_prior
        self.stochasticity = stochasticity
        self.temp = temp
        self.init_schedulers(timesteps, scheduler_type, s, sqrt, nu, clip)

    def init_schedulers(self, timesteps, scheduler_type, s, sqrt, nu, clip):
        self.schedule_type = scheduler_type
        if scheduler_type == "linear":  #! vpe_linear is just linear with an update weight of recip_time_to_go
            self.discrete_time_only = False
            time = torch.linspace(self.min_t, 1, self.timesteps)
            self.register_buffer("time", time)
            self.register_buffer("forward_data_schedule", time)
            self.register_buffer("forward_noise_schedule", 1.0 - time)
        elif scheduler_type == "vpe":
            self.discrete_time_only = True
            self.scheduler = build_scheduler(scheduler_type, timesteps, s, sqrt, nu, clip)
            alphas, betas = self.scheduler.get_alphas_and_betas()
            # FlowMol defines alpha as 1 - cos ^2
            # self.forward_data_schedule = self.alphas
            # self.reverse_data_schedule = 1.0 - self.alphas
            # self.derivative_forward_data_schedule = self.alphas_prime
            # self.alpha_bar = self.alphas
            self.register_buffer('alphas', alphas)
            self.register_buffer('betas', betas)
            self.register_buffer('alpha_bar', alphas)
            self.register_buffer('forward_data_schedule', alphas)
            self.register_buffer('reverse_data_schedule', 1.0 - self.alphas)

    def loss_weight_t(self, time):
        #! No loss weightining is used for discrete data in MultiFlow or Semla
        weight = torch.ones_like(time).to(time.device)
        return weight
        # if self.time_type == "continuous":
        #     # return torch.clamp(time / (1 - time), min=0.05, max=1.5)
        #     # loss scale for "frameflow":
        #     return (1 - torch.clamp(t, 0, self.max_t_sample)) / (1 - torch.clamp(t, 0, self.clip_t))**2
        # else:
        #     if self.schedule_type == "linear":
        #         t = time / self.timesteps
        #         return (1 - torch.clamp(t, 0, self.max_t_sample)) / (1 - torch.clamp(t, 0, self.clip_t))**2
        #     else:
        #         return torch.clamp(self.snr(time), min=0.05, max=1.5)

    def update_weight(self, t):
        # if self.vector_field_type == "endpoint":
        weight = torch.ones_like(t).to(t.device)
        # elif self.vector_field_type == "standard":
        # weight = torch.clamp(1 / (1 - t), max=self.timesteps)  # at T = 1 this makes data_scale = 1
        return weight

    def forward_schedule(self, batch, time):
        if self.time_type == "continuous":
            if self.schedule_type == "linear":
                return time[batch].unsqueeze(1), (1.0 - time)[batch].unsqueeze(1)
            else:
                raise NotImplementedError("Continuoys time is only implemented with linear schedule")
        else:
            return (
                self.forward_data_schedule[time].unsqueeze(1)[batch],
                self.forward_noise_schedule[time].unsqueeze(1)[batch],
            )

    def interpolate(self, batch, x1, time):
        """
        Interpolate using discrete interpolation method.
        """
        # TODO: how to get the mask out when we only want discrete loss on masked states?
        if self.prior_type in ["mask", "absorb", "uniform", "custom"]:
            x0 = self.prior(None, x1.shape, x1.device)
            if self.time_type == "continuous":
                t = time
            else:
                t = time / self.timesteps
            t = t[batch]
            xt = x1.clone()
            corrupt_mask = torch.rand_like(t) < (1 - t)
            xt[corrupt_mask] = x0[corrupt_mask]
        else:
            raise ValueError("Only uniform and mask are supported")

        return x1, xt, x0

    def interpolate_edges(self, batch, x1, x1_index, time):
        """
        Interpolate using discrete interpolation method.
        Similar to sample_edges_categorical https://github.com/tuanle618/eqgat-diff/blob/68aea80691a8ba82e00816c82875347cbda2c2e5/eqgat_diff/experiments/diffusion/categorical.py#L242
        """
        j, i = x1_index
        mask = j < i
        mask_i = i[mask]
        edge_attr_triu = x1[mask]
        batch = batch[mask_i]
        _, edge_attr_t, upper_probs = self.interpolate(batch[mask_i], edge_attr_triu, time)
        edge_index_global_perturbed, edge_attr_global_perturbed = self.clean_edges(x1_index, edge_attr_t)
        return x1, edge_attr_global_perturbed, _

    def prior(self, batch, shape, device, one_hot=False):
        """
        Returns discrete index (num_samples, 1) or one hot if True (num_samples, num_classes)
        """
        num_samples = shape[0]
        if self.prior_type in ["absorb", "mask"]:
            x0 = torch.ones((num_samples,)).to(torch.int64) * (self.num_classes - 1)
        elif self.prior_type == "uniform":
            x0 = torch.randint(0, self.num_classes, (num_samples,)).to(torch.int64)
        elif self.prior_type in ["custom", "data"]:
            x0 = torch.multinomial(self.custom_prior, num_samples, replacement=True).to(torch.int64)
        else:
            raise ValueError("Only uniform and mask/absorb are supported")
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
        edge_attr_triu = self.prior(None, (num_samples, self.num_classes), device)
        edge_index_global, edge_attr_global, mask, mask_i = self.clean_edges(index, edge_attr_triu, return_masks=True)
        if one_hot:
            edge_attr_global = F.one_hot(edge_attr_global, num_classes=self.num_classes).float()
        if return_masks:
            return edge_attr_global.to(device), edge_index_global.to(device), mask, mask_i
        else:
            return edge_attr_global.to(device), edge_index_global.to(device)

    def step_uniform(
        self,
        batch,
        xt,  #! if takes in one hot it will convert it
        x_hat,  #! assumes input is logits
        time,
        dt,
    ):
        """
        Perform a euler step in the discrete interpolant method.
        """
        if len(xt.shape) > 1:
            xt = xt.argmax(dim=-1)

        S = self.num_classes
        if self.time_type == "continuous":
            t = time
        else:
            t = time / self.timesteps
        t = t[batch].unsqueeze(1)
        N = torch.zeros_like(t)
        N[t + dt < 1.0] = self.stochasticity

        t = torch.clamp(t, min=self.min_t, max=self.max_t)

        logits_1 = x_hat
        pt_x1_probs = F.softmax(logits_1 / self.temp, dim=-1)
        pt_x1_eq_xt_prob = torch.gather(pt_x1_probs, dim=-1, index=xt.long().unsqueeze(-1))

        first = dt * pt_x1_probs * ((1 + 2 * N * (S - 1) * t) / (1 - t))
        second = dt * N * pt_x1_eq_xt_prob
        step_probs = (first + second).clamp(max=1.0)

        # On-diagonal step probs
        step_probs.scatter_(-1, xt.long().unsqueeze(-1), 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, xt.long().unsqueeze(-1), diags)

        samples = torch.distributions.Categorical(step_probs).sample()
        return samples

    def step_absorb(
        self,
        batch,
        xt,  #! if takes in one hot it will convert it
        x_hat,  #! assumes input is logits
        time,
        dt,
        absorb_step="sample_first",  # "sample_first" "sample_last", "purity"
    ):
        """
        Perform a euler step in the discrete interpolant method.
        """
        if len(xt.shape) > 1:
            xt = xt.argmax(dim=-1)

        N = self.stochasticity
        S = self.num_classes
        MASK_TOKEN_INDEX = S - 1
        if self.time_type == "continuous":
            t = time
        else:
            t = time / self.timesteps
        og_time = t
        t = t[batch].unsqueeze(1)

        t = torch.clamp(t, min=self.min_t, max=self.max_t)
        xt_is_mask = (xt == MASK_TOKEN_INDEX).view(-1, 1).float()
        limit = dt * ((1 + (N * t)) / ((1 - t)))
        if absorb_step == "sample_first":
            # Matches Semla and MultiFlow Paper but not Code
            xt_is_mask = xt_is_mask.squeeze(1)
            pt_x1_probs = F.softmax(x_hat / self.temp, dim=-1)
            x_next = torch.distributions.Categorical(pt_x1_probs).sample()

            unmask = torch.rand_like(xt.float()) < limit.squeeze(1)
            unmask = unmask * xt_is_mask

            # Choose elements to mask
            mask = torch.rand_like(xt.float()) < (N * dt)
            mask = mask * (1 - xt_is_mask)
            mask[t.squeeze(1) + dt >= 1.0] = 0.0

            xt[unmask == 1] = x_next[unmask == 1]
            xt[mask == 1] = MASK_TOKEN_INDEX
            x_next = xt
        elif absorb_step == "sample_last":
            # matches MultiFlow Code but not Paper
            mask_one_hot = torch.zeros((S,), device=xt.device)
            mask_one_hot[MASK_TOKEN_INDEX] = 1.0
            logits_1 = x_hat
            logits_1[:, MASK_TOKEN_INDEX] = -1e9

            pt_x1_probs = F.softmax(logits_1 / self.temp, dim=-1)

            xt_is_mask = (xt == MASK_TOKEN_INDEX).view(-1, 1).float()
            step_probs = limit * pt_x1_probs  #!UNMASK
            step_probs += dt * N * (1 - xt_is_mask) * mask_one_hot.view(1, -1)  #!MASK UNMASKED STATES
            step_probs = step_probs.clamp(min=0, max=1.0)
            # On-diagonal step probs
            step_probs.scatter_(-1, xt.long().unsqueeze(-1), 0.0)
            diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
            step_probs.scatter_(-1, xt.long().unsqueeze(-1), diags)

            x_next = torch.multinomial(step_probs.view(-1, S), num_samples=1).view(xt.shape)
        elif absorb_step == "purity":
            logits_1 = x_hat
            aatypes_t = xt.unsqueeze(1)
            num_atoms, S = logits_1.shape
            noise = self.stochasticity
            assert aatypes_t.shape == (num_atoms, 1)
            assert S == self.num_classes
            device = logits_1.device
            MASK_TOKEN_INDEX = S - 1
            logits_1_wo_mask = logits_1[:, 0:-1]  # (N, S-1)
            pt_x1_probs = F.softmax(logits_1_wo_mask / self.temp, dim=-1)  # (N, S-1)
            #! Things need to be in batches to have things per molecule

            unique_values, counts = batch.unique(return_counts=True)
            batch_size = max(unique_values).item() + 1
            batch_pt_x1_probs = torch.zeros((batch_size, max(counts), S - 1)).to(x_hat.device)
            batch_aatypes_t = torch.zeros((batch_size, max(counts))).to(x_hat.device)
            for b in range(batch_size):
                batch_pt_x1_probs[b, : counts[b], :] = pt_x1_probs[batch == b]
                batch_aatypes_t[b, : counts[b]] = aatypes_t[batch == b].view(1, -1)
            # max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0]  # (N)
            max_logprob = torch.max(torch.log(batch_pt_x1_probs), dim=-1)[0]  # B x N

            # bias so that only currently masked positions get chosen to be unmasked
            # max_logprob = max_logprob - (aatypes_t.squeeze(1) != MASK_TOKEN_INDEX).float() * 1e9
            max_logprob = max_logprob - (batch_aatypes_t != MASK_TOKEN_INDEX).float() * 1e9
            sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True)  # (N)

            # unmask_probs = (dt * ((1 + noise * t) / (1 - t)).to(device)).clamp(max=1)  # (N, 1)
            unmask_probs = (dt * ((1 + noise * og_time) / (1 - og_time)).to(device)).clamp(max=1)
            # number_to_unmask = torch.binomial(count=torch.count_nonzero(aatypes_t.squeeze(1) == MASK_TOKEN_INDEX, dim=-1).float(), prob=unmask_probs.squeeze(1)) #(N)
            number_to_unmask = torch.binomial(
                count=torch.count_nonzero(batch_aatypes_t == MASK_TOKEN_INDEX, dim=-1).float(), prob=unmask_probs
            )
            unmasked_samples = torch.multinomial(pt_x1_probs.view(-1, S - 1), num_samples=1).view(num_atoms, 1)
            batch_unmasked_samples = torch.zeros((batch_size, max(counts))).to(x_hat.device)
            for b in range(batch_size):
                batch_unmasked_samples[b, : counts[b]] = unmasked_samples[batch == b].view(1, -1)
            num_res = max(counts)

            D_grid = torch.arange(num_res, device=device).view(1, -1).repeat(batch_size, 1)  # batch x num_atoms
            mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
            inital_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, num_res)
            masked_sorted_max_logprobs_idcs = (
                mask1 * sorted_max_logprobs_idcs + (1 - mask1) * inital_val_max_logprob_idcs
            ).long()
            mask2 = torch.zeros((batch_size, num_res), device=device)
            mask2.scatter_(
                dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((batch_size, num_res), device=device)
            )
            unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, num_res).float()
            mask2 = mask2 * (1 - unmask_zero_row)
            # aatypes_t = aatypes_t * (1 - mask2) + unmasked_samples * mask2
            batch_aatypes_t = batch_aatypes_t * (1 - mask2) + batch_unmasked_samples * mask2
            # re-mask
            u = torch.rand(batch_size, num_res, device=x_hat.device)
            re_mask_mask = (u < dt * noise).float()
            # aatypes_t = aatypes_t * (1 - re_mask_mask) + MASK_TOKEN_INDEX * re_mask_mask
            batch_aatypes_t = batch_aatypes_t * (1 - re_mask_mask) + MASK_TOKEN_INDEX * re_mask_mask
            x_next = torch.zeros(num_atoms).to(x_hat.device)
            counter = 0
            for b in range(batch_size):
                x_next[counter : counter + counts[b]] = batch_aatypes_t[b, : counts[b]]
                counter += counts[b]

        return x_next

    def step(
        self,
        batch,
        xt,  #! if takes in one hot it will convert it
        x_hat,  #! assumes input is logits
        time,
        dt,
        x0=None,
    ):
        """
        Perform a euler step in the discrete interpolant method.
        """
        if self.prior_type in ["uniform", "data", "custom"]:
            x_next = self.step_uniform(batch, xt, x_hat.clone(), time, dt)

        elif self.prior_type in ["mask", "absorb"]:
            #! Masking is initalized with one more column as the mask state
            x_next = self.step_absorb(batch, xt, x_hat.clone(), time, dt)

        return x_next

    def step_edges(self, batch, edge_index, edge_attr_t, edge_attr_hat, time, dt):
        """
        Given N*N input predictions only work on lower triangluar portion.
        """
        j, i = edge_index
        mask = j < i
        mask_i = i[mask]
        edge_attr_t = edge_attr_t[mask]
        edge_attr_hat = edge_attr_hat[mask]
        edge_attr_next = self.step(batch[mask_i], edge_attr_t, edge_attr_hat, time, dt)
        return self.clean_edges(edge_index, edge_attr_next)

    def clean_edges(self, edge_index, edge_attr_next, one_hot=False, return_masks=False):
        """
        Takes lower triangular edge tensor and creates fully symmetric square.
        Example: If i --> j now j --> i
        Note Semla predicts full N^2 unike EQGAT
        """
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


if __name__ == "__main__":
    ligand_pos = torch.rand((75, 3))
    batch_ligand = torch.Tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
        ]
    ).to(torch.int64)
    ligand_feats = torch.Tensor(
        [
            2,
            4,
            2,
            4,
            2,
            4,
            4,
            3,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            3,
            1,
            1,
            1,
            2,
            4,
            2,
            4,
            2,
            4,
            4,
            3,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            3,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            12,
            2,
            5,
            2,
            3,
            5,
            1,
            5,
            2,
            4,
            2,
            4,
            2,
            4,
            4,
            3,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            5,
            1,
            3,
            1,
            1,
            12,
        ]
    ).to(torch.int64)
    num_classes = 13
    # Initialize the adjacency matrix with zeros
    adj_matrix = torch.zeros((75, 75, 5), dtype=torch.int64)
    no_bond = torch.zeros(5)
    no_bond[0] = 1
    # Using broadcasting to create the adjacency matrix
    adj_matrix[batch_ligand.unsqueeze(1) == batch_ligand] = 1
    for idx, i in enumerate(batch_ligand):
        for jdx, j in enumerate(batch_ligand):
            if idx == jdx:
                adj_matrix[idx][jdx] = no_bond
            elif i == j:
                adj_matrix[idx][jdx] = torch.nn.functional.one_hot(torch.randint(0, 5, (1,)), 5).squeeze(0)

    atom_embedder = torch.nn.Linear(num_classes, 64)
    X = ligand_pos
    H = atom_embedder(F.one_hot(ligand_feats, num_classes).float())
    A = adj_matrix
    mask = batch_ligand.unsqueeze(1) == batch_ligand.unsqueeze(0)  # Shape: (75, 75)
    E_idx = mask.nonzero(as_tuple=False).t()
    self_loops = E_idx[0] != E_idx[1]
    E_idx = E_idx[:, self_loops]

    dfm = DiscreteFlowMatchingInterpolant(num_classes=num_classes)
    xt = ligand_feats
    x_hat = torch.rand((75, num_classes))
    time = torch.tensor([0.2, 0.4, 0.6, 0.8])
    res = dfm.step(batch_ligand, xt, x_hat, time, dt=1 / 500)
    dfm = DiscreteFlowMatchingInterpolant(num_classes=num_classes, prior_type="absorb")
    xt = ligand_feats
    x_hat = torch.rand((75, num_classes))
    time = torch.tensor([0.2, 0.4, 0.6, 0.8])
    res = dfm.step(batch_ligand, xt, x_hat, time, dt=1 / 500)
    print("Success")