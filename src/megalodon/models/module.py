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



import pickle

import numpy as np
import torch
import torch.nn.functional as F
from lightning import pytorch as pl
from omegaconf import DictConfig, OmegaConf
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from tqdm import tqdm

from megalodon.dynamics.utils import InterpolantLossFunction
from megalodon.interpolant.builder import build_interpolant
from megalodon.models.denoising_models import ModelBuilder
from megalodon.models.self_conditioning import SelfConditioningBuilder
from megalodon.models.utils_train import ModelWithEMA


class Graph3DInterpolantModel(pl.LightningModule):
    def __init__(
            self,
            loss_params: DictConfig,
            optimizer_params: DictConfig,
            lr_scheduler_params: DictConfig,
            dynamics_params: DictConfig,
            interpolant_params: DictConfig,
            sampling_params: DictConfig,
            self_cond_params: DictConfig = None,
            ema=True,
            loss_fn=None,
            batch_preprocessor=None,
    ):
        super(Graph3DInterpolantModel, self).__init__()
        self.save_hyperparameters()
        self.optimizer_params = optimizer_params
        self.lr_scheduler_params = lr_scheduler_params
        self.dynamics_params = dynamics_params
        self.interpolant_params = interpolant_params
        self.global_variable = interpolant_params.global_variable_name
        self.loss_params = loss_params
        self.loss_functions = self.initialize_loss_functions()
        self.interpolants = self.initialize_interpolants()
        self.sampling_params = sampling_params
        self.node_distribution = self.initialize_inference()
        self.dynamics = ModelBuilder().create_model(
            dynamics_params.model_name, dynamics_params.model_args, dynamics_params.wrapper_args
        )
        self.dynamics = ModelWithEMA(self.dynamics, ema)

        self.self_conditioning_module = None
        if self_cond_params is not None:
            self.self_conditioning_module = self.configure_self_cond(self_cond_params)
            self.self_conditioning_module = ModelWithEMA(self.self_conditioning_module, ema)

        self.loss_fn = None
        if loss_fn is not None:
            self.loss_fn = loss_fn

        self.batch_preprocessor = batch_preprocessor

    def configure_self_cond(self, self_cond_params):
        self_cond_params = OmegaConf.to_container(self_cond_params, resolve=True)
        for var in self_cond_params["variables"]:
            var["inp_dim"] = self.interpolants[var["variable_name"]].num_classes
        return SelfConditioningBuilder().create_self_cond(self_cond_params)

    # def setup(self, stage = None):
    #     self.loss_functions = self.initialize_loss_functions()

    def initialize_loss_functions(self):
        loss_functions = {}
        # Make loss clamps configurable instead of hardcoded
        self.use_loss_clamps = getattr(self.loss_params, 'use_loss_clamps', False)
        self.loss_clamp_epoch_threshold = getattr(self.loss_params, 'loss_clamp_epoch_threshold', 10)
        
        if self.use_loss_clamps:
            self.loss_clamps = self.loss_params['loss_clamps']
        else:
            self.loss_clamps = {}
            
        for loss_params in self.loss_params.variables:
            index = loss_params.variable_name
            if "use_distance" in loss_params:
                loss_functions[index] = InterpolantLossFunction(
                    loss_scale=loss_params.loss_scale,
                    aggregation=loss_params.aggregate,
                    continuous=loss_params.continuous,
                    use_distance=loss_params.use_distance,
                    distance_scale=loss_params.distance_scale,  # TODO make these optional
                )
            else:
                loss_functions[index] = InterpolantLossFunction(
                    loss_scale=loss_params.loss_scale,
                    aggregation=loss_params.aggregate,
                    continuous=loss_params.continuous,
                )
        return loss_functions

    def load_prior(self, fpath):
        if fpath[-3:] == "npy":
            array = np.load(fpath)
            tensor = torch.tensor(array)  # .to(self.device)
        else:
            raise ValueError("Currently only supports numpy prior arrays")
        return tensor

    def initialize_interpolants(self):
        interpolants = torch.nn.ModuleDict()
        for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
            index = interp_param.variable_name
            if not interp_param.interpolant_type:
                interpolants[index] = None
                continue
            if interp_param.prior_type in ["mask", "absorb"]:
                interp_param.num_classes += 1
            elif interp_param.prior_type in ["custom", "data"]:
                interp_param = dict(interp_param)
                interp_param["custom_prior"] = self.load_prior(interp_param["custom_prior"]).float()
                interp_param["custom_prior"] = interp_param["custom_prior"][:interp_param["num_classes"]]
            interpolants[index] = build_interpolant(**interp_param)
        self.interpolant_param_variables = {
            interp_param.variable_name: interp_param for interp_param in
            self.interpolant_params.variables
        }
        return interpolants

    def configure_optimizers(self):
        if self.optimizer_params.type == "adamw":
            optimizer = torch.optim.AdamW(
                self.dynamics.parameters(),
                lr=self.optimizer_params.lr,
                amsgrad=self.optimizer_params.amsgrad,
                weight_decay=self.optimizer_params.weight_decay,
            )
        else:
            raise NotImplementedError('Optimizer not supported: %s' % self.optimizer_params.type)

        if self.lr_scheduler_params:
            if self.lr_scheduler_params.type == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=self.lr_scheduler_params.factor,
                    patience=self.lr_scheduler_params.patience,
                    min_lr=self.lr_scheduler_params.min_lr,
                    cooldown=self.lr_scheduler_params.cooldown,
                )
            elif self.lr_scheduler_params.type == "linear_warmup":
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.lr_scheduler_params.initial_lr
                                 / self.lr_scheduler_params.final_lr,
                    # Start factor (initial learning rate / final learning rate)
                    end_factor=1.0,  # End factor (final learning rate / final learning rate)
                    total_iters=self.lr_scheduler_params.num_warmup_steps,
                    # Number of iterations to go from start_factor to end_factor
                )
            elif self.lr_scheduler_params.type == "linear_warmup_decay":
                # Warm-up phase using LinearLR
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.lr_scheduler_params.initial_lr / self.lr_scheduler_params.final_lr,
                    end_factor=1.0,
                    total_iters=self.lr_scheduler_params.num_warmup_steps,  # Steps
                )

                # Decay phase using LinearLR (kicks in after milestone)
                decay_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=self.lr_scheduler_params.min_lr_decay / self.lr_scheduler_params.final_lr,
                    total_iters=self.lr_scheduler_params.num_decay_steps,  # Steps
                )

                # SequentialLR to combine both schedulers
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, decay_scheduler],
                    milestones=[self.lr_scheduler_params.milestone_steps],  # Milestone in steps
                )
            else:
                raise NotImplementedError(
                    'LR Scheduler not supported: %s' % self.lr_scheduler_params.type)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.lr_scheduler_params.interval,
                    # "monitor": self.lr_scheduler_params.monitor,
                    "frequency": self.lr_scheduler_params.frequency,
                    "strict": False,
                },
            }
        else:
            return {
                "optimizer": optimizer,
            }

    def sample_time(self, batch):
        batch_size = int(batch.batch.max()) + 1
        time = self.interpolants[self.global_variable].sample_time(
            num_samples=batch_size,
            device=batch[self.global_variable].device,
            method=self.interpolant_params.sample_time_method,
            mean=self.interpolant_params.sample_time_mean,
            scale=self.interpolant_params.sample_time_scale,
            min_t=self.interpolant_params.min_t,
        )
        return time

    def pre_format_molecules(self, batch, batch_size):
        # for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
        for index, interp_param in self.interpolant_param_variables.items():
            if "discrete" in interp_param.interpolant_type and interp_param.prior_type in ["mask", "absorb"]:
                    batch[index] = self.add_adsorbtion_state(batch[index])

            if index == "charges":
                batch["charges"] = batch["charges"] + interp_param.offset
        return batch

    def add_adsorbtion_state(self, h):
        # h is (N, C) and we want to add a column of all zeros at the end
        N, C = h.shape
        zeros_column = torch.zeros(N, 1, device=h.device)
        return torch.cat([h, zeros_column], dim=1)

    def interpolate(self, batch, time):
        # for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
        for index, interp_param in self.interpolant_param_variables.items():
            interpolant = self.interpolants[f"{interp_param.variable_name}"]
            if interpolant is None:
                batch[f"{interp_param.variable_name}_t"] = batch[f"{interp_param.variable_name}"]
            else:
                if interp_param.variable_name == "edge_attr":
                    _, batch[f"{interp_param.variable_name}_t"], _ = interpolant.interpolate_edges(
                        batch.batch, batch[f"{interp_param.variable_name}"], batch["edge_index"],
                        time
                    )
                else:
                    _, batch[f"{interp_param.variable_name}_t"], _ = interpolant.interpolate(
                        batch.batch, batch[f"{interp_param.variable_name}"], time
                    )
            if "concat" in interp_param or "discrete" in interp_param.interpolant_type:
                batch[f"{interp_param.variable_name}_t"] = F.one_hot(
                    batch[f"{interp_param.variable_name}_t"], interp_param.num_classes
                ).float()

        return batch

    def aggregate_discrete_variables(self, batch):
        """
        Concatenate the flagged variable on its target and save the original in the batch via _og.
        """
        for interp_param in self.interpolant_params.variables:
            if 'concat' in interp_param and interp_param['concat'] is not None:
                batch[f"{interp_param.concat}_og"] = batch[f"{interp_param.concat}_t"]
                batch[f"{interp_param.concat}_t"] = torch.concat(
                    [batch[f"{interp_param.concat}_t"], batch[f"{interp_param.variable_name}_t"]],
                    dim=-1
                )
        return batch

    def separate_discrete_variables(self, out, batch):
        """
        Iterates throgh all outputs and restores interpolation for any aggregated variables.
        Converts output logits to the input necessary for Interpolant.step()
            - Discrete Diffusion Models assume class probabilities are given
            - Discrete Flow Models operate on the raw logits.
        Produces [Variable]_hat
        """
        for interp_param in self.interpolant_params.variables:
            if "concat" in interp_param or "discrete" in interp_param.interpolant_type:
                key = interp_param.variable_name
                combined_keys = [key]
                interpolant_type = interp_param.interpolant_type
                if "concat" in interp_param:
                    target = interp_param.concat
                    combined_keys.append(target)
                    K = interp_param.num_classes
                    N = out[f"{target}_logits"].shape[-1]
                    out[f"{target}_logits"], out[f"{key}_logits"] = torch.split(
                        out[f"{target}_logits"], [N - K, K], dim=-1
                    )
                    interpolant_type = self.interpolant_param_variables[target].interpolant_type

                for _key in combined_keys:
                    if f"{_key}_og" in batch:
                        batch[f"{_key}_t"] = batch[f"{_key}_og"]
                    if self.interpolants[_key] and self.interpolants[_key].prior_type in ["absorb",
                                                                                          "mask"]:
                        logits = out[f"{_key}_logits"].clone()
                        logits[:, -1] = -1e9
                    else:
                        logits = out[f"{_key}_logits"]
                    if "diffusion" in interpolant_type:  # ! Diffusion operates over the probability
                        out[f"{_key}_hat"] = logits.softmax(dim=-1)
                    else:  # ! DFM assumes that you operate over the logits
                        out[f"{_key}_hat"] = out[f"{_key}_logits"]

        return out, batch

    def validation_step(self, batch, batch_idx):
        if self.batch_preprocessor is not None:
            batch = self.batch_preprocessor(batch)
        time = self.sample_time(batch)
        out, batch, time = self(batch, time)
        loss, predictions = self.calculate_loss(batch, out, time, "val")
        return loss

    def training_step(self, batch, batch_idx):
        if self.batch_preprocessor is not None:
            batch = self.batch_preprocessor(batch)
        # ! Swapping names for now
        time = self.sample_time(batch)
        out, batch, time = self(batch, time)
        loss, predictions = self.calculate_loss(batch, out, time, "train")
        return loss

    def on_train_batch_end(self, outputs, batch, b_idx):
        self.dynamics.update_ema_parameters()
        if self.self_conditioning_module is not None:
            self.self_conditioning_module.update_ema_parameters()

    def calculate_loss(self, batch, out, time, stage="train"):
        batch_geo = batch.batch
        batch_size = int(batch.batch.max()) + 1
        ws_t = self.interpolants[self.global_variable].loss_weight_t(time)
        loss = 0
        predictions = {}
        for key, loss_fn in self.loss_functions.items():
            # Use configurable loss clamps if enabled
            level = None
            if self.use_loss_clamps and self.current_epoch >= self.loss_clamp_epoch_threshold:
                level = self.loss_clamps.get(key, None)
            elif not self.use_loss_clamps and key == "x":
                # Keep the original hardcoded logic for backwards compatibility
                level = 10.
                
            if "edge" in key:
                sub_loss, sub_pred = loss_fn.edge_loss(
                    batch_geo,
                    out['edge_attr_logits'],
                    batch['edge_attr'],
                    index=batch['edge_index'][1],
                    num_atoms=batch_geo.size(0),
                    batch_weight=ws_t,
                    level=level,
                )
            else:
                if loss_fn.continuous:
                    sub_loss, sub_pred = loss_fn(
                        batch_geo, out[f'{key}_hat'], batch[f'{key}'], batch_weight=ws_t,
                        level=level
                    )
                else:
                    true_data = batch[f'{key}']
                    if len(true_data.shape) > 1:
                        if true_data.size(1) == 1:
                            true_data = true_data.unsqueeze(1)
                        else:
                            true_data = true_data.argmax(dim=-1)
                    sub_loss, sub_pred = loss_fn(
                        batch_geo, out[f'{key}_logits'], true_data, batch_weight=ws_t, level=level
                    )

            self.log(f"{stage}/{key}_loss", sub_loss, batch_size=batch_size, prog_bar=True)
            loss = loss + sub_loss

            if loss_fn.use_distance in ["single", "triple"]:
                if "Z_hat" in out.keys() and loss_fn.use_distance == "triple":
                    z_hat = out["Z_hat"]
                else:
                    z_hat = None
                distance_loss_tp, distance_loss_tz, distance_loss_pz = loss_fn.distance_loss(
                    batch_geo, out[f'{key}_hat'], batch[f'{key}'], z_hat
                )
                distance_loss = distance_loss_tp + distance_loss_tz + distance_loss_pz
                self.log(f"{stage}/distance_loss", distance_loss, batch_size=batch_size)
                self.log(f"{stage}/distance_loss_tp", distance_loss_tp, batch_size=batch_size)
                self.log(f"{stage}/distance_loss_tz", distance_loss_tz, batch_size=batch_size)
                self.log(f"{stage}/distance_loss_pz", distance_loss_pz, batch_size=batch_size)
                loss = loss + loss_fn.distance_scale * distance_loss
        if self.loss_fn is not None:
            add_loss = self.loss_fn(batch, out, time, ws_t, stage="train")
            loss += add_loss
            self.log(f"{stage}/additional_loss_term", add_loss, batch_size=batch_size,
                     prog_bar=True)
        self.log(f"{stage}/loss", loss, batch_size=batch_size)
        return loss, predictions

    def forward(self, batch, time):
        """
        This forward function assumes we are doing some form (including none) of interpolation on positions X, node features H, and edge attributes edge_attr.
        1. Sample time from the distribution that is defined via the X interpolant params
        2. Shift X to 0 CoM, add absorbing state for H, create fully connected graph and edge features for edge_attr
        3. Interpolate all needed variables which are defined by "string" args in the config.
        4. Aggregate all the discrete non-edge features in the H variable for modeling.
        5. Dynamics forward pass to predict clean data given noisy data.
        6. Seperate the aggregated discrete predictions for easier loss calculation.
        """
        batch_size = int(batch.batch.max()) + 1
        batch = self.pre_format_molecules(batch, batch_size)
        batch = self.interpolate(batch, time)  # ! discrete variables are one hot after this point
        if self.self_conditioning_module is not None:
            batch, _ = self.self_conditioning(batch, time)
        batch = self.aggregate_discrete_variables(batch)
        out = self.dynamics(batch, time)
        out, batch = self.separate_discrete_variables(out, batch)
        return out, batch, time

    def one_hot(self, batch):
        """
        Convert class indices to one hot vectors.
        """
        for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
            if interp_param.interpolant_type is not None and "discrete" in interp_param.interpolant_type:
                batch[f"{interp_param.variable_name}_t"] = F.one_hot(
                    batch[f"{interp_param.variable_name}_t"], interp_param.num_classes
                ).float()
        return batch

    def initialize_inference(self):
        if self.sampling_params.node_distribution:
            with open(self.sampling_params.node_distribution, "rb") as f:
                node_dict = pickle.load(f)
            max_n_nodes = max(node_dict.keys())
            n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
            for key, value in node_dict.items():
                n_nodes[key] += value

            n_nodes = n_nodes / n_nodes.sum()
        else:
            n_nodes = None
        return n_nodes

    # def on_after_backward(self):
    #     # Compute and log the maximum gradient norm before clipping
    #     max_grad_norm = 0
    #     for p in self.dynamics.parameters():
    #         if p.grad is not None:
    #             param_norm = p.grad.data.norm(2)
    #             if param_norm > max_grad_norm:
    #                 max_grad_norm = param_norm

    #     # Log the maximum gradient norm before clipping
    #     print("TEST", max_grad_norm)
    #     results = {'train/max_grad_norm_before_clipping': max_grad_norm}
    #     self.log_dict(results, sync_dist=True)

    @torch.no_grad()
    def sample(self, num_samples=None, timesteps=500, time_discretization="linear", batch=None,
            num_atoms=None, pre_format=True):
        """
        Generates num_samples. Can supply a batch for inital starting points for conditional sampling for any interpolants set to None.
        """

        assert num_samples is not None or batch is not None

        time_type = self.interpolants[self.global_variable].time_type
        if "sample_time_discretization" in self.interpolant_params:
            time_discretization = self.interpolant_params["sample_time_discretization"]
        if time_type == "continuous":
            if time_discretization == "linear":
                timeline = torch.linspace(
                    self.interpolant_params.min_t, 1, timesteps + 1
                ).tolist()  # [0, 1.0] timestpes + 1
            elif time_discretization == "log":
                timeline = (
                    (1 - torch.logspace(-2, 0, timesteps + 1)).flip(dims=[0]).tolist()
                )  # [0, 0.99] #timestpes + 1
            # timeline = torch.logspace(-2, 0, timesteps + 1) #[0.01, 1.0]
            DT = [t1 - t0 for t0, t1 in zip(timeline[:-1], timeline[1:])]  # timesteps
        else:
            timeline = torch.arange(timesteps + 1)
            DT = [1 / timesteps] * timesteps

        if batch is None:

            if num_atoms is None:
                if self.node_distribution is not None:
                    num_atoms = torch.multinomial(
                        input=self.node_distribution,
                        num_samples=num_samples,
                        replacement=True,
                    )
                else:
                    num_atoms = torch.randint(20, 55, (num_samples,)).to(torch.int64)
            batch_index = torch.repeat_interleave(torch.arange(num_samples), num_atoms).to(
                self.device)
            edge_index = (
                torch.eq(batch_index.unsqueeze(0), batch_index.unsqueeze(-1)).int().fill_diagonal_(
                    0).to(self.device)
            )  # N x N
            edge_index, _ = dense_to_sparse(edge_index)  # 2 x E
            edge_index = sort_edge_index(edge_index, sort_by_row=False)
            total_num_atoms = num_atoms.sum().item()
        else:
            batch_size = int(batch.batch.max()) + 1
            if pre_format:
                if self.batch_preprocessor is not None:
                    batch = self.batch_preprocessor(batch)
                batch = self.pre_format_molecules(batch, batch_size=batch_size)
            batch_index = batch["batch"]
            edge_index = batch["edge_index"]
            total_num_atoms = batch_index.shape[0]
            num_samples = batch_size

        data, prior = {}, {}
        # Sample from all Priors
        for key, interpolant in self.interpolants.items():
            if interpolant is None:
                if batch is not None:
                    prior[key] = batch[key]
                    data[f"{key}_t"] = prior[key]
                else:
                    # If no batch is supplied just give zeros
                    data[f"{key}_t"] = torch.zeros(
                        (total_num_atoms, self.interpolant_param_variables[key].num_classes)
                    ).to(self.device)
                    if "offset" in self.interpolant_param_variables[key]:
                        data[f"{key}_t"] += self.interpolant_param_variables[key].offset
                continue
            if "edge" in key:
                shape = (edge_index.size(1), interpolant.num_classes)
                prior[key], edge_index = interpolant.prior_edges(batch_index, shape, edge_index,
                                                                 self.device)
                data[f"{key}_t"] = prior[key]
            else:
                shape = (total_num_atoms, interpolant.num_classes)
                data[f"{key}_t"] = prior[key] = interpolant.prior(batch_index, shape, self.device)
        # Iterate through time, query the dynamics, apply interpolant step update

        out = {}
        for idx in tqdm(list(range(len(DT))), total=len(DT)):
            t = timeline[idx]
            dt = DT[idx]
            time = torch.tensor([t] * num_samples).to(self.device)
            data = self.one_hot(data)
            # Apply Self Conditioning
            pre_conditioning_variables = {}
            # ! Try turning off self conditioning --> fixed some but still had edge blow ups can try adding norms here TODO
            if self.self_conditioning_module is not None:
                data, pre_conditioning_variables = self.self_conditioning(data, time,
                                                                          conditional_batch=out)
            data = self.aggregate_discrete_variables(data)
            data["batch"] = batch_index
            data["edge_index"] = edge_index
            out = self.dynamics(data, time, conditional_batch=out, timesteps=timesteps)
            # ! Error is for FM sampling EQGAT is producing NANs in discrete logits
            out, data = self.separate_discrete_variables(out, data)

            for key in pre_conditioning_variables:
                data[key] = pre_conditioning_variables[key]
            for key, interpolant in self.interpolants.items():
                if interpolant is None:
                    prior[key] = batch[key]
                    data[f"{key}_t"] = prior[key]
                    continue
                if "edge" in key:
                    edge_index, data[f"{key}_t"] = interpolant.step_edges(
                        batch_index,
                        edge_index=edge_index,
                        edge_attr_t=data[f"{key}_t"],
                        edge_attr_hat=out[f"{key}_hat"],
                        time=time,
                        dt=dt,
                    )
                else:
                    data[f"{key}_t"] = interpolant.step(
                        xt=data[f"{key}_t"],
                        x_hat=out[f"{key}_hat"],
                        x0=prior[key],
                        batch=batch_index,
                        time=time,
                        dt=dt,
                    )

        # samples = {key: data[f"{key}_t"] for key in self.interpolants.keys()}

        ### TMP change here
        samples = {}
        for interp_param in self.interpolant_params.variables:
            key = interp_param.variable_name
            if "discrete" in interp_param.interpolant_type:
                samples[key] = torch.argmax(out[f"{key}_hat"], dim=-1)
            else:
                samples[key] = out[f"{key}_hat"]

        samples["batch"] = batch_index
        samples["edge_index"] = edge_index

        for interp_params in self.interpolant_params.variables:
            if "offset" in interp_params:
                samples[interp_params.variable_name] -= interp_params.offset

        return samples

    def self_conditioning(self, batch, time, conditional_batch=None):
        pre_conditioning_variables = {}
        if self.training:
            with torch.no_grad():
                batch = self.aggregate_discrete_variables(batch)
                out = self.dynamics(batch, time)
                conditional_batch, batch = self.separate_discrete_variables(out, batch)
                for key in conditional_batch:
                    conditional_batch[key].detach()
            batch, pre_conditioning_variables = self.self_conditioning_module(batch,
                                                                              conditional_batch)
            if torch.rand(1).item() <= 0.5:
                for key in pre_conditioning_variables:
                    # hack to avoid unused parameters error
                    batch[key] = pre_conditioning_variables[key] + 0 * batch[key]
        else:
            if conditional_batch is not None and len(conditional_batch) > 0:
                batch, pre_conditioning_variables = self.self_conditioning_module(batch,
                                                                                  conditional_batch)
        return batch, pre_conditioning_variables
