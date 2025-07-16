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
import torch.nn as nn


class ModelWithEMA(nn.Module):
    def __init__(self, model, ema=True, ema_decay=0.999):
        """
        Initializes the ModelWithEMA wrapper.

        Args:
            model (nn.Module): The base model.
            ema (bool): Whether to use EMA for evaluation.
            ema_decay (float): Decay rate for the EMA model.
        """
        super().__init__()
        self.model = model
        self.ema = ema
        self.ema_decay = ema_decay

        if ema:
            avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay)
            self.ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=avg_fn)

            # Ensure ema_model parameters are not trainable
            for param in self.ema_model.parameters():
                param.requires_grad = False

    def update_ema_parameters(self):
        if self.ema:
            self.ema_model.update_parameters(self.model)


    def forward(self, *args, **kwargs):
        """
        Forward pass through the appropriate model.
        Uses the base model during training and the EMA model during evaluation if EMA is enabled.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output from the model.
        """
        if not self.ema or self.training:
            return self.model(*args, **kwargs)
        return self.ema_model(*args, **kwargs)
