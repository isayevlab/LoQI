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
from torch import nn
from torch_scatter import scatter_mean


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "gelu_tanh": nn.GELU(approximate='tanh'),
    "sigmoid": nn.Sigmoid(),
}


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        num_hidden_layers: int = 0,
        activation: str = 'silu',
        dropout: float = 0.0,
        last_act: str = None,
    ):
        """
        Initialize the MLP.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_size (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output features.
            num_hidden_layers (int): Number of hidden layers.
            activation (str): Activation function to use ('relu', 'silu', etc.).
            dropout (float): Dropout probability (between 0 and 1).
        """
        super(MLP, self).__init__()

        if activation not in NONLINEARITIES:
            raise ValueError(f"Activation function must be one of {list(NONLINEARITIES.keys())}")

        self.act_layer = NONLINEARITIES[activation]

        # Create a list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(NONLINEARITIES[activation])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(NONLINEARITIES[activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim))
        if last_act:
            layers.append(NONLINEARITIES[last_act])

        # Combine all layers into a sequential module
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)



class PredictionHead(nn.Module):
    def __init__(self, num_classes, feat_dim, discrete=True, edge_prediction=False, distance_prediction=False):
        super().__init__()
        self.num_classes = num_classes
        self.discrete = discrete
        self.projection = MLP(feat_dim, feat_dim, num_classes)
        if edge_prediction:
            self.post_process = MLP(num_classes, num_classes, num_classes)
        if distance_prediction:
            self.post_process = MLP(feat_dim, feat_dim, num_classes, last_act="sigmoid")
            self.embedding = MLP(feat_dim, feat_dim, feat_dim)

    #! Even if we have a masking state we still predict it but always mask it even on the forward pass as done in MultiFlow. The loss is taken over teh logits so its not masked
    def forward(self, batch, H):
        logits = self.projection(H)
        if self.discrete:
            probs = F.softmax(logits, dim=-1)  # scatter_softmax(logits, index=batch, dim=0, dim_size=H.size(0))
        else:
            probs = H - scatter_mean(H, index=batch, dim=0)[batch]
        return logits, probs

    def predict_edges(self, batch, E, E_idx):
        #! EQGAT also uses hi hj and Xi-Xj along with f(E) see coordsatomsbonds.py line 121 https://github.com/tuanle618/eqgat-diff/blob/68aea80691a8ba82e00816c82875347cbda2c2e5/eqgat_diff/e3moldiffusion/coordsatomsbonds.py#L121C32-L121C44
        # import ipdb; ipdb.set_trace()
        E = self.projection(E)
        src, dst = E_idx
        N = batch.size(0)
        e_dense = torch.zeros(N, N, E.size(-1), device=E.device)
        e_dense[src, dst, :] = E
        e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
        e = e_dense[src, dst, :]  # E x 5
        logits = self.post_process(e)  # E x 5
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def predict_distances(self, batch, Z):
        input = Z + Z.permute(1, 0, 2)
        logits = self.post_process(input) * self.embedding(input)
        logits = self.projection(logits).squeeze(-1)
        return logits


class InterpolantLossFunction(nn.Module):
    def __init__(
        self,
        continuous=True,
        aggregation='mean',
        loss_scale=1.0,
        discrete_class_weight=None,
        use_distance=None,
        distance_scale=None,
    ):
        super().__init__()
        if continuous:
            self.f_continuous = nn.MSELoss(reduction='none')  # can also use HuberLoss
        else:
            if discrete_class_weight is None:
                self.f_discrete = nn.CrossEntropyLoss(reduction='none')
            else:
                self.f_discrete = nn.CrossEntropyLoss(weight=discrete_class_weight, reduction='none')
                #! We can up weight certain bonds to make sure this is correct
        self.continuous = continuous
        self.aggregation = aggregation
        self.scale = loss_scale
        self.use_distance = use_distance
        self.distance_scale = distance_scale

    def forward(self, batch, logits, data, batch_weight=None, element_weight=None, level=10000):
        # d (λx, λh, λe) = (3, 0.4, 2)
        batch_size = len(batch.unique())
        if self.continuous:
            loss = self.f_continuous(logits, data).mean(-1)  # [N] #! this hsould prbably be sum
            output = logits
        else:
            loss = self.f_discrete(logits, data)
            output = torch.argmax(logits, dim=-1)
        if element_weight is not None:
            loss = loss * element_weight
        loss = scatter_mean(loss, index=batch, dim=0, dim_size=batch_size)
        if batch_weight is not None:
            loss = loss * batch_weight  # .unsqueeze(1)
        if level is not None:
            loss = loss.clamp(0, level)
        # print(level)
        if self.aggregation == "mean":
            loss = self.scale * loss.mean()
        elif self.aggregation == "sum":
            loss = self.scale * loss.sum()

        return loss, output

    def backbone_loss(self, batch, logits, data, batch_weight, cutoff=2.5):
        # import ipdb; ipdb.set_trace()
        # a, b = self.forward(batch, logits, data)
        batch_size = len(batch.unique())
        grel = logits.unsqueeze(1) - logits.unsqueeze(0)
        trel = data.unsqueeze(1) - data.unsqueeze(0)
        gnorm = torch.linalg.norm(grel, dim=-1)
        tnorm = torch.linalg.norm(trel, dim=-1)
        mask = tnorm < cutoff
        mask.fill_diagonal_(False)
        gbackbone = gnorm * mask.int()
        tbackbone = tnorm * mask.int()
        loss = self.f_continuous(gbackbone, tbackbone)
        loss_mask = (batch.unsqueeze(0) == batch.unsqueeze(1)).int()
        loss = loss * loss_mask
        loss = loss.sum(-1)
        loss = scatter_mean(loss, index=batch, dim=0, dim_size=batch_size) * batch_weight
        return loss.sum() / batch_weight.sum()

    def edge_loss(self, batch, logits, data, index, num_atoms, batch_weight=None, element_weight=None, level=10000):
        batch_size = len(batch.unique())
        loss = self.f_discrete(logits, data)
        loss = 0.5 * scatter_mean(loss, index=index, dim=0, dim_size=num_atoms)  # Aggregate on the bonds first
        output = torch.argmax(logits, dim=-1)
        if element_weight:
            loss = loss * element_weight
        loss = scatter_mean(loss, index=batch, dim=0, dim_size=batch_size)
        if batch_weight is not None:
            loss = loss * batch_weight  # .unsqueeze(1)
        # loss = loss.clamp(0, level)
        if level is not None:
            loss = loss.clamp(0, level)
        if self.aggregation == "mean":
            loss = self.scale * loss.mean()
        elif self.aggregation == "sum":
            loss = self.scale * loss.sum()
        return loss, output

    def distance_loss(self, batch, X_pred, X_true, Z_pred=None, time=None, time_cutoff=0.5):
        if Z_pred is None:
            true_distance = torch.tensor([], device=X_true.device)
            x_pred_distance = torch.tensor([], device=X_true.device)
            batch_size = len(batch.unique())
            c_batch = []
            for element in range(batch_size):
                x_true = X_true[batch == element]
                x_pred = X_pred[batch == element]
                c_batch.extend([element] * x_true.size(0) * x_true.size(0))
                dist = torch.cdist(x_true, x_true).flatten()
                dist_pred = torch.cdist(x_pred, x_pred).flatten()
                true_distance = torch.cat([true_distance, dist], dim=-1)
                x_pred_distance = torch.cat([x_pred_distance, dist_pred], dim=-1)
            c_batch = torch.Tensor(c_batch).to(torch.int64).to(X_true.device)
            A = self.f_continuous(true_distance, x_pred_distance)
            time_filter = time > time_cutoff
            A = scatter_mean(A, c_batch, dim=0, dim_size=batch_size) * time_filter
            if self.aggregation == "mean":
                loss = A.mean()
            elif self.aggregation == "sum":
                loss = A.sum()
            return loss, 0, 0
        else:
            true_distance = torch.tensor([], device=X_true.device)
            x_pred_distance = torch.tensor([], device=X_true.device)
            z_pred_distance = torch.tensor([], device=X_true.device)
            batch_size = len(batch.unique())
            c_batch = []
            for element in range(batch_size):
                x_true = X_true[batch == element]
                x_pred = X_pred[batch == element]
                c_batch.extend([element] * x_true.size(0) * x_true.size(0))
                dist = torch.cdist(x_true, x_true).flatten()
                dist_pred = torch.cdist(x_pred, x_pred).flatten()
                dist_z = Z_pred[batch == element][:, batch == element].flatten()
                true_distance = torch.cat([true_distance, dist], dim=-1)
                x_pred_distance = torch.cat([x_pred_distance, dist_pred], dim=-1)
                z_pred_distance = torch.cat([z_pred_distance, dist_z], dim=-1)
            c_batch = torch.Tensor(c_batch).to(torch.int64).to(X_true.device)
            A = self.f_continuous(true_distance, x_pred_distance)
            B = self.f_continuous(true_distance, z_pred_distance)
            C = self.f_continuous(x_pred_distance, z_pred_distance)
            time_filter = time > time_cutoff
            A = scatter_mean(A, c_batch, dim=0, dim_size=batch_size) * time_filter
            B = scatter_mean(B, c_batch, dim=0, dim_size=batch_size) * time_filter
            C = scatter_mean(C, c_batch, dim=0, dim_size=batch_size) * time_filter
            if self.aggregation == "mean":
                return A.mean(), B.mean(), C.mean()
            elif self.aggregation == "sum":
                return A.sum(), B.sum(), C.sum()


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
            1,
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
                # import ipdb; ipdb.set_trace()
                adj_matrix[idx][jdx] = torch.nn.functional.one_hot(torch.randint(0, 5, (1,)), 5).squeeze(0)
    # print(adj_matrix)

    atom_embedder = nn.Linear(num_classes, 64)
    X = ligand_pos
    H = atom_embedder(F.one_hot(ligand_feats, num_classes).float())
    A = adj_matrix
    mask = batch_ligand.unsqueeze(1) == batch_ligand.unsqueeze(0)  # Shape: (75, 75)
    E_idx = mask.nonzero(as_tuple=False).t()
    self_loops = E_idx[0] != E_idx[1]
    E_idx = E_idx[:, self_loops]
    Z = atom_embedder(F.one_hot(ligand_feats, num_classes).float()).unsqueeze(1) * atom_embedder(
        F.one_hot(ligand_feats, num_classes).float()
    ).unsqueeze(0)

    source, target = E_idx
    E = A[source, target]  # E x 5
    # E_all = torch.cat((d.unsqueeze(1), a.unsqueeze(1), r_norm, E), dim=-1)  # E x 10
    edge_embedder = nn.Linear(5, 32)
    E = edge_embedder(E.float())

    loss_function = InterpolantLossFunction()
    # import ipdb; ipdb.set_trace()
    out = loss_function.distance_loss(batch_ligand, X, X, Z.sum(-1))
