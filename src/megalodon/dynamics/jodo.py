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


import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import dense_to_sparse, softmax
from torch_scatter import scatter


def coord2dist(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    return radial


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def coord2diff_adj(x, edge_index, spatial_th=2.0):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    with torch.no_grad():
        adj_spatial = radial.clone()
        adj_spatial[adj_spatial <= spatial_th] = 1.0
        adj_spatial[adj_spatial > spatial_th] = 0.0
    return radial, adj_spatial


def to_dense_edge_attr(edge_index, edge_attr, edge_final, bs, n_nodes):
    edge_idx1, edge_idx2 = edge_index
    idx0 = torch.div(edge_idx1, n_nodes, rounding_mode='floor')
    idx1 = edge_idx1 - idx0 * n_nodes
    idx2 = edge_idx2 - idx0 * n_nodes
    idx = idx0 * n_nodes * n_nodes + idx1 * n_nodes + idx2
    idx = idx.unsqueeze(-1).expand(edge_attr.size())
    edge_final.scatter_add_(0, idx, edge_attr)
    return edge_final.reshape(bs, n_nodes, n_nodes, -1)


def remove_mean_with_mask(x, node_mask):
    # masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    # assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class TransMixLayer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm.
    Extra attention heads from adjacency matrix."""

    _alpha: OptTensor

    def __init__(
        self,
        x_channels: int,
        out_channels: int,
        extra_heads: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        inf: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.extra_heads = extra_heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.sub_heads = sub_heads = heads - extra_heads
        self.sub_channels = sub_channels = (heads * out_channels) // sub_heads
        self.set_inf = inf

        self.lin_key = Linear(in_channels, sub_heads * sub_channels, bias=bias)
        self.lin_query = Linear(in_channels, sub_heads * sub_channels, bias=bias)
        self.lin_value = Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = Linear(edge_dim, sub_heads * sub_channels, bias=False)
        self.lin_edge1 = Linear(edge_dim, heads * out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()

    def forward(self, x: OptTensor, edge_index: Adj, edge_attr: OptTensor, extra_heads: OptTensor) -> Tensor:
        (
            H,
            E,
            C,
        ) = (
            self.heads,
            self.sub_heads,
            self.out_channels,
        )

        # expand the extra heads
        cur_extra_heads = extra_heads.size(-1)
        if cur_extra_heads != self.extra_heads:
            n_expand = self.extra_heads // cur_extra_heads
            extra_heads = extra_heads.unsqueeze(-1).repeat(1, 1, n_expand)
            extra_heads = extra_heads.reshape(-1, self.extra_heads)

        x_feat = x
        query = self.lin_query(x_feat).reshape(-1, E, self.sub_channels)
        key = self.lin_key(x_feat).reshape(-1, E, self.sub_channels)
        value = self.lin_value(x_feat).reshape(-1, H, C)
        out_x = self.propagate(
            edge_index=edge_index, query=query, key=key, value=value, extra_heads=extra_heads, edge_attr=edge_attr
        )

        out_x = out_x.view(-1, self.heads * self.out_channels)

        return out_x

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        extra_heads: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        edge_attn = self.lin_edge0(edge_attr).view(-1, self.sub_heads, self.sub_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)

        # set 0 to -inf/1e-10 in extra_heads
        if self.set_inf:
            extra_inf_heads = extra_heads.clone()
            # extra_inf_heads[extra_inf_heads==0.] = -float('inf')
            extra_inf_heads[extra_inf_heads == 0.0] = -1e10
            alpha = torch.cat([extra_inf_heads, alpha], dim=-1)
        else:
            alpha = torch.cat([extra_heads, alpha], dim=-1)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * torch.tanh(self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)


class CondGaussianLayer(nn.Module):
    """Gaussian basis function layer for 3D distance features, with time embedding condition"""

    def __init__(self, K, time_dim):
        super().__init__()
        self.K = K - 1
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2))
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, time_emb=None):
        if time_emb is not None:
            scale, shift = self.time_mlp(time_emb).chunk(2, dim=1)
            x = x * (scale + 1) + shift
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return torch.cat([x, gaussian(x, mean, std).type_as(self.means.weight)], dim=-1)


class LearnedSinusodialposEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8"""

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class MultiCondEquiUpdate(nn.Module):
    """Update atom coordinates equivariantly, use time emb condition."""

    def __init__(self, hidden_dim, edge_dim, dist_dim, time_dim, extra_heads):
        super().__init__()
        self.coord_norm = CoorsNorm(scale_init=1e-2)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, hidden_dim * 2))
        input_ch = hidden_dim * 2 + edge_dim + dist_dim
        update_heads = 1 + extra_heads
        self.input_lin = nn.Linear(input_ch, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, update_heads, bias=False)
        )

    def forward(self, h, pos, edge_index, edge_attr, dist, time_emb, adj_extra):
        row, col = edge_index
        h_input = torch.cat([h[row], h[col], edge_attr, dist], dim=1)
        coord_diff = pos[row] - pos[col]
        coord_diff = self.coord_norm(coord_diff)

        if time_emb is not None:
            shift, scale = self.time_mlp(time_emb).chunk(2, dim=1)
            inv = modulate(self.ln(self.input_lin(h_input)), shift, scale)
        else:
            inv = self.ln(self.input_lin(h_input))
        inv = torch.tanh(self.coord_mlp(inv))

        # multi channel adjacency matrix
        adj_dense = torch.ones((adj_extra.size(0), 1), device=adj_extra.device)
        adjs = torch.cat([adj_dense, adj_extra], dim=-1)
        inv = (inv * adjs).mean(-1, keepdim=True)

        # aggregate position
        trans = coord_diff * inv
        agg = scatter(trans, edge_index[0], 0, reduce='add', dim_size=pos.size(0))
        pos = pos + agg

        return pos


class EquivariantMixBlock(nn.Module):
    """Equivariant block based on graph relational transformer layer."""

    def __init__(
        self,
        node_dim,
        edge_dim,
        time_dim,
        num_extra_heads,
        num_heads,
        cond_time,
        dist_gbf,
        softmax_inf,
        mlp_ratio=2,
        act=nn.SiLU(),
        dropout=0.0,
        gbf_name='GaussianLayer',
        trans_name='TransMixLayer',
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.cond_time = cond_time
        self.dist_gbf = dist_gbf
        if dist_gbf:
            dist_dim = edge_dim
        else:
            dist_dim = 1
        self.edge_emb = nn.Linear(edge_dim + dist_dim, edge_dim)
        self.node2edge_lin = nn.Linear(node_dim, edge_dim)

        # message passing layer
        self.attn_mpnn = eval(trans_name)(
            node_dim, node_dim // num_heads, num_extra_heads, num_heads, edge_dim=edge_dim, inf=softmax_inf
        )

        # Normalization for MPNN
        self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
        self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)

        # Feed forward block -> node.
        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)
        self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)

        # Feed forward block -> edge.
        self.ff_linear3 = nn.Linear(edge_dim, edge_dim * mlp_ratio)
        self.ff_linear4 = nn.Linear(edge_dim * mlp_ratio, edge_dim)
        self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)

        # equivariant edge update layer
        self.equi_update = MultiCondEquiUpdate(node_dim, edge_dim, dist_dim, time_dim, num_extra_heads)

        self.node_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, node_dim * 6))
        self.edge_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, edge_dim * 6))

        if self.dist_gbf:
            self.dist_layer = eval(gbf_name)(dist_dim, time_dim)

    def _ff_block_node(self, x):
        x = self.dropout(self.act(self.ff_linear1(x)))
        return self.dropout(self.ff_linear2(x))

    def _ff_block_edge(self, x):
        x = self.dropout(self.act(self.ff_linear3(x)))
        return self.dropout(self.ff_linear4(x))

    def forward(self, pos, h, edge_attr, edge_index, node_mask, extra_heads, node_time_emb=None, edge_time_emb=None):
        """
        Params:
            pos: [B*N, 3]
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
            node_mask: [B*N, 1]
            extra_heads: [N_edge, extra_heads]
        """
        h_in_node = h
        h_in_edge = edge_attr

        # obtain distance feature
        distance = coord2dist(pos, edge_index)
        if self.dist_gbf:
            distance = self.dist_layer(distance, edge_time_emb)
        edge_attr = self.edge_emb(torch.cat([distance, edge_attr], dim=-1))

        # time (noise level) condition
        if self.cond_time:
            (
                node_shift_msa,
                node_scale_msa,
                node_gate_msa,
                node_shift_mlp,
                node_scale_mlp,
                node_gate_mlp,
            ) = self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            (
                edge_shift_msa,
                edge_scale_msa,
                edge_gate_msa,
                edge_shift_mlp,
                edge_scale_mlp,
                edge_gate_mlp,
            ) = self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)

            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
            edge_attr = modulate(self.norm1_edge(edge_attr), edge_shift_msa, edge_scale_msa)
        else:
            h = self.norm1_node(h)
            edge_attr = self.norm1_edge(edge_attr)

        # apply transformer-based message passing, update node features and edge features (FFN + norm)
        h_node = self.attn_mpnn(h, edge_index, edge_attr, extra_heads)
        h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
        h_edge = self.node2edge_lin(h_edge)

        h_node = h_in_node + node_gate_msa * h_node if self.cond_time else h_in_node + h_node
        h_node = (
            modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp) * node_mask
            if self.cond_time
            else self.norm2_node(h_node) * node_mask
        )
        h_out = (
            (h_node + node_gate_mlp * self._ff_block_node(h_node)) * node_mask
            if self.cond_time
            else (h_node + self._ff_block_node(h_node)) * node_mask
        )

        h_edge = h_in_edge + edge_gate_msa * h_edge if self.cond_time else h_in_edge + h_edge
        h_edge = (
            modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp)
            if self.cond_time
            else self.norm2_edge(h_edge)
        )
        h_edge_out = (
            h_edge + edge_gate_mlp * self._ff_block_edge(h_edge)
            if self.cond_time
            else h_edge + self._ff_block_edge(h_edge)
        )

        # apply equivariant coordinate update
        pos = self.equi_update(h_out, pos, edge_index, h_edge_out, distance, edge_time_emb, extra_heads)

        return h_out, h_edge_out, pos


class DGT_concat(nn.Module):
    """Diffusion Graph Transformer with self-conditioning."""

    def __init__(
        self,
        num_h_features,
        nf,
        n_heads,
        dropout,
        dist_gbf,
        gbf_name,
        edge_quan_th,
        n_extra_heads,
        CoM,
        mlp_ratio,
        spatial_cut_off,
        softmax_inf,
        edge_ch,
        cond_time,
        n_layers,
        trans_name,
    ):
        super().__init__()

        in_node_dim = num_h_features
        hidden_dim = nf
        edge_hidden_dim = nf // 4
        n_heads = n_heads
        dropout = dropout
        self.dist_gbf = dist_gbf = dist_gbf
        gbf_name = gbf_name
        self.edge_th = edge_quan_th
        n_extra_heads = n_extra_heads
        self.CoM = CoM
        mlp_ratio = mlp_ratio
        self.spatial_cut_off = spatial_cut_off
        softmax_inf = softmax_inf

        if dist_gbf:
            dist_dim = edge_hidden_dim
        else:
            dist_dim = 1

        in_edge_dim = edge_ch * 2 + dist_dim
        self.cond_time = cond_time = cond_time
        self.n_layers = n_layers = n_layers
        time_dim = hidden_dim * 4
        self.dist_dim = dist_dim
        self.node_emb = nn.Linear(in_node_dim * 2, hidden_dim)
        self.edge_emb = nn.Linear(in_edge_dim, edge_hidden_dim)

        if self.dist_gbf:
            self.dist_layer = eval(gbf_name)(dist_dim, time_dim)

        cat_node_dim = (hidden_dim * 2) // n_layers
        cat_edge_dim = (edge_hidden_dim * 2) // n_layers

        for i in range(n_layers):
            self.add_module(
                "e_block_%d" % i,
                EquivariantMixBlock(
                    hidden_dim,
                    edge_hidden_dim,
                    time_dim,
                    n_extra_heads,
                    n_heads,
                    cond_time,
                    dist_gbf,
                    softmax_inf,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    gbf_name=gbf_name,
                    trans_name=trans_name,
                ),
            )
            self.add_module("node_%d" % i, nn.Linear(hidden_dim, cat_node_dim))
            self.add_module("edge_%d" % i, nn.Linear(edge_hidden_dim, cat_edge_dim))

        self.node_pred_mlp = nn.Sequential(
            nn.Linear(cat_node_dim * n_layers + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, in_node_dim),
        )
        self.edge_type_mlp = nn.Sequential(
            nn.Linear(cat_edge_dim * n_layers + edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim // 2, edge_ch - 1),
        )
        self.edge_exist_mlp = nn.Sequential(
            nn.Linear(cat_edge_dim * n_layers + edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim // 2, 1),
        )

        if cond_time:
            learned_dim = 16
            sinu_pos_emb = LearnedSinusodialposEmb(learned_dim)
            self.time_mlp = nn.Sequential(
                sinu_pos_emb, nn.Linear(learned_dim + 1, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
            )

    def forward(self, t, xh, node_mask, edge_mask, context=None, *args, **kwargs):
        """
        Parameters
        ----------
        t: [B] time steps in [0, 1]
        xh: [B, N, ch1] atom feature (positions, types, formal charges)
        node_mask: [B, N, 1]
        edge_mask: [B*N*N, 1]
        context:
        kwargs: 'edge_x' [B, N, N, ch2]

        Returns
        -------

        """
        edge_x = kwargs['edge_x']
        cond_x = kwargs.get("cond_x", None)
        cond_edge_x = kwargs.get("cond_edge_x", None)
        cond_adj_2d = kwargs.get("cond_adj_2d", None)

        bs, n_nodes, dims = xh.shape
        pos = xh[:, :, 0:3].clone().reshape(bs * n_nodes, -1)
        h = xh[:, :, 3:].clone().reshape(bs * n_nodes, -1)

        adj_mask = edge_mask.reshape(bs, n_nodes, n_nodes)
        dense_index = adj_mask.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(adj_mask)

        # extra structural features
        if cond_x is None:
            cond_x = torch.zeros_like(xh)
            cond_edge_x = torch.zeros_like(edge_x)
            cond_adj_2d = torch.ones((edge_index.size(1), 1), device=edge_x.device)

        # concat self_cond node feature
        cond_pos = cond_x[:, :, 0:3].clone().reshape(bs * n_nodes, -1)
        cond_h = cond_x[:, :, 3:].clone().reshape(bs * n_nodes, -1)
        h = torch.cat([h, cond_h], dim=-1)

        if self.cond_time:
            noise_level = kwargs['noise_level']
            time_emb = self.time_mlp(noise_level)  # [B, hid_dim*4]
            node_time_emb = time_emb.unsqueeze(1).expand(-1, n_nodes, -1).reshape(bs * n_nodes, -1)
            edge_batch_id = torch.div(edge_index[0], n_nodes, rounding_mode='floor')
            edge_time_emb = time_emb[edge_batch_id]
        else:
            node_time_emb = None
            edge_time_emb = None

        # obtain distance from self_cond position
        distances, cond_adj_spatial = coord2diff_adj(cond_pos, edge_index, self.spatial_cut_off)

        if self.dist_gbf:
            gbf_distances = self.dist_layer(distances, edge_time_emb)
            distances = torch.where(distances.sum() == 0, distances.repeat(1, self.dist_dim), gbf_distances)

        cur_edge_attr = edge_x[dense_index]
        cond_edge_attr = cond_edge_x[dense_index]

        extra_adj = torch.cat([cond_adj_2d, cond_adj_spatial], dim=-1)
        edge_attr = torch.cat([cur_edge_attr, cond_edge_attr, distances], dim=-1)  # [N_edge, ch]

        # add structural features
        h = self.node_emb(h)
        edge_attr = self.edge_emb(edge_attr)

        # run the equivariant block
        atom_hids = [h]
        edge_hids = [edge_attr]
        for i in range(0, self.n_layers):
            h, edge_attr, pos = self._modules['e_block_%d' % i](
                pos, h, edge_attr, edge_index, node_mask.reshape(-1, 1), extra_adj, node_time_emb, edge_time_emb
            )
            if self.CoM:
                pos = remove_mean_with_mask(pos.reshape(bs, n_nodes, -1), node_mask).reshape(bs * n_nodes, -1)
            atom_hids.append(self._modules['node_%d' % i](h))
            edge_hids.append(self._modules['edge_%d' % i](edge_attr))

        # type prediction
        atom_hids = torch.cat(atom_hids, dim=-1)
        edge_hids = torch.cat(edge_hids, dim=-1)
        atom_pred = self.node_pred_mlp(atom_hids).reshape(bs, n_nodes, -1) * node_mask
        edge_pred = torch.cat([self.edge_exist_mlp(edge_hids), self.edge_type_mlp(edge_hids)], dim=-1)  # [N_edge, ch]

        # convert sparse edge_pred to dense form
        edge_final = torch.zeros_like(edge_x).reshape(bs * n_nodes * n_nodes, -1)  # [B*N*N, ch]
        edge_final = to_dense_edge_attr(edge_index, edge_pred, edge_final, bs, n_nodes)
        edge_final = 0.5 * (edge_final + edge_final.permute(0, 2, 1, 3))

        pos = pos * node_mask.reshape(-1, 1)

        if torch.any(torch.isnan(pos)):
            print('Warning: detected nan, resetting output to zero.')
            pos = torch.zeros_like(pos)

        pos = pos.reshape(bs, n_nodes, -1)

        return torch.cat([pos, atom_pred], dim=2), edge_final
