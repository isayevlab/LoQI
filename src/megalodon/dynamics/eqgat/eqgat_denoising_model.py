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

from abc import ABC
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor
from torch_scatter import scatter, scatter_mean, scatter_softmax

from megalodon.dynamics.eqgat.eqgat_modules import (
    AdaptiveLayerNorm,
    DenseLayer,
    GatedEquivBlock,
    LayerNorm,
    SE3Norm,
)


def cross_product(a: Tensor, b: Tensor, dim: int) -> Tensor:
    if a.dtype != torch.float16 and b.dtype != torch.float16:
        return torch.linalg.cross(a, b, dim=dim)
    else:
        s1 = a[:, 1, :] * b[:, -1, :] - a[:, -1, :] * b[:, 1, :]
        s2 = a[:, -1, :] * b[:, 0, :] - a[:, 0, :] * b[:, -1, :]
        s3 = a[:, 0, :] * b[:, 1, :] - a[:, 1, :] * b[:, 0, :]
        cross = torch.stack([s1, s2, s3], dim=dim)
        return cross


class EQGATGlobalEdgeConvFinal(MessagePassing, ABC):
    """
    Slightly modified SO(3) equivariant graph attention convolution described in
    @inproceedings{
        le2022representation,
        title={Representation Learning on Biomolecular Structures using Equivariant Graph Attention},
        author={Tuan Le and Frank Noe and Djork-Arn{\'e} Clevert},
        booktitle={The First Learning on Graphs Conference},
        year={2022},
        url={https://openreview.net/forum?id=kv4xUo5Pu6}
    }

    Intention for this layer is to be used as a global fully-connected message passing layer.
    """

    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        edge_dim: int,
        eps: float = 1e-6,
        has_v_in: bool = False,
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
        use_cross_product: bool = False,
        edge_mp: bool = False,
        use_pos_norm: bool = True,
    ):
        super(EQGATGlobalEdgeConvFinal, self).__init__(node_dim=0, aggr=None, flow="source_to_target")

        assert edge_dim is not None

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        self.use_cross_product = use_cross_product
        self.silu = nn.SiLU()
        self.use_pos_norm = use_pos_norm
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3 if use_cross_product else 2
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        if use_pos_norm:
            self.posnorm = SE3Norm()
        else:
            self.posnorm = None

        self.edge_pre = DenseLayer(edge_dim, edge_dim)
        self.edge_dim = edge_dim
        # input_edge_dim = (
        #     2 * self.si + edge_dim + 2 + 2
        #     if self.use_pos_norm
        #     else 2 * self.si + edge_dim + 2
        # )
        input_edge_dim = 2 * self.si + edge_dim + 2 + 2

        self.edge_net = nn.Sequential(
            DenseLayer(input_edge_dim, self.si, bias=True, activation=nn.SiLU()),
            DenseLayer(self.si, self.v_mul * self.vi + self.si + 1 + edge_dim, bias=True),
        )
        self.edge_post = DenseLayer(edge_dim, edge_dim)

        self.edge_mp = edge_mp


        emlp = False

        if edge_mp:
            from torch_sparse import SparseTensor
            if emlp:
                self.edge_lin = nn.Sequential(
                    DenseLayer(2 * edge_dim + 3, edge_dim, activation=nn.SiLU()),
                    DenseLayer(edge_dim, edge_dim),
                )
            else:
                self.edge_lin = DenseLayer(2 * edge_dim + 3, edge_dim)
        # previously, still keep in case old model checkpoints are loaded
        else:
            self.edge_lin = None

        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(
            in_dims=(self.si, self.vi),
            hs_dim=self.si,
            hv_dim=self.vi,
            out_dims=(self.so, self.vo),
            norm_eps=eps,
            use_mlp=use_mlp_update,
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        if self.has_v_in:
            reset(self.vector_net)
        reset(self.scalar_net)
        reset(self.update_net)
        if self.posnorm:
            self.posnorm.reset_parameters()

    @staticmethod
    def get_triplet(edge_index: torch.Tensor, num_nodes: int):
        """
        Compute triplets of nodes and corresponding edge indices in a graph.

        Args:
            edge_index (torch.Tensor): The edge index tensor representing
            the connections between source and target nodes.
            num_nodes (int): The total number of nodes in the graph.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple containing:
                - input_edge_index (torch.Tensor): The input edge index tensor.
                - idx_i (torch.Tensor): Node indices i in the triplets (k->j->i).
                - idx_j (torch.Tensor): Node indices j in the triplets (k->j->i).
                - idx_k (torch.Tensor): Node indices k in the triplets (k->j->i).
                - idx_kj (torch.Tensor): Edge indices (k-j) in the triplets.
                - idx_ji (torch.Tensor): Edge indices (j->i) in the triplets.
        """
        assert edge_index.size(0) == 2
        input_edge_index = edge_index.clone()
        source, target = edge_index  # j->i
        # create identifiers for edges based on (source, target)
        value = torch.arange(source.size(0), device=source.device)
        # as row-index select the target (column) nodes --> transpose
        # create neighbours from j
        adj_t = SparseTensor(row=target, col=source, value=value, sparse_sizes=(num_nodes, num_nodes))
        # get neighbours from j
        adj_t_row = adj_t[source]
        # returns the target nodes (k) that include the source (j)
        # note there can be path i->j->k where k is i
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)
        # print(num_triplets)
        # Node indices (k->j->i) for triplets.
        idx_i = target.repeat_interleave(num_triplets)
        idx_j = source.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()  # get index for k
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        # print(idx_i); print(idx_j); print(idx_k)
        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return input_edge_index, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def edge_message_passing(
        self,
        p: Tensor,
        batch: Tensor,
        k: int,
        edge_index_full: Tensor,
        edge_attr_full: Tensor,
    ):
        """
        Perform message passing over edges in a graph. It computes KNN graph and uses all
        possible triplets in the KNN graph. The updated edge attribute average all
        possible distances, angles and edge attributes in the triplets to compuete
        updated edge attribute.

        Args:
            p (Tensor): The positional tensor representing node positions.
            batch (Tensor): The batch tensor containing the batch indices of nodes.
            k (int): The number of nearest neighbors for kNN graph construction.
            edge_index_full (Tensor): The edge index tensor of the full graph.
            edge_attr_full (Tensor): The edge attribute tensor of the full graph.

        Returns:
            Tensor: The updated edge attribute tensor after message passing.
        """
        num_nodes = p.size(0)

        E_full = torch.zeros(
            size=(num_nodes, num_nodes, edge_attr_full.size(-1)),
            device=edge_attr_full.device,
            dtype=edge_attr_full.dtype,
        )
        E_full[edge_index_full[0], edge_index_full[1], :] = edge_attr_full

        # create kNN graph
        edge_index_knn = knn_graph(x=p, k=k, batch=batch, flow="source_to_target")
        j, i = edge_index_knn

        p_ij = p[j] - p[i]
        p_ij_n = F.normalize(p_ij, p=2, dim=-1)
        d_ij = torch.pow(p_ij, 2).sum(-1, keepdim=True).sqrt()

        edge_ij = E_full[j, i, :]

        edge_index_knn, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.get_triplet(edge_index_knn, num_nodes=num_nodes)

        p_jk = -1.0 * p_ij_n[idx_kj]
        p_ji = p_ij_n[idx_ji]

        theta_ijk = torch.sum(p_jk * p_ji, -1, keepdim=True).clamp_(-1.0 + 1e-7, 1.0 - 1e-7)
        theta_ijk = torch.arccos(theta_ijk)
        d_ji = d_ij[idx_ji]
        d_jk = d_ij[idx_kj]
        edge_0 = edge_ij[idx_ji]
        edge_1 = edge_ij[idx_kj]
        f_ijk = torch.cat([edge_0, edge_1, theta_ijk, d_ji, d_jk], dim=-1)
        f_ijk = self.edge_lin(f_ijk)
        aggr_edges = scatter(
            src=f_ijk,
            index=idx_ji,
            dim=0,
            reduce="mean",
            dim_size=edge_index_knn.size(-1),
        )
        E_aggr = torch.zeros_like(E_full)
        E_aggr[edge_index_knn[0], edge_index_knn[1], :] = aggr_edges

        E_out = E_full + E_aggr

        edge_attr_full = E_out[edge_index_full[0], edge_index_full[1], :]

        return edge_attr_full

    def forward(
        self,
        x: Tuple[Tensor, Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch: Tensor,
        batch_lig: Tensor = None,
        pocket_mask: Tensor = None,
    ):
        s, v, p = x
        d, a, r, e = edge_attr

        e = self.edge_pre(e)

        if self.edge_mp:
            e = self.edge_message_passing(p=p, batch=batch, k=4, edge_index_full=edge_index, edge_attr_full=e)

        ms, mv, mp, me = self.propagate(
            edge_index=edge_index,
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            p=p,
            edge_attr=(d, a, r, e),
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        if self.posnorm:
            p = (
                p + self.posnorm(mp, batch, batch_lig, pocket_mask) * pocket_mask
                if pocket_mask is not None
                else p + self.posnorm(mp, batch)
            )
        else:
            p = p + mp * pocket_mask if pocket_mask is not None else p + mp
        e = F.silu(me + e)
        e = self.edge_post(e)

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        out = {"s": s, "v": v, "p": p, "e": e}
        return out

    def aggregate(
        self,
        inputs: Tuple[Tensor, Tensor, Tensor, Tensor],
        index: Tensor,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        p = scatter(inputs[2], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        edge = inputs[3]
        return s, v, p, edge

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        p_i: Tensor,
        p_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        d, a, r, e = edge_attr

        de0 = d.view(-1, 1)
        a0 = a.view(-1, 1)

        if self.use_pos_norm:
            d_i, d_j = (
                torch.pow(p_i, 2).sum(-1, keepdim=True).clamp(min=1e-6).sqrt(),
                torch.pow(p_j, 2).sum(-1, keepdim=True).clamp(min=1e-6).sqrt(),
            )
        else:
            d_i, d_j = torch.zeros_like(a0).to(a.device), torch.zeros_like(a0).to(a.device)
        aij = torch.cat([torch.cat([sa_i, sa_j], dim=-1), de0, a0, e, d_i, d_j], dim=-1)
        # else:
        #     aij = torch.cat([torch.cat([sa_i, sa_j], dim=-1), de0, a0, e], dim=-1)
        aij = self.edge_net(aij)

        fdim = aij.shape[-1]
        aij, gij = aij.split([fdim - 1, 1], dim=-1)
        fdim = aij.shape[-1]
        aij, edge = aij.split([fdim - self.edge_dim, self.edge_dim], dim=-1)
        pj = gij * r

        if self.has_v_in:
            aij, vij0 = aij.split([self.si, self.v_mul * self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            if self.use_cross_product:
                vij0, vij1, vij2 = vij0.chunk(3, dim=-1)
            else:
                vij0, vij1 = vij0.chunk(2, dim=-1)
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            if self.use_cross_product:
                v_ij_cross = cross_product(va_i, va_j, dim=1)
                nv2_j = vij2 * v_ij_cross
                nv_j = nv0_j + nv1_j + nv2_j
            else:
                nv_j = nv0_j + nv1_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j, pj, edge


class EQGATEdgeGNN(nn.Module):
    """_summary_
    EQGAT GNN Network updating node-level scalar, vectors and position features as well as edge-features.
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        hn_dim: Tuple[int, int] = (64, 16),
        edge_dim: Optional[int] = 16,
        num_layers: int = 5,
        latent_dim: Optional[int] = None,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        fully_connected: bool = True,
        recompute_edge_attributes: bool = True,
        edge_mp: bool = False,
        p1: bool = True,
        use_pos_norm: bool = True,
        ligand_pocket_interaction: bool = False,
    ):
        super(EQGATEdgeGNN, self).__init__()

        assert fully_connected

        self.num_layers = num_layers
        self.fully_connected = fully_connected
        self.recompute_edge_attributes = recompute_edge_attributes
        self.p1 = p1
        self.ligand_pocket_interaction = ligand_pocket_interaction

        self.sdim, self.vdim = hn_dim
        self.edge_dim = edge_dim

        convs = []

        for i in range(num_layers):
            ## second or second last layer
            # lb = (i == 1 or i == num_layers - 2)
            lb = (i % 2 == 0) and (i != 0)
            # new: every second layer
            edge_mp_select = lb & edge_mp
            convs.append(
                EQGATGlobalEdgeConvFinal(
                    in_dims=hn_dim,
                    out_dims=hn_dim,
                    edge_dim=edge_dim,
                    has_v_in=i > 0,
                    use_mlp_update=i < (num_layers - 1),
                    vector_aggr=vector_aggr,
                    use_cross_product=use_cross_product,
                    edge_mp=edge_mp_select,
                    use_pos_norm=use_pos_norm,
                )
            )

        self.convs = nn.ModuleList(convs)

        if latent_dim:
            norm_module = AdaptiveLayerNorm
        else:
            norm_module = LayerNorm

        self.norms = nn.ModuleList([norm_module(dims=hn_dim, latent_dim=latent_dim) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()

    def calculate_edge_attrs(
        self,
        edge_index: Tensor,
        edge_attr: OptTensor,
        pos: Tensor,
        sqrt: bool = True,
        batch: Tensor = None,
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        if self.ligand_pocket_interaction:
            mask = source != target
            pos[mask] = pos[mask] / torch.norm(pos[mask], dim=1).unsqueeze(1)
            a = pos[target] * pos[source]
        else:
            a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def to_dense_edge_tensor(self, edge_index, edge_attr, num_nodes):
        E = torch.zeros(
            num_nodes,
            num_nodes,
            edge_attr.size(-1),
            device=edge_attr.device,
            dtype=edge_attr.dtype,
        )
        E[edge_index[0], edge_index[1], :] = edge_attr
        return E

    def from_dense_edge_tensor(self, edge_index, E):
        return E[edge_index[0], edge_index[1], :]

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        p: Tensor,
        edge_index_global: Tensor,
        edge_attr_global: Tuple[Tensor, Tensor, Tensor, Tensor],
        z: OptTensor = None,
        batch: Tensor = None,
        context: OptTensor = None,
        batch_lig: OptTensor = None,
        pocket_mask: OptTensor = None,
    ) -> Dict:
        # edge_attr_xyz (distances, cosines, relative_positions, edge_features)
        # (E, E, E x 3, E x F)

        for i in range(len(self.convs)):
            edge_index_in = edge_index_global
            edge_attr_in = edge_attr_global

            if context is not None and (i == 1 or i == len(self.convs) - 1):
                s = s + context
            s, v = self.norms[i](x={"s": s, "v": v, "z": z}, batch=batch)
            out = self.convs[i](
                x=(s, v, p),
                batch=batch,
                edge_index=edge_index_in,
                edge_attr=edge_attr_in,
                batch_lig=batch_lig,
                pocket_mask=pocket_mask,
            )

            s, v, p, e = out["s"], out["v"], out["p"], out["e"]
            # p = p - scatter_mean(p, batch, dim=0)[batch]
            if self.recompute_edge_attributes:
                edge_attr_global = self.calculate_edge_attrs(
                    edge_index=edge_index_global,
                    pos=p,
                    edge_attr=e,
                    sqrt=True,
                    batch=batch if self.ligand_pocket_interaction else None,
                )

            e = edge_attr_global[-1]

        out = {"s": s, "v": v, "e": e, "p": p}

        return out


class PredictionHeadEdge(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int],
        edge_dim: int,
        num_atom_features: int,
        num_bond_types: int = 5,
        coords_param: str = "data",
    ) -> None:
        super(PredictionHeadEdge, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_atom_features = num_atom_features

        self.shared_mapping = DenseLayer(self.sdim, self.sdim, bias=True, activation=nn.SiLU())

        self.bond_mapping = DenseLayer(edge_dim, self.sdim, bias=True)

        self.bonds_lin_0 = DenseLayer(in_features=self.sdim + 1, out_features=self.sdim, bias=True)
        self.bonds_lin_1 = DenseLayer(in_features=self.sdim, out_features=num_bond_types, bias=True)
        self.coords_lin = DenseLayer(in_features=self.vdim, out_features=1, bias=False)
        self.atoms_lin = DenseLayer(in_features=self.sdim, out_features=num_atom_features, bias=True)

        self.coords_param = coords_param

        self.reset_parameters()

    def reset_parameters(self):
        self.shared_mapping.reset_parameters()
        self.coords_lin.reset_parameters()
        self.atoms_lin.reset_parameters()
        self.bonds_lin_0.reset_parameters()
        self.bonds_lin_1.reset_parameters()

    def forward(
        self,
        x: Dict,
        batch: Tensor,
        edge_index_global: Tensor,
        edge_index_global_lig: Tensor = None,
        batch_lig: Tensor = None,
        pocket_mask: Tensor = None,
        edge_mask: Tensor = None,
    ) -> Dict:
        s, v, p, e = x["s"], x["v"], x["p"], x["e"]
        s = self.shared_mapping(s)

        coords_pred = self.coords_lin(v).squeeze()
        atoms_pred = self.atoms_lin(s)

        if batch_lig is not None and pocket_mask is not None:
            s = (s * pocket_mask)[pocket_mask.squeeze(), :]
            j, i = edge_index_global_lig
            atoms_pred = (atoms_pred * pocket_mask)[pocket_mask.squeeze(), :]
            coords_pred = (coords_pred * pocket_mask)[pocket_mask.squeeze(), :]
            p = p[pocket_mask.squeeze(), :]
            # coords_pred = (
            #    coords_pred
            #    - scatter_mean(coords_pred, index=batch_lig, dim=0)[batch_lig]
            # )
            d = (coords_pred[i] - coords_pred[j]).pow(2).sum(-1, keepdim=True)

        elif self.coords_param == "data":
            j, i = edge_index_global
            n = s.size(0)
            coords_pred = p + coords_pred
            coords_pred = coords_pred - scatter_mean(coords_pred, index=batch, dim=0)[batch]
            d = (coords_pred[i] - coords_pred[j]).pow(2).sum(-1, keepdim=True)  # .sqrt()
        else:
            j, i = edge_index_global
            n = s.size(0)
            d = (p[i] - p[j]).pow(2).sum(-1, keepdim=True)  # .sqrt()
            coords_pred = coords_pred - scatter_mean(coords_pred, index=batch, dim=0)[batch]

        if edge_mask is not None and edge_index_global_lig is not None:
            n = len(batch_lig)
            e = (e * edge_mask.unsqueeze(1))[edge_mask]
            e_dense = torch.zeros(n, n, e.size(-1), device=e.device)
            e_dense[edge_index_global_lig[0], edge_index_global_lig[1], :] = e
            e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
            e = e_dense[edge_index_global_lig[0], edge_index_global_lig[1], :]
        else:
            e_dense = torch.zeros(n, n, e.size(-1), device=e.device)
            e_dense[edge_index_global[0], edge_index_global[1], :] = e
            e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
            e = e_dense[edge_index_global[0], edge_index_global[1], :]

        f = s[i] + s[j] + self.bond_mapping(e)
        edge = torch.cat([f, d], dim=-1)
        bonds_pred = F.silu(self.bonds_lin_0(edge))
        bonds_pred = self.bonds_lin_1(bonds_pred)

        out = {
            "coords_pred": coords_pred,
            "atoms_pred": atoms_pred,
            "bonds_pred": bonds_pred,
        }

        return out


class DenoisingEdgeNetwork(nn.Module):
    """_summary_
    Denoising network that inputs:
        atom features, edge features, position features
    The network is tasked for data prediction, i.e. x0 parameterization as commonly known in the literature:
        atom features, edge features, position features
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_atom_features: int,
        num_bond_types: int = 5,
        hn_dim: Tuple[int, int] = (256, 64),
        edge_dim: int = 32,
        num_layers: int = 5,
        latent_dim: Optional[int] = None,
        use_cross_product: bool = False,
        fully_connected: bool = True,
        recompute_edge_attributes: bool = True,
        vector_aggr: str = "mean",
        atom_mapping: bool = True,
        bond_mapping: bool = True,
        edge_mp: bool = False,
        p1: bool = True,
        use_pos_norm: bool = True,
        context_mapping: bool = False,
        num_context_features: int = 0,
        bond_prediction: bool = False,
        coords_param: str = "data",
        ligand_pocket_interaction: bool = False,
    ) -> None:
        super(DenoisingEdgeNetwork, self).__init__()
        self.bond_prediction = bond_prediction
        self.num_bond_types = num_bond_types

        self.ligand_pocket_interaction = ligand_pocket_interaction

        self.time_mapping_atom = DenseLayer(1, hn_dim[0])
        self.time_mapping_bond = DenseLayer(1, edge_dim)

        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        else:
            self.atom_mapping = nn.Identity()

        if bond_mapping or bond_prediction:
            if bond_prediction:
                num_bond_types = 1 * num_atom_features + 1
            self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
        else:
            self.bond_mapping = nn.Identity()

        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])
        self.bond_time_mapping = DenseLayer(edge_dim, edge_dim)

        self.context_mapping = context_mapping
        if self.context_mapping:
            self.context_mapping = DenseLayer(num_context_features, hn_dim[0])
            self.atom_context_mapping = DenseLayer(hn_dim[0], hn_dim[0])

        assert fully_connected

        self.sdim, self.vdim = hn_dim

        self.fully_connected = fully_connected

        assert fully_connected

        if latent_dim:
            if context_mapping:
                latent_dim_ = None
            else:
                latent_dim_ = latent_dim
        else:
            latent_dim_ = None

        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim_,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected,
            recompute_edge_attributes=recompute_edge_attributes,
            edge_mp=edge_mp,
            p1=p1,
            use_pos_norm=use_pos_norm,
            ligand_pocket_interaction=ligand_pocket_interaction,
        )

        self.prediction_head = PredictionHeadEdge(
            hn_dim=hn_dim,
            edge_dim=edge_dim,
            num_atom_features=num_atom_features,
            num_bond_types=num_bond_types,
            coords_param=coords_param,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        if hasattr(self.bond_mapping, "reset_parameters"):
            self.bond_mapping.reset_parameters()
        self.time_mapping_atom.reset_parameters()
        self.atom_time_mapping.reset_parameters()
        if self.context_mapping and hasattr(self.atom_context_mapping, "reset_parameters"):
            self.atom_context_mapping.reset_parameters()
        if self.context_mapping and hasattr(self.context_mapping, "reset_parameters"):
            self.context_mapping.reset_parameters()
        self.time_mapping_bond.reset_parameters()
        self.bond_time_mapping.reset_parameters()
        self.gnn.reset_parameters()
        self.prediction_head.reset_parameters()

    def calculate_edge_attrs(
        self,
        edge_index: Tensor,
        edge_attr: OptTensor,
        pos: Tensor,
        sqrt: bool = True,
        batch: Tensor = None,
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        if self.ligand_pocket_interaction:
            mask = source != target
            pos[mask] = pos[mask] / torch.norm(pos[mask], dim=1).unsqueeze(1)
            a = pos[target] * pos[source]
        else:
            a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        pos: Tensor,
        edge_index_global: Tensor,
        edge_index_global_lig: OptTensor = None,
        edge_attr_global: OptTensor = None,
        batch: OptTensor = None,
        batch_edge_global: OptTensor = None,
        z: OptTensor = None,
        context: OptTensor = None,
        batch_lig: OptTensor = None,
        pocket_mask: OptTensor = None,
        edge_mask: OptTensor = None,
    ) -> Dict:
        if pocket_mask is None:
            pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]
        # t: (batch_size,)
        ta = self.time_mapping_atom(t)
        tb = self.time_mapping_bond(t)
        tnode = ta[batch]

        # edge_index_global (2, E*)
        tedge_global = tb[batch_edge_global]

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        s = self.atom_mapping(x)
        cemb = None
        if context is not None and self.context_mapping:
            cemb = self.context_mapping(context)
            s = self.atom_context_mapping(s + cemb)
        s = self.atom_time_mapping(s + tnode)

        if self.bond_prediction:
            # symmetric initial edge-feature
            d = (pos[edge_index_global[1]] - pos[edge_index_global[0]]).pow(2).sum(-1, keepdim=True).sqrt()
            edge_attr_global = torch.concat([x[edge_index_global[1]] + x[edge_index_global[0]], d], dim=-1)
        edge_attr_global_transformed = self.bond_mapping(edge_attr_global)
        edge_attr_global_transformed = self.bond_time_mapping(edge_attr_global_transformed + tedge_global)

        # global
        edge_attr_global_transformed = self.calculate_edge_attrs(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global_transformed,
            pos=pos,
            sqrt=True,
            batch=batch if self.ligand_pocket_interaction else None,
        )

        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s,
            v=v,
            p=pos,
            z=z,
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_transformed,
            batch=batch,
            context=cemb,
            batch_lig=batch_lig,
            pocket_mask=pocket_mask,
        )
        h_latent = out["s"]
        out = self.prediction_head(
            x=out,
            batch=batch,
            edge_index_global=edge_index_global,
            edge_index_global_lig=edge_index_global_lig,
            batch_lig=batch_lig,
            pocket_mask=pocket_mask,
            edge_mask=edge_mask,
        )
        out["h_latent"] = h_latent
        return out


if __name__ == "__main__":
    model = DenoisingEdgeNetwork(num_atom_features=5, num_bond_types=3, edge_mp=True, use_cross_product=True)
    t = torch.rand((6, 1))
    x = torch.rand((6, 5))
    pos = torch.rand((6, 3))
    edge_index_global = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]])
    edge_attr_global = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]).float()
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    batch_edge_global = torch.tensor([0, 0, 0, 1, 1, 1])
    out = model(
        t=t,
        x=x,
        pos=pos,
        edge_index_global=edge_index_global,
        batch=batch,
        batch_edge_global=batch_edge_global,
        edge_attr_global=edge_attr_global,
    )
    print(out)