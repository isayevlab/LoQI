"""PyG-native building blocks for SEMLA.

Every class preserves the original SEMLA constructor shapes so ``load_state_dict`` stays compatible without key or shape remapping.  Only the ``forward`` methods differ – they accept
sparse PyG tensors instead of padded dense batches.

Coordinate convention
---------------------
Dense SEMLA uses ``[B, n_sets, N, 3]``.
Here we use ``[total_N, 3, n_sets]`` which keeps the spatial dimension in
position 1 (matching the mega_pyg convention ``[N, 3, K]``) and lets
``nn.Linear`` act naturally on the last (set) dimension.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.utils import softmax

from megalodon.dynamics.semla.functional import zero_com_pyg


# ---------------------------------------------------------------------------
# CoordNorm
# ---------------------------------------------------------------------------

class CoordNorm(nn.Module):
    def __init__(self, n_coord_sets, norm="length", eps=1e-6):
        super().__init__()
        norm = "none" if norm is None else norm
        if norm not in ("none", "gvp", "length"):
            raise ValueError(f"Unknown normalisation type '{norm}'")
        self.n_coord_sets = n_coord_sets
        self.norm = norm
        self.eps = eps
        # Same shape as dense version for state_dict compatibility
        self.set_weights = nn.Parameter(torch.ones((1, n_coord_sets, 1, 1)))

    def forward(self, coord_sets, batch):
        """
        Args:
            coord_sets: ``[N, 3, n_sets]``
            batch: ``[N]``
        """
        coord_sets = zero_com_pyg(coord_sets, batch)

        if self.norm == "length":
            lengths = torch.linalg.vector_norm(coord_sets, dim=1)  # [N, S]
            sum_len = scatter(lengths, batch, dim=0, reduce="sum")  # [B, S]
            n_atoms = scatter(
                torch.ones(coord_sets.size(0), 1, device=coord_sets.device),
                batch, dim=0, reduce="sum",
            )  # [B, 1]
            avg_len = sum_len / n_atoms  # [B, S]
            coord_div = avg_len[batch].unsqueeze(1) + self.eps  # [N, 1, S]
        elif self.norm == "gvp":
            lengths = torch.linalg.vector_norm(coord_sets, dim=1)  # [N, S]
            coord_div = (lengths.unsqueeze(1) + self.eps) / np.sqrt(self.n_coord_sets)
        else:
            coord_div = 1.0

        weights = self.set_weights.view(1, 1, -1)  # [1, 1, S]
        return (coord_sets * weights) / coord_div


# ---------------------------------------------------------------------------
# EdgeMessages
# ---------------------------------------------------------------------------

class EdgeMessages(nn.Module):
    def __init__(self, d_model, d_message, d_out, n_coord_sets,
                 d_ff=None, d_edge=None, eps=1e-6):
        super().__init__()
        edge_feats = 0 if d_edge is None else d_edge
        d_ff = d_out if d_ff is None else d_ff
        in_feats = (d_message * 2) + n_coord_sets + edge_feats

        self.n_coord_sets = n_coord_sets
        self.d_edge = d_edge
        self.eps = eps

        self.coord_norm = CoordNorm(n_coord_sets, norm="none")
        self.node_norm = nn.LayerNorm(d_model)
        self.edge_norm = nn.LayerNorm(d_edge) if d_edge is not None else None
        self.node_proj = nn.Linear(d_model, d_message)
        self.message_mlp = nn.Sequential(
            nn.Linear(in_feats, d_ff),
            nn.SiLU(inplace=False),
            nn.Linear(d_ff, d_out),
        )

    def forward(self, coords, node_feats, batch, edge_index, edge_feats=None):
        """
        Returns: ``[E, d_out]``
        """
        src, dst = edge_index

        node_feats = self.node_norm(node_feats)
        coords = self.coord_norm(coords, batch)                        # [N, 3, S]

        coord_feats = (coords[dst] * coords[src]).sum(dim=1)           # [E, S]

        node_feats = self.node_proj(node_feats)                        # [N, d_msg]
        node_pairs = torch.cat((node_feats[dst], node_feats[src]), -1) # [E, 2*d_msg]

        inp = torch.cat((node_pairs, coord_feats), dim=-1)
        if edge_feats is not None:
            inp = torch.cat((inp, self.edge_norm(edge_feats)), dim=-1)
        return self.message_mlp(inp)


# ---------------------------------------------------------------------------
# NodeAttention
# ---------------------------------------------------------------------------

class NodeAttention(nn.Module):
    def __init__(self, d_model, n_attn_heads, d_attn=None):
        super().__init__()
        d_attn = d_model if d_attn is None else d_attn
        d_head = d_model // n_attn_heads
        if d_attn % n_attn_heads != 0:
            raise ValueError("n_attn_heads must divide d_model (or d_attn) exactly.")

        self.d_model = d_model
        self.d_attn = d_attn
        self.n_attn_heads = n_attn_heads
        self.d_head = d_head

        self.feat_norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_attn)
        self.out_proj = nn.Linear(d_attn, d_model)

    def forward(self, node_feats, messages, edge_index, batch):
        """
        Args:
            node_feats: ``[N, d_model]``
            messages:   ``[E, n_attn_heads]``  (logits, pre-softmax)
        Returns: ``[N, d_model]``
        """
        src, dst = edge_index
        N = node_feats.size(0)

        attn = softmax(messages, dst, num_nodes=N)                     # [E, H]

        proj = self.in_proj(self.feat_norm(node_feats))                # [N, d_attn]
        heads = proj.view(N, self.n_attn_heads, self.d_head)           # [N, H, D]

        weighted = heads[src] * attn.unsqueeze(-1)                     # [E, H, D]
        out = scatter(weighted, dst, dim=0, reduce="sum", dim_size=N)  # [N, H, D]

        # Variance-preserving weights (GNN-VPA)
        vpa = torch.sqrt(
            scatter(attn ** 2, dst, dim=0, reduce="sum", dim_size=N)
        )                                                               # [N, H]
        out = out * vpa.unsqueeze(-1)

        return self.out_proj(out.reshape(N, self.d_attn))


# ---------------------------------------------------------------------------
# CoordAttention
# ---------------------------------------------------------------------------

class CoordAttention(nn.Module):
    def __init__(self, n_coord_sets, proj_sets=None, coord_norm="length", eps=1e-6):
        super().__init__()
        proj_sets = n_coord_sets if proj_sets is None else proj_sets
        self.eps = eps
        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)
        self.coord_proj = nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.attn_proj = nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, coord_sets, messages, edge_index, batch):
        """
        Args:
            coord_sets: ``[N, 3, S]``
            messages:   ``[E, P]``  (logits, pre-softmax)
        Returns: ``[N, 3, S]``
        """
        src, dst = edge_index
        N = coord_sets.size(0)

        coord_sets = self.coord_norm(coord_sets, batch)
        proj = self.coord_proj(coord_sets)                              # [N, 3, P]

        vec = proj[dst] - proj[src]                                     # [E, 3, P]
        length = torch.linalg.vector_norm(vec, dim=1, keepdim=True)     # [E, 1, P]
        normed = vec / (length + self.eps)                              # [E, 3, P]

        attn = softmax(messages, dst, num_nodes=N)                      # [E, P]

        upd = normed * attn.unsqueeze(1)                                # [E, 3, P]
        agg = scatter(upd, dst, dim=0, reduce="sum", dim_size=N)        # [N, 3, P]

        vpa = torch.sqrt(
            scatter(attn ** 2, dst, dim=0, reduce="sum", dim_size=N)
        )                                                                # [N, P]
        agg = agg * vpa.unsqueeze(1)

        return self.attn_proj(agg)                                       # [N, 3, S]


# ---------------------------------------------------------------------------
# LengthsMLP
# ---------------------------------------------------------------------------

class LengthsMLP(nn.Module):
    def __init__(self, d_model, n_coord_sets, d_ff=None):
        super().__init__()
        d_ff = d_model * 4 if d_ff is None else d_ff
        self.node_ff = nn.Sequential(
            nn.Linear(d_model + n_coord_sets, d_ff),
            nn.SiLU(inplace=False),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, coord_sets, node_feats):
        """coord_sets [N, 3, S], node_feats [N, D] → [N, D]"""
        lengths = torch.linalg.vector_norm(coord_sets, dim=1)  # [N, S]
        return self.node_ff(torch.cat((node_feats, lengths), dim=-1))


# ---------------------------------------------------------------------------
# EquivariantMLP
# ---------------------------------------------------------------------------

class EquivariantMLP(nn.Module):
    def __init__(self, d_model, n_coord_sets, proj_sets=None):
        super().__init__()
        proj_sets = n_coord_sets if proj_sets is None else proj_sets
        self.node_proj = nn.Sequential(
            nn.Linear(d_model, proj_sets),
            nn.SiLU(inplace=False),
            nn.Linear(proj_sets, proj_sets),
        )
        self.coord_proj = nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.attn_proj = nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, coord_sets, node_feats):
        """coord_sets [N, 3, S], node_feats [N, D] → [N, 3, S]"""
        inv = self.node_proj(node_feats)        # [N, P]
        proj = self.coord_proj(coord_sets)      # [N, 3, P]
        # outer-product style equivariant mixing
        att = inv.unsqueeze(1).unsqueeze(-1) * proj.unsqueeze(-2)  # [N, 3, P, P]
        att = att.sum(-1)                                          # [N, 3, P]
        return self.attn_proj(att)                                 # [N, 3, S]


# ---------------------------------------------------------------------------
# NodeFeedForward
# ---------------------------------------------------------------------------

class NodeFeedForward(nn.Module):
    def __init__(self, d_model, n_coord_sets, d_ff=None, proj_sets=None, coord_norm="length"):
        super().__init__()
        self.node_norm = nn.LayerNorm(d_model)
        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)
        self.invariant_mlp = LengthsMLP(d_model, n_coord_sets, d_ff=d_ff)
        self.equivariant_mlp = EquivariantMLP(d_model, n_coord_sets, proj_sets=proj_sets)

    def forward(self, coord_sets, node_feats, batch):
        node_feats = self.node_norm(node_feats)
        coord_sets = self.coord_norm(coord_sets, batch)
        return self.equivariant_mlp(coord_sets, node_feats), \
               self.invariant_mlp(coord_sets, node_feats)


# ---------------------------------------------------------------------------
# BondRefine
# ---------------------------------------------------------------------------

class BondRefine(nn.Module):
    def __init__(self, d_model, d_message, d_edge, d_ff=None):
        super().__init__()
        d_ff = d_message if d_ff is None else d_ff
        in_feats = (2 * d_message) + d_edge + 2

        self.coord_norm = CoordNorm(1, norm="none")
        self.node_norm = nn.LayerNorm(d_model)
        self.edge_norm = nn.LayerNorm(d_edge)
        self.node_proj = nn.Linear(d_model, d_message)
        self.message_mlp = nn.Sequential(
            nn.Linear(in_feats, d_ff),
            nn.SiLU(inplace=False),
            nn.Linear(d_ff, d_edge),
        )

    def forward(self, coords, node_feats, batch, edge_index, edge_feats):
        """
        Args:
            coords: ``[N, 3]``  (single coordinate set, post-projection)
            edge_feats: ``[E, d_edge]``
        Returns: ``[E, d_edge]``
        """
        src, dst = edge_index

        coords = self.coord_norm(coords.unsqueeze(-1), batch).squeeze(-1)  # [N,3]

        diff = coords[dst] - coords[src]                                   # [E,3]
        dists = (diff ** 2).sum(-1, keepdim=True)                          # [E,1]
        dots  = (coords[dst] * coords[src]).sum(-1, keepdim=True)          # [E,1]

        h = self.node_proj(self.node_norm(node_feats))                     # [N,d_msg]
        pairs = torch.cat((h[dst], h[src]), -1)                            # [E,2*d_msg]

        inp = torch.cat((pairs, dists, dots, self.edge_norm(edge_feats)), -1)
        return self.message_mlp(inp)


# ---------------------------------------------------------------------------
# EquiMessagePassingLayer
# ---------------------------------------------------------------------------

class EquiMessagePassingLayer(nn.Module):
    def __init__(self, d_model, d_message, n_coord_sets,
                 n_attn_heads=None, d_message_hidden=None,
                 d_edge_in=None, d_edge_out=None,
                 coord_norm="length", eps=1e-6):
        super().__init__()

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        if d_model % n_attn_heads != 0:
            raise ValueError(
                f"n_attn_heads must exactly divide d_model, got {n_attn_heads} and {d_model}")

        self.d_model = d_model
        self.d_message = d_message
        self.n_coord_sets = n_coord_sets
        self.n_attn_heads = n_attn_heads
        self.d_message_hidden = d_message_hidden
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out
        self.d_coord_message = n_coord_sets
        self.eps = eps

        d_ff = d_model * 4
        d_attn = d_model
        d_msg_out = n_attn_heads + self.d_coord_message
        d_msg_out = d_msg_out + d_edge_out if d_edge_out is not None else d_msg_out

        self.node_ff = NodeFeedForward(
            d_model, n_coord_sets, d_ff=d_ff,
            proj_sets=d_message, coord_norm=coord_norm)
        self.message_ff = EdgeMessages(
            d_model, d_message, d_msg_out, n_coord_sets,
            d_ff=d_message_hidden, d_edge=d_edge_in, eps=eps)
        self.coord_attn = CoordAttention(
            n_coord_sets, self.d_coord_message, coord_norm=coord_norm, eps=eps)
        self.node_attn = NodeAttention(d_model, n_attn_heads, d_attn=d_attn)

    @property
    def hparams(self):
        return {
            "d_model": self.d_model, "d_message": self.d_message,
            "n_coord_sets": self.n_coord_sets,
            "n_attn_heads": self.n_attn_heads,
            "d_message_hidden": self.d_message_hidden,
        }

    def forward(self, coords, node_feats, edge_index, batch, edge_feats=None):
        """
        Args:
            coords:     ``[N, 3, S]``
            node_feats: ``[N, D]``
            edge_index: ``[2, E]``
            batch:      ``[N]``
            edge_feats: ``[E, d_edge_in]``  (optional)
        Returns:
            ``(coords, node_feats)`` or ``(coords, node_feats, edge_out)``
        """
        c_upd, h_upd = self.node_ff(coords, node_feats, batch)
        coords = coords + c_upd
        node_feats = node_feats + h_upd

        msgs = self.message_ff(coords, node_feats, batch, edge_index, edge_feats=edge_feats)
        node_msgs  = msgs[:, :self.n_attn_heads]
        coord_msgs = msgs[:, self.n_attn_heads:self.n_attn_heads + self.d_coord_message]

        node_feats = node_feats + self.node_attn(node_feats, node_msgs, edge_index, batch)
        coords = coords + self.coord_attn(coords, coord_msgs, edge_index, batch)

        if self.d_edge_out is not None:
            edge_out = msgs[:, self.n_attn_heads + self.d_coord_message:]
            if edge_feats is not None:
                edge_out = edge_feats + edge_out
            return coords, node_feats, edge_out

        return coords, node_feats
