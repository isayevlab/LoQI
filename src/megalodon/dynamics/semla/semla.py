"""PyG-native SEMLA core: EquiInvDynamics and SemlaGenerator.

PyG-native SEMLA implementations replacing the older dense-tensor path.
All ``nn.Module`` attributes keep identical names and shapes so that
``load_state_dict`` works without remapping.
"""

import copy

import torch
import torch.nn as nn
from torch_scatter import scatter

from megalodon.dynamics.semla.functional import zero_com_pyg, find_reverse_edges
from megalodon.dynamics.semla.modules import (
    BondRefine,
    CoordNorm,
    EquiMessagePassingLayer,
    NodeFeedForward,
)


# ---------------------------------------------------------------------------
# EquiInvDynamics
# ---------------------------------------------------------------------------

class EquiInvDynamics(nn.Module):
    def __init__(self, d_model, d_message, n_coord_sets, n_layers,
                 n_attn_heads=None, d_message_hidden=None, d_edge=None,
                 bond_refine=True, self_cond=False, coord_norm="length",
                 coords_only=False, eps=1e-6):
        super().__init__()

        self.coords_only = coords_only

        if coords_only:
            extra_layers = 1 if d_edge is not None else 0
        else:
            extra_layers = 2 if d_edge is not None else 0
        if extra_layers > n_layers:
            raise ValueError("n_layers is too small.")

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        if d_model % n_attn_heads != 0:
            raise ValueError("n_attn_heads must exactly divide d_model")

        self._hparams = {
            "d_model": d_model, "d_message": d_message,
            "n_coord_sets": n_coord_sets, "n_layers": n_layers,
            "n_attn_heads": n_attn_heads, "d_message_hidden": d_message_hidden,
            "d_edge": d_edge, "bond_refine": bond_refine, "self_cond": self_cond,
            "coord_norm": coord_norm, "coords_only": coords_only, "eps": eps,
        }

        self.d_model = d_model
        self.n_coord_sets = n_coord_sets
        self.d_edge = d_edge
        self.bond_refine = bond_refine and d_edge is not None and not coords_only
        self.self_cond = self_cond

        core = EquiMessagePassingLayer(
            d_model, d_message, n_coord_sets,
            n_attn_heads=n_attn_heads, d_message_hidden=d_message_hidden,
            coord_norm=coord_norm, eps=eps,
        )
        layers = [copy.deepcopy(core) for _ in range(n_layers - extra_layers)]

        if d_edge is not None:
            in_layer = EquiMessagePassingLayer(
                d_model, d_message, n_coord_sets,
                n_attn_heads=n_attn_heads, d_message_hidden=None,
                d_edge_in=d_edge, coord_norm=coord_norm, eps=eps,
            )
            if coords_only:
                layers = [in_layer] + layers
            else:
                out_layer = EquiMessagePassingLayer(
                    d_model, d_message, n_coord_sets,
                    n_attn_heads=n_attn_heads, d_message_hidden=None,
                    d_edge_out=d_edge, coord_norm=coord_norm, eps=eps,
                )
                layers = [in_layer] + layers + [out_layer]

        self.layers = nn.ModuleList(layers)

        if coords_only:
            from megalodon.dynamics.semla.modules import EquivariantMLP
            self.final_node_norm = nn.LayerNorm(d_model)
            self.final_coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)
            self.final_eq_mlp = EquivariantMLP(
                d_model, n_coord_sets, proj_sets=d_message)
        else:
            self.final_ff_block = NodeFeedForward(
                d_model, n_coord_sets, coord_norm=coord_norm)
            self.feat_norm = nn.LayerNorm(d_model)
            if d_edge is not None:
                self.bond_norm = nn.LayerNorm(d_edge)
            if self.bond_refine:
                self.refine_layer = BondRefine(d_model, d_message, d_edge)

        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)

        in_coord_sets = 2 if self_cond else 1
        self.coord_proj = nn.Linear(in_coord_sets, n_coord_sets, bias=False)
        self.coord_head = nn.Linear(n_coord_sets, 1, bias=False)

    @property
    def hparams(self):
        return self._hparams

    def forward(self, coords, inv_feats, edge_index, batch,
                edge_feats=None, cond_coords=None):
        """
        Args:
            coords:      ``[N, 3]``
            inv_feats:   ``[N, D]``
            edge_index:  ``[2, E]``
            batch:       ``[N]``
            edge_feats:  ``[E, d_edge]``  (optional)
            cond_coords: ``[N, 3]``       (optional, self-conditioning)
        Returns:
            If ``coords_only``: ``[N, 3]``
            Otherwise: ``(pred_coords, inv_feats)`` or
            ``(pred_coords, inv_feats, edge_feats)``
        """
        if cond_coords is not None:
            coords = torch.stack((coords, cond_coords), dim=-1)   # [N, 3, 2]
        else:
            coords = coords.unsqueeze(-1)                         # [N, 3, 1]
        coords = self.coord_proj(coords)                          # [N, 3, S]

        for layer in self.layers:
            out = layer(coords, inv_feats, edge_index, batch, edge_feats=edge_feats)
            if len(out) == 2:
                coords, inv_feats = out
                edge_feats = None
            else:
                coords, inv_feats, edge_feats = out

        if self.coords_only:
            h_normed = self.final_node_norm(inv_feats)
            c_normed = self.final_coord_norm(coords, batch)
            coords = self.final_eq_mlp(c_normed, h_normed)

            out_coords = self.coord_norm(coords, batch)
            return self.coord_head(out_coords).squeeze(-1)

        coords, inv_feats = self.final_ff_block(coords, inv_feats, batch)

        out_coords = self.coord_norm(coords, batch)               # [N, 3, S]
        out_coords = self.coord_head(out_coords).squeeze(-1)      # [N, 3]

        if self.bond_refine:
            edge_feats = self.refine_layer(
                out_coords, inv_feats, batch, edge_index, edge_feats)

        inv_feats = self.feat_norm(inv_feats)

        if self.d_edge is None:
            return out_coords, inv_feats

        edge_feats = self.bond_norm(edge_feats)
        return out_coords, inv_feats, edge_feats


# ---------------------------------------------------------------------------
# SemlaGenerator
# ---------------------------------------------------------------------------

class SemlaGenerator(nn.Module):
    """PyG-native SEMLA generator (replaces dense SemlaGenerator)."""

    def __init__(self, d_model, dynamics, vocab_size, n_atom_feats,
                 d_edge=None, n_edge_types=None, self_cond=False,
                 size_emb=64, max_atoms=256):
        super().__init__()

        self._hparams = {
            "d_model": d_model, "vocab_size": vocab_size,
            "n_atom_feats": n_atom_feats, "d_edge": d_edge,
            "n_edge_types": n_edge_types, "self_cond": self_cond,
            "size_emb": size_emb, "max_atoms": max_atoms,
            **dynamics.hparams,
        }

        self.self_cond = self_cond

        if d_edge is not None or n_edge_types is not None:
            if None in (d_edge, n_edge_types):
                raise ValueError("Both d_edge and n_edge_types must be provided.")
            edge_in = n_edge_types * 2 if self_cond else n_edge_types
            self.edge_in_proj = nn.Sequential(
                nn.Linear(edge_in, d_edge), nn.SiLU(inplace=False),
                nn.Linear(d_edge, d_edge),
            )
            self.edge_out_proj = nn.Sequential(
                nn.Linear(d_edge, d_edge), nn.SiLU(inplace=False),
                nn.Linear(d_edge, n_edge_types),
            )

        in_feats = n_atom_feats + vocab_size if self_cond else n_atom_feats
        in_feats += size_emb

        self.size_emb = nn.Embedding(max_atoms, size_emb)
        self.feat_proj = nn.Sequential(
            nn.Linear(in_feats, d_model), nn.SiLU(inplace=False),
            nn.Linear(d_model, d_model),
        )
        self.dynamics = dynamics
        self.atom_classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(inplace=False),
            nn.Linear(d_model, vocab_size),
        )
        self.charge_classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(inplace=False),
            nn.Linear(d_model, 6),
        )

    @property
    def hparams(self):
        return self._hparams

    def forward(self, coords, inv_feats, edge_index, batch,
                edge_feats=None, cond_coords=None,
                cond_atomics=None, cond_bonds=None):
        """
        Args:
            coords:       ``[N, 3]``
            inv_feats:    ``[N, n_atom_feats]``  (time + atom types)
            edge_index:   ``[2, E]``
            batch:        ``[N]``
            edge_feats:   ``[E, n_edge_types]``  (optional)
            cond_coords:  ``[N, 3]``             (optional)
            cond_atomics: ``[N, vocab_size]``     (optional)
            cond_bonds:   ``[E, n_edge_types]``  (optional)
        Returns:
            ``(pred_coords, type_logits, edge_logits, charge_logits)``
        """
        n_atoms = torch.bincount(batch)                            # [B]
        size_emb = self.size_emb(n_atoms)[batch]                   # [N, size_emb]

        inv_feats = torch.cat((inv_feats, size_emb), dim=-1)
        if cond_atomics is not None:
            inv_feats = torch.cat((inv_feats, cond_atomics), dim=-1)

        atom_feats = self.feat_proj(inv_feats)

        if edge_feats is not None:
            edge_feats = edge_feats.float()
            if cond_bonds is not None:
                edge_feats = torch.cat((edge_feats, cond_bonds), dim=-1)
            edge_feats = self.edge_in_proj(edge_feats)

        out = self.dynamics(
            coords, atom_feats, edge_index, batch,
            edge_feats=edge_feats, cond_coords=cond_coords,
        )

        pred_edges = None
        if len(out) == 2:
            pred_coords, pred_feats = out
        else:
            pred_coords, pred_feats, pred_edges = out

        pred_coords = zero_com_pyg(pred_coords, batch)

        type_logits   = self.atom_classifier_head(pred_feats)
        charge_logits = self.charge_classifier_head(pred_feats)

        if pred_edges is not None:
            rev = find_reverse_edges(edge_index)
            pred_edges = pred_edges + pred_edges[rev]
            edge_logits = self.edge_out_proj(pred_edges)
            return pred_coords, type_logits, edge_logits, charge_logits

        return pred_coords, type_logits, charge_logits
