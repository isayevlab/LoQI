"""SEMLA conformer model — predicts coordinates only, echoes atom/edge features."""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from megalodon.scatter import scatter

from megalodon.dynamics.semla.functional import zero_com_pyg
from megalodon.dynamics.semla.semla import EquiInvDynamics


class SemlaConf(nn.Module):
    """PyG-native SEMLA for conformer generation.

    Unlike the full ``SemlaGenerator`` this model predicts only coordinates.
    Atom types and bond types are used as *conditioning* input but are not
    predicted — they stay fixed throughout the diffusion process.
    """

    def __init__(self, d_model, dynamics, atom_classes, edge_classes=None,
                 d_edge=None, size_emb=64, max_atoms=256):
        super().__init__()

        in_feats = atom_classes + 1 + size_emb  # +1 for time

        self.size_emb = nn.Embedding(max_atoms, size_emb)
        self.feat_proj = nn.Sequential(
            nn.Linear(in_feats, d_model),
            nn.SiLU(inplace=False),
            nn.Linear(d_model, d_model),
        )

        self.has_edges = d_edge is not None and edge_classes is not None
        if self.has_edges:
            self.edge_in_proj = nn.Sequential(
                nn.Linear(edge_classes, d_edge),
                nn.SiLU(inplace=False),
                nn.Linear(d_edge, d_edge),
            )

        self.dynamics = dynamics

    def forward(self, coords, node_feats, edge_index, batch, edge_feats=None):
        """
        Args:
            coords:     ``[N, 3]``       noisy coordinates
            node_feats: ``[N, F]``       time (1) + atom features
            edge_index: ``[2, E]``
            batch:      ``[N]``
            edge_feats: ``[E, C]``       bond type features (optional)
        Returns:
            ``[N, 3]``  predicted coordinates (velocity)
        """
        n_atoms = torch.bincount(batch)
        size_emb = self.size_emb(n_atoms)[batch]

        inv_feats = torch.cat((node_feats, size_emb), dim=-1)
        atom_feats = self.feat_proj(inv_feats)

        if edge_feats is not None and self.has_edges:
            edge_feats = self.edge_in_proj(edge_feats.float())

        out = self.dynamics(
            coords, atom_feats, edge_index, batch, edge_feats=edge_feats,
        )
        pred_coords = out if isinstance(out, torch.Tensor) else out[0]

        return zero_com_pyg(pred_coords, batch)


class SemlaConfWrapper(SemlaConf):
    """Wrapper matching the ``(batch, time, …)`` interface of LoQI wrappers.

    Returns ``x_hat`` from the model and echoes ``h_logits`` / ``edge_attr_logits``
    from the input batch (atom/edge types are fixed during conformer generation).
    """

    def __init__(self, args_dict, time_type="continuous", timesteps=None):
        args_dict = OmegaConf.to_container(args_dict, resolve=True) \
            if not isinstance(args_dict, dict) else dict(args_dict)
        self.time_type = time_type
        self.timesteps = timesteps

        dynamics_args = args_dict.pop("dynamics_args")
        dynamics = EquiInvDynamics(**dynamics_args)
        args_dict["dynamics"] = dynamics
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        timesteps = timesteps if timesteps is not None else self.timesteps
        if self.time_type == "discrete" and timesteps is not None:
            time = (timesteps - time.float()) / timesteps

        h_t = batch["h_t"]
        time_feat = time[batch["batch"]].unsqueeze(-1)
        node_feats = torch.cat([time_feat, h_t], dim=-1)

        pred_coords = super().forward(
            coords=batch["x_t"],
            node_feats=node_feats,
            edge_index=batch["edge_index"],
            batch=batch["batch"],
            edge_feats=batch.get("edge_attr_t"),
        )

        return {
            "x_hat": pred_coords,
            "h_logits": batch["h_t"],
            "edge_attr_logits": batch["edge_attr_t"],
        }
