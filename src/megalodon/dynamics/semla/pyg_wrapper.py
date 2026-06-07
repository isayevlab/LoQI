"""Thin wrapper that maps a PyG batch dict → SemlaGenerator sparse forward."""

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from megalodon.dynamics.semla.semla import EquiInvDynamics, SemlaGenerator


class SemlaPyGWrapper(SemlaGenerator):
    """Accepts the same ``(batch, time, …)`` interface as the legacy wrapper
    but runs the full model in sparse PyG mode — no dense-tensor conversion."""

    def __init__(self, args_dict):
        args_dict = OmegaConf.to_container(args_dict, resolve=True)
        dynamics_args = args_dict["dynamics_args"]
        dynamics = EquiInvDynamics(**dynamics_args)
        args_dict["dynamics"] = dynamics
        del args_dict["dynamics_args"]
        self.vocab_size = args_dict["vocab_size"]
        super().__init__(**args_dict)

    def forward(self, batch, time, conditional_batch=None, timesteps=None):
        # Preserve original h_t shape for output padding
        out_h = torch.zeros_like(batch["h_t"])

        # Prepare node features: trim to vocab_size, prepend time
        h_t = batch["h_t"][:, : self.vocab_size].clone()
        h_t = torch.cat((time[batch["batch"]].view(-1, 1), h_t), dim=-1)

        coords     = batch["x_t"]
        edge_feats = batch["edge_attr_t"]
        edge_index = batch["edge_index"]
        batch_idx  = batch["batch"]

        if conditional_batch is None or len(conditional_batch) == 0:
            cond_coords   = torch.zeros_like(coords)
            cond_atomics  = torch.zeros(
                coords.size(0), h_t.size(-1) - 1,
                device=coords.device, dtype=coords.dtype)
            cond_bonds    = torch.zeros_like(edge_feats)
        else:
            cond_coords  = conditional_batch["cond_X"]
            cond_atomics = conditional_batch["cond_H"]
            cond_bonds   = conditional_batch["cond_E"]

        # Self-conditioning (stochastic during training)
        if self.training and torch.rand(1).item() > 0.5:
            with torch.no_grad():
                res = super().forward(
                    coords, h_t, edge_index, batch_idx,
                    edge_feats=edge_feats,
                    cond_coords=cond_coords,
                    cond_atomics=cond_atomics,
                    cond_bonds=cond_bonds,
                )
                cond_coords  = res[0].detach()
                cond_atomics = F.softmax(res[1], dim=-1).detach()
                cond_bonds   = F.softmax(res[2], dim=-1).detach()

        pred_coords, type_logits, edge_logits, charge_logits = super().forward(
            coords, h_t, edge_index, batch_idx,
            edge_feats=edge_feats,
            cond_coords=cond_coords,
            cond_atomics=cond_atomics,
            cond_bonds=cond_bonds,
        )

        h_cat = torch.cat([type_logits, charge_logits], dim=-1)
        out_h[:, : h_cat.size(-1)] = h_cat

        return {
            "cond_X": pred_coords.detach(),
            "cond_H": F.softmax(type_logits, dim=-1).detach(),
            "cond_E": F.softmax(edge_logits, dim=-1).detach(),
            "x_hat": pred_coords,
            "h_logits": out_h,
            "edge_attr_logits": edge_logits,
        }
