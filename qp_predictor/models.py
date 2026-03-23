from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0, num_layers: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, num_layers - 1)):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Phase1Net(nn.Module):
    def __init__(self, self_dim: int, meta_dim: int, cfg: dict):
        super().__init__()
        hid = int(cfg["model"]["head_hidden"])
        dropout = float(cfg["model"]["dropout"])
        in_dim = self_dim + meta_dim + 1
        self.net = MLP(in_dim, hid, 2, dropout=dropout, num_layers=3)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = torch.cat([batch["self_feats"], batch["meta_feats"], batch["qp"]], dim=-1)
        pred = self.net(x)
        return {"pred": pred}


class Phase2Net(nn.Module):
    def __init__(self, self_dim: int, pair_dim: int, meta_dim: int, cfg: dict):
        super().__init__()
        hid = int(cfg["model"]["head_hidden"])
        dropout = float(cfg["model"]["dropout"])
        in_dim = self_dim + meta_dim + 1 + 2 * (self_dim + pair_dim + 1 + 1)
        self.net = MLP(in_dim, hid, 2, dropout=dropout, num_layers=3)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ref_valid = batch["ref_valid_mask"].unsqueeze(-1)
        ref_pack = torch.cat([
            batch["ref_feats"],
            batch["pair_feats"],
            batch["ref_qps"],
            ref_valid,
        ], dim=-1).reshape(batch["self_feats"].shape[0], -1)

        x = torch.cat([batch["self_feats"], batch["meta_feats"], batch["qp"], ref_pack], dim=-1)
        pred = self.net(x)
        return {"pred": pred}


class Phase3Net(nn.Module):
    def __init__(self, self_dim: int, pair_dim: int, meta_dim: int, cfg: dict):
        super().__init__()
        self.state_dim = int(cfg["model"]["state_dim"])
        self.self_hidden = int(cfg["model"]["self_hidden"])
        self.edge_hidden = int(cfg["model"]["edge_hidden"])
        self.state_hidden = int(cfg["model"]["state_hidden"])
        self.head_hidden = int(cfg["model"]["head_hidden"])
        self.dropout = float(cfg["model"]["dropout"])

        self.self_encoder = MLP(
            in_dim=self_dim + meta_dim + 1,
            hidden_dim=self.self_hidden,
            out_dim=self.self_hidden,
            dropout=self.dropout,
            num_layers=3,
        )
        self.edge_encoder = MLP(
            in_dim=(self.self_hidden * 2) + self.state_dim + pair_dim + 1 + 1,
            hidden_dim=self.edge_hidden,
            out_dim=self.edge_hidden,
            dropout=self.dropout,
            num_layers=3,
        )
        self.state_encoder = MLP(
            in_dim=self.self_hidden + self.edge_hidden,
            hidden_dim=self.state_hidden,
            out_dim=self.state_dim,
            dropout=self.dropout,
            num_layers=3,
        )
        self.main_head = MLP(
            in_dim=self.self_hidden + self.edge_hidden + self.state_dim,
            hidden_dim=self.head_hidden,
            out_dim=2,
            dropout=self.dropout,
            num_layers=3,
        )
        self.aux_head = MLP(
            in_dim=self.state_dim,
            hidden_dim=self.state_hidden,
            out_dim=2,
            dropout=self.dropout,
            num_layers=2,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self_feats = batch["self_feats"]
        meta_feats = batch["meta_feats"]
        qps = batch["qps"]
        ref_idx = batch["ref_idx"]
        pair_feats = batch["pair_feats"]

        B, T, _ = self_feats.shape
        device = self_feats.device

        topo_order = batch["topo_order"]
        topo = topo_order[0] if topo_order.dim() == 2 else topo_order

        u = self.self_encoder(torch.cat([self_feats, meta_feats, qps], dim=-1))
        z = torch.zeros(B, T, self.state_dim, device=device, dtype=u.dtype)
        pred = torch.zeros(B, T, 2, device=device, dtype=u.dtype)
        aux = torch.zeros(B, T, 2, device=device, dtype=u.dtype)

        for t_idx in topo.tolist():
            u_t = u[:, t_idx, :]
            q_t = qps[:, t_idx, :]

            edge_embeds = []
            edge_masks = []
            for k in range(2):
                ref_local = ref_idx[:, t_idx, k]
                valid = (ref_local >= 0).float().unsqueeze(-1)

                u_r = torch.zeros_like(u_t)
                z_r = torch.zeros(B, self.state_dim, device=device, dtype=u.dtype)
                q_r = torch.zeros(B, 1, device=device, dtype=u.dtype)

                valid_flat = ref_local >= 0
                if valid_flat.any():
                    b_idx = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
                    r_idx = ref_local[valid_flat].long()
                    u_r[b_idx] = u[b_idx, r_idx, :]
                    z_r[b_idx] = z[b_idx, r_idx, :]
                    q_r[b_idx] = qps[b_idx, r_idx, :]

                pair_k = pair_feats[:, t_idx, k, :]
                edge_in = torch.cat([u_t, u_r, z_r, pair_k, q_t, q_r], dim=-1)
                e = self.edge_encoder(edge_in) * valid
                edge_embeds.append(e)
                edge_masks.append(valid)

            edge_stack = torch.stack(edge_embeds, dim=1)
            mask_stack = torch.stack(edge_masks, dim=1)
            denom = mask_stack.sum(dim=1).clamp_min(1.0)
            c_t = edge_stack.sum(dim=1) / denom

            z_t = self.state_encoder(torch.cat([u_t, c_t], dim=-1))
            main_t = self.main_head(torch.cat([u_t, c_t, z_t], dim=-1))
            aux_t = self.aux_head(z_t)

            z[:, t_idx, :] = z_t
            pred[:, t_idx, :] = main_t
            aux[:, t_idx, :] = aux_t

        return {"pred": pred, "aux_pred": aux, "state": z, "u": u}
