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
    def __init__(self, self_dim: int, meta_dim: int, cfg: dict, pass1_dim: int = 0):
        super().__init__()
        hid = int(cfg["model"]["head_hidden"])
        dropout = float(cfg["model"]["dropout"])
        self.pass1_dim = pass1_dim
        in_dim = self_dim + meta_dim + 1 + pass1_dim
        self.net = MLP(in_dim, hid, 2, dropout=dropout, num_layers=3)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        parts = [batch["self_feats"], batch["meta_feats"], batch["qp"]]
        if self.pass1_dim > 0 and "pass1_feats" in batch:
            parts.append(batch["pass1_feats"])
        x = torch.cat(parts, dim=-1)
        pred = self.net(x)
        return {"pred": pred}


class Phase2Net(nn.Module):
    def __init__(self, self_dim: int, pair_dim: int, meta_dim: int, cfg: dict, pass1_dim: int = 0):
        super().__init__()
        hid = int(cfg["model"]["head_hidden"])
        dropout = float(cfg["model"]["dropout"])
        self.pass1_dim = pass1_dim
        ref_single = self_dim + pair_dim + 1 + 1 + pass1_dim
        in_dim = self_dim + meta_dim + 1 + pass1_dim + 2 * ref_single
        self.net = MLP(in_dim, hid, 2, dropout=dropout, num_layers=3)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ref_valid = batch["ref_valid_mask"].unsqueeze(-1)
        ref_parts = [
            batch["ref_feats"],
            batch["pair_feats"],
            batch["ref_qps"],
            ref_valid,
        ]
        if self.pass1_dim > 0 and "ref_pass1_feats" in batch:
            ref_parts.append(batch["ref_pass1_feats"])
        ref_pack = torch.cat(ref_parts, dim=-1).reshape(batch["self_feats"].shape[0], -1)

        main_parts = [batch["self_feats"], batch["meta_feats"], batch["qp"]]
        if self.pass1_dim > 0 and "pass1_feats" in batch:
            main_parts.append(batch["pass1_feats"])
        main_parts.append(ref_pack)
        x = torch.cat(main_parts, dim=-1)
        pred = self.net(x)
        return {"pred": pred}


class Phase3Net(nn.Module):
    def __init__(self, self_dim: int, pair_dim: int, meta_dim: int, cfg: dict, pass1_dim: int = 0):
        super().__init__()
        self.state_dim = int(cfg["model"]["state_dim"])
        self.self_hidden = int(cfg["model"]["self_hidden"])
        self.edge_hidden = int(cfg["model"]["edge_hidden"])
        self.state_hidden = int(cfg["model"]["state_hidden"])
        self.head_hidden = int(cfg["model"]["head_hidden"])
        self.dropout = float(cfg["model"]["dropout"])
        self.pass1_dim = pass1_dim

        self.self_encoder = MLP(
            in_dim=self_dim + meta_dim + 1 + pass1_dim,
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
        if topo_order.dim() == 2 and topo_order.size(0) > 1:
            first = topo_order[0 : 1].expand_as(topo_order)
            if not torch.equal(topo_order, first):
                raise RuntimeError(
                    "Phase3 batch 内各样本的 topo_order 不一致；当前实现按 batch[0] 的拓扑递推，"
                    "多序列 batch 会算错。请将 train.batch_size_phase3 设为 1，或保证各 segment 拓扑相同。"
                )
        topo = topo_order[0] if topo_order.dim() == 2 else topo_order

        enc_parts = [self_feats, meta_feats, qps]
        if self.pass1_dim > 0 and "pass1_feats" in batch:
            enc_parts.append(batch["pass1_feats"])
        u = self.self_encoder(torch.cat(enc_parts, dim=-1))
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
