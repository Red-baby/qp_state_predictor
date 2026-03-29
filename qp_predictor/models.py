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
    def __init__(self, self_dim: int, meta_dim: int, cfg: dict, pass1_dim: int = 0, out_dim: int = 2):
        super().__init__()
        hid = int(cfg["model"]["head_hidden"])
        dropout = float(cfg["model"]["dropout"])
        self.pass1_dim = pass1_dim
        self.out_dim = int(out_dim)
        in_dim = self_dim + meta_dim + 1 + pass1_dim
        self.net = MLP(in_dim, hid, self.out_dim, dropout=dropout, num_layers=3)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        parts = [batch["self_feats"], batch["meta_feats"], batch["qp"]]
        if self.pass1_dim > 0 and "pass1_feats" in batch:
            parts.append(batch["pass1_feats"])
        x = torch.cat(parts, dim=-1)
        pred = self.net(x)
        return {"pred": pred}


def _zero_last_linear(module: nn.Module) -> None:
    if not hasattr(module, "net"):
        return
    for layer in reversed(list(module.net)):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


class Phase1_1Net(nn.Module):
    """Phase1_1: 共享 backbone + shared base head + TL residual heads。"""

    def __init__(self, self_dim: int, meta_dim: int, cfg: dict, pass1_dim: int = 0, out_dim: int = 2):
        super().__init__()
        hid = int(cfg["model"]["head_hidden"])
        dropout = float(cfg["model"]["dropout"])
        self.pass1_dim = pass1_dim
        self.out_dim = int(out_dim)
        in_dim = self_dim + meta_dim + 1 + pass1_dim

        self.backbone = MLP(in_dim, hid, hid, dropout=dropout, num_layers=3)
        self.base_head = MLP(hid, hid, self.out_dim, dropout=dropout, num_layers=2)
        self.tl0_head = MLP(hid, hid, self.out_dim, dropout=dropout, num_layers=2)
        self.tl1_head = MLP(hid, hid, self.out_dim, dropout=dropout, num_layers=2)
        self.other_head = MLP(hid, hid, self.out_dim, dropout=dropout, num_layers=2)

        # 从原 Phase1 起步，逐步学习 TL 专用修正，避免小样本 TL0 一开始把基线拉坏。
        _zero_last_linear(self.tl0_head)
        _zero_last_linear(self.tl1_head)
        _zero_last_linear(self.other_head)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        parts = [batch["self_feats"], batch["meta_feats"], batch["qp"]]
        if self.pass1_dim > 0 and "pass1_feats" in batch:
            parts.append(batch["pass1_feats"])
        x = torch.cat(parts, dim=-1)
        h = self.backbone(x)

        base = self.base_head(h)
        tl = batch["temporal_layer"].to(dtype=torch.long)

        residual = self.other_head(h)
        residual = torch.where((tl == 0).unsqueeze(-1), self.tl0_head(h), residual)
        residual = torch.where((tl == 1).unsqueeze(-1), self.tl1_head(h), residual)
        pred = base + residual
        return {"pred": pred}


class Phase1_2Net(nn.Module):
    """Phase1_2: 共享 backbone + shared base head + 仅 TL0 的轻量 gated residual。"""

    def __init__(self, self_dim: int, meta_dim: int, cfg: dict, pass1_dim: int = 0, out_dim: int = 2):
        super().__init__()
        hid = int(cfg["model"]["head_hidden"])
        dropout = float(cfg["model"]["dropout"])
        self.pass1_dim = pass1_dim
        self.out_dim = int(out_dim)
        in_dim = self_dim + meta_dim + 1 + pass1_dim

        self.backbone = MLP(in_dim, hid, hid, dropout=dropout, num_layers=3)
        self.base_head = MLP(hid, hid, self.out_dim, dropout=dropout, num_layers=2)
        self.tl0_residual = MLP(hid, hid, self.out_dim, dropout=dropout, num_layers=2)
        self.tl0_gate = MLP(hid, hid, 1, dropout=dropout, num_layers=2)

        # 默认退化成旧 Phase1，只让 TL0 residual 逐步学出来。
        _zero_last_linear(self.tl0_residual)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        parts = [batch["self_feats"], batch["meta_feats"], batch["qp"]]
        if self.pass1_dim > 0 and "pass1_feats" in batch:
            parts.append(batch["pass1_feats"])
        x = torch.cat(parts, dim=-1)
        h = self.backbone(x)

        base = self.base_head(h)
        tl = batch["temporal_layer"].to(dtype=torch.long)
        tl0_mask = (tl == 0).unsqueeze(-1).to(dtype=h.dtype)
        tl0_strength = torch.sigmoid(self.tl0_gate(h))
        residual = self.tl0_residual(h) * tl0_strength * tl0_mask
        pred = base + residual
        return {
            "pred": pred,
            "tl0_strength": tl0_strength,
        }


class Phase2Net(nn.Module):
    def __init__(self, self_dim: int, pair_dim: int, meta_dim: int, cfg: dict, pass1_dim: int = 0, out_dim: int = 2):
        super().__init__()
        hid = int(cfg["model"]["head_hidden"])
        dropout = float(cfg["model"]["dropout"])
        self.pass1_dim = pass1_dim
        self.out_dim = int(out_dim)
        ref_single = self_dim + pair_dim + 1 + 1 + pass1_dim
        in_dim = self_dim + meta_dim + 1 + pass1_dim + 2 * ref_single
        self.net = MLP(in_dim, hid, self.out_dim, dropout=dropout, num_layers=3)

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


class Phase2_1Net(nn.Module):
    def __init__(self, self_dim: int, pair_dim: int, meta_dim: int, cfg: dict, pass1_dim: int = 0, out_dim: int = 2):
        super().__init__()
        hid = int(cfg["model"]["head_hidden"])
        edge_hid = int(cfg["model"]["edge_hidden"])
        dropout = float(cfg["model"]["dropout"])
        self.pass1_dim = pass1_dim
        self.out_dim = int(out_dim)

        cur_in = self_dim + meta_dim + 1 + pass1_dim
        ref_in = self_dim + 1 + pass1_dim
        edge_in = (hid * 3) + pair_dim + 1 + 1 + pass1_dim

        self.current_encoder = MLP(cur_in, hid, hid, dropout=dropout, num_layers=3)
        self.ref_encoder = MLP(ref_in, hid, hid, dropout=dropout, num_layers=3)
        self.edge_encoder = MLP(edge_in, edge_hid, edge_hid, dropout=dropout, num_layers=3)
        self.edge_gate = nn.Linear(edge_hid, 1)
        self.trunk = MLP(hid + edge_hid, hid, hid, dropout=dropout, num_layers=3)

        if self.out_dim == 1:
            self.single_head = MLP(hid, hid, 1, dropout=dropout, num_layers=2)
        else:
            self.bits_head = MLP(hid, hid, 1, dropout=dropout, num_layers=2)
            self.distortion_head = MLP(hid, hid, 1, dropout=dropout, num_layers=2)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self_feats = batch["self_feats"]
        meta_feats = batch["meta_feats"]
        qp = batch["qp"]
        ref_feats = batch["ref_feats"]
        pair_feats = batch["pair_feats"]
        ref_qps = batch["ref_qps"]
        ref_valid = batch["ref_valid_mask"].unsqueeze(-1)

        B = self_feats.shape[0]
        ref_count = ref_feats.shape[1]

        cur_parts = [self_feats, meta_feats, qp]
        cur_pass1 = None
        if self.pass1_dim > 0:
            if "pass1_feats" in batch:
                cur_pass1 = batch["pass1_feats"]
            else:
                cur_pass1 = torch.zeros(B, self.pass1_dim, device=self_feats.device, dtype=self_feats.dtype)
            cur_parts.append(cur_pass1)
        u_t = self.current_encoder(torch.cat(cur_parts, dim=-1))

        ref_parts = [ref_feats, ref_qps]
        ref_pass1 = None
        if self.pass1_dim > 0:
            if "ref_pass1_feats" in batch:
                ref_pass1 = batch["ref_pass1_feats"]
            else:
                ref_pass1 = torch.zeros(
                    B, ref_count, self.pass1_dim, device=self_feats.device, dtype=self_feats.dtype
                )
            ref_parts.append(ref_pass1)
        u_r = self.ref_encoder(torch.cat(ref_parts, dim=-1))

        u_t_rep = u_t.unsqueeze(1).expand(-1, ref_count, -1)
        edge_parts = [
            u_t_rep,
            u_r,
            torch.abs(u_t_rep - u_r),
            pair_feats,
            qp.unsqueeze(1) - ref_qps,
        ]
        if self.pass1_dim > 0 and cur_pass1 is not None and ref_pass1 is not None:
            edge_parts.append(cur_pass1.unsqueeze(1).expand(-1, ref_count, -1) - ref_pass1)
        edge_parts.append(ref_valid)

        edge_embed = self.edge_encoder(torch.cat(edge_parts, dim=-1)) * ref_valid
        gate_logits = self.edge_gate(edge_embed)
        # AMP 下 logits 可能为 float16，-1e9 无法表示；用 dtype 可表示的最小值做 softmax mask
        gate_logits = gate_logits.masked_fill(ref_valid <= 0, torch.finfo(gate_logits.dtype).min)
        edge_weights = torch.softmax(gate_logits, dim=1) * ref_valid
        edge_weights = edge_weights / edge_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        context = (edge_weights * edge_embed).sum(dim=1)

        h = self.trunk(torch.cat([u_t, context], dim=-1))
        if self.out_dim == 1:
            pred = self.single_head(h)
        else:
            pred = torch.cat([self.bits_head(h), self.distortion_head(h)], dim=-1)
        return {"pred": pred, "ref_weights": edge_weights.squeeze(-1), "context": context, "u_t": u_t}


class Phase2_2Branch(nn.Module):
    """面向单任务的 Phase2 分支。

    bits 分支强调「当前帧主导、参考做残差校正」；
    distortion/vmaf 分支强调「结构化参考交互」。
    """

    def __init__(
        self,
        *,
        self_dim: int,
        pair_dim: int,
        meta_dim: int,
        hidden_dim: int,
        edge_hidden: int,
        pass1_dim: int,
        dropout: float,
        gate_mode: str,
        use_base_residual: bool,
        aux_out_dim: int = 1,
    ):
        super().__init__()
        self.pass1_dim = int(pass1_dim)
        self.gate_mode = str(gate_mode).lower().strip()
        self.use_base_residual = bool(use_base_residual)

        cur_in = self_dim + meta_dim + 1 + self.pass1_dim
        ref_in = self_dim + 1 + self.pass1_dim
        edge_in = (hidden_dim * 3) + pair_dim + 1 + 1 + self.pass1_dim

        self.current_encoder = MLP(cur_in, hidden_dim, hidden_dim, dropout=dropout, num_layers=3)
        self.ref_encoder = MLP(ref_in, hidden_dim, hidden_dim, dropout=dropout, num_layers=3)
        self.edge_encoder = MLP(edge_in, edge_hidden, edge_hidden, dropout=dropout, num_layers=3)
        self.edge_gate = nn.Linear(edge_hidden, 1)
        self.context_gate = MLP(hidden_dim + edge_hidden, hidden_dim, 1, dropout=dropout, num_layers=2)
        self.trunk = MLP(hidden_dim + edge_hidden, hidden_dim, hidden_dim, dropout=dropout, num_layers=3)

        if self.use_base_residual:
            self.base_head = MLP(hidden_dim, hidden_dim, 1, dropout=dropout, num_layers=2)
            self.residual_head = MLP(hidden_dim, hidden_dim, 1, dropout=dropout, num_layers=2)
        else:
            self.main_head = MLP(hidden_dim, hidden_dim, 1, dropout=dropout, num_layers=2)
        self.aux_head = MLP(hidden_dim, hidden_dim, aux_out_dim, dropout=dropout, num_layers=2) if aux_out_dim > 0 else None

    def forward(
        self,
        *,
        self_feats: torch.Tensor,
        meta_feats: torch.Tensor,
        qp: torch.Tensor,
        ref_feats: torch.Tensor,
        pair_feats: torch.Tensor,
        ref_qps: torch.Tensor,
        ref_valid: torch.Tensor,
        cur_pass1: torch.Tensor | None,
        ref_pass1: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        ref_count = ref_feats.shape[1]

        cur_parts = [self_feats, meta_feats, qp]
        if self.pass1_dim > 0 and cur_pass1 is not None:
            cur_parts.append(cur_pass1)
        u_t = self.current_encoder(torch.cat(cur_parts, dim=-1))

        ref_parts = [ref_feats, ref_qps]
        if self.pass1_dim > 0 and ref_pass1 is not None:
            ref_parts.append(ref_pass1)
        u_r = self.ref_encoder(torch.cat(ref_parts, dim=-1))

        u_t_rep = u_t.unsqueeze(1).expand(-1, ref_count, -1)
        edge_parts = [
            u_t_rep,
            u_r,
            torch.abs(u_t_rep - u_r),
            pair_feats,
            qp.unsqueeze(1) - ref_qps,
        ]
        if self.pass1_dim > 0 and cur_pass1 is not None and ref_pass1 is not None:
            edge_parts.append(cur_pass1.unsqueeze(1).expand(-1, ref_count, -1) - ref_pass1)
        edge_parts.append(ref_valid)

        edge_embed = self.edge_encoder(torch.cat(edge_parts, dim=-1)) * ref_valid
        if self.gate_mode == "sigmoid":
            edge_weights = torch.sigmoid(self.edge_gate(edge_embed)) * ref_valid
            denom = edge_weights.sum(dim=1).clamp_min(1.0)
            context_raw = (edge_weights * edge_embed).sum(dim=1) / denom
        else:
            gate_logits = self.edge_gate(edge_embed)
            gate_logits = gate_logits.masked_fill(ref_valid <= 0, torch.finfo(gate_logits.dtype).min)
            edge_weights = torch.softmax(gate_logits, dim=1) * ref_valid
            edge_weights = edge_weights / edge_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            context_raw = (edge_weights * edge_embed).sum(dim=1)

        context_strength = torch.sigmoid(self.context_gate(torch.cat([u_t, context_raw], dim=-1)))
        context = context_raw * context_strength
        h = self.trunk(torch.cat([u_t, context], dim=-1))

        if self.use_base_residual:
            main = self.base_head(u_t) + self.residual_head(h)
        else:
            main = self.main_head(h)
        aux = self.aux_head(h) if self.aux_head is not None else None
        return {
            "main": main,
            "aux": aux,
            "u_t": u_t,
            "context": context,
            "context_strength": context_strength,
            "ref_weights": edge_weights.squeeze(-1),
        }


class Phase2_2Net(nn.Module):
    """Phase2_2: bits / distortion 拆分归纳偏置，并保留弱辅助任务做正则。"""

    def __init__(self, self_dim: int, pair_dim: int, meta_dim: int, cfg: dict, pass1_dim: int = 0, out_dim: int = 2):
        super().__init__()
        hid = int(cfg["model"]["head_hidden"])
        edge_hid = int(cfg["model"]["edge_hidden"])
        dropout = float(cfg["model"]["dropout"])
        self.pass1_dim = pass1_dim
        self.out_dim = int(out_dim)
        self.mode = str(cfg.get("model", {}).get("mode", "single")).lower().strip()
        self.double_target = str(cfg.get("model", {}).get("double_target", "bits")).lower().strip()

        self.bits_branch = Phase2_2Branch(
            self_dim=self_dim,
            pair_dim=pair_dim,
            meta_dim=meta_dim,
            hidden_dim=hid,
            edge_hidden=edge_hid,
            pass1_dim=pass1_dim,
            dropout=dropout,
            gate_mode="sigmoid",
            use_base_residual=True,
            aux_out_dim=1,
        )
        self.dist_branch = Phase2_2Branch(
            self_dim=self_dim,
            pair_dim=pair_dim,
            meta_dim=meta_dim,
            hidden_dim=hid,
            edge_hidden=edge_hid,
            pass1_dim=pass1_dim,
            dropout=dropout,
            gate_mode="softmax",
            use_base_residual=False,
            aux_out_dim=1,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self_feats = batch["self_feats"]
        meta_feats = batch["meta_feats"]
        qp = batch["qp"]
        ref_feats = batch["ref_feats"]
        pair_feats = batch["pair_feats"]
        ref_qps = batch["ref_qps"]
        ref_valid = batch["ref_valid_mask"].unsqueeze(-1)

        B = self_feats.shape[0]
        ref_count = ref_feats.shape[1]
        cur_pass1 = None
        ref_pass1 = None
        if self.pass1_dim > 0:
            cur_pass1 = batch.get("pass1_feats")
            if cur_pass1 is None:
                cur_pass1 = torch.zeros(B, self.pass1_dim, device=self_feats.device, dtype=self_feats.dtype)
            ref_pass1 = batch.get("ref_pass1_feats")
            if ref_pass1 is None:
                ref_pass1 = torch.zeros(B, ref_count, self.pass1_dim, device=self_feats.device, dtype=self_feats.dtype)

        if self.out_dim == 1 and self.mode == "double" and self.double_target == "bits":
            bits_out = self.bits_branch(
                self_feats=self_feats,
                meta_feats=meta_feats,
                qp=qp,
                ref_feats=ref_feats,
                pair_feats=pair_feats,
                ref_qps=ref_qps,
                ref_valid=ref_valid,
                cur_pass1=cur_pass1,
                ref_pass1=ref_pass1,
            )
            return {
                "pred": bits_out["main"],
                "aux_dist": bits_out["aux"],
                "ref_weights_bits": bits_out["ref_weights"],
                "bits_context": bits_out["context"],
                "bits_context_strength": bits_out["context_strength"],
                "bits_u_t": bits_out["u_t"],
            }

        if self.out_dim == 1:
            dist_out = self.dist_branch(
                self_feats=self_feats,
                meta_feats=meta_feats,
                qp=qp,
                ref_feats=ref_feats,
                pair_feats=pair_feats,
                ref_qps=ref_qps,
                ref_valid=ref_valid,
                cur_pass1=cur_pass1,
                ref_pass1=ref_pass1,
            )
            return {
                "pred": dist_out["main"],
                "aux_bits": dist_out["aux"],
                "ref_weights_dist": dist_out["ref_weights"],
                "dist_context": dist_out["context"],
                "dist_context_strength": dist_out["context_strength"],
                "dist_u_t": dist_out["u_t"],
            }

        bits_out = self.bits_branch(
            self_feats=self_feats,
            meta_feats=meta_feats,
            qp=qp,
            ref_feats=ref_feats,
            pair_feats=pair_feats,
            ref_qps=ref_qps,
            ref_valid=ref_valid,
            cur_pass1=cur_pass1,
            ref_pass1=ref_pass1,
        )
        dist_out = self.dist_branch(
            self_feats=self_feats,
            meta_feats=meta_feats,
            qp=qp,
            ref_feats=ref_feats,
            pair_feats=pair_feats,
            ref_qps=ref_qps,
            ref_valid=ref_valid,
            cur_pass1=cur_pass1,
            ref_pass1=ref_pass1,
        )
        pred = torch.cat([bits_out["main"], dist_out["main"]], dim=-1)
        return {
            "pred": pred,
            "aux_dist_from_bits": bits_out["aux"],
            "aux_bits_from_dist": dist_out["aux"],
            "ref_weights_bits": bits_out["ref_weights"],
            "ref_weights_dist": dist_out["ref_weights"],
            "bits_context": bits_out["context"],
            "dist_context": dist_out["context"],
            "bits_context_strength": bits_out["context_strength"],
            "dist_context_strength": dist_out["context_strength"],
            "bits_u_t": bits_out["u_t"],
            "dist_u_t": dist_out["u_t"],
        }


class Phase3Net(nn.Module):
    def __init__(self, self_dim: int, pair_dim: int, meta_dim: int, cfg: dict, pass1_dim: int = 0, head_out_dim: int = 2):
        super().__init__()
        self.state_dim = int(cfg["model"]["state_dim"])
        self.self_hidden = int(cfg["model"]["self_hidden"])
        self.edge_hidden = int(cfg["model"]["edge_hidden"])
        self.state_hidden = int(cfg["model"]["state_hidden"])
        self.head_hidden = int(cfg["model"]["head_hidden"])
        self.dropout = float(cfg["model"]["dropout"])
        self.pass1_dim = pass1_dim
        self.head_out_dim = int(head_out_dim)

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
            out_dim=self.head_out_dim,
            dropout=self.dropout,
            num_layers=3,
        )
        self.aux_head = MLP(
            in_dim=self.state_dim,
            hidden_dim=self.state_hidden,
            out_dim=self.head_out_dim,
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
        pred = torch.zeros(B, T, self.head_out_dim, device=device, dtype=u.dtype)
        aux = torch.zeros(B, T, self.head_out_dim, device=device, dtype=u.dtype)

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
                # qps 来自 DataLoader 常为 float32；AMP 下 u 为 float16，勿用 u.dtype 建 q_r，否则 index_put 与 qps 切片 dtype 不一致
                q_r = torch.zeros(B, 1, device=device, dtype=qps.dtype)

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
