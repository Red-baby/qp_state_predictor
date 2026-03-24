from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Tuple

import pandas as pd


def _recursive_b_frames(left: int, right: int, layer: int, out: dict):
    if right - left <= 1:
        return
    mid = (left + right) // 2
    if mid == left or mid == right:
        return
    out[mid] = {
        "frame_type": "B",
        "temporal_layer": layer,
        "ref_local_1": left,
        "ref_local_2": right,
        "num_refs": 2,
    }
    _recursive_b_frames(left, mid, layer + 1, out)
    _recursive_b_frames(mid, right, layer + 1, out)


def build_default_local_template(i_interval: int = 125, gop_size: int = 16) -> pd.DataFrame:
    records = []
    mapping = {
        0: {
            "frame_type": "I",
            "temporal_layer": 0,
            "ref_local_1": -1,
            "ref_local_2": -1,
            "num_refs": 0,
            "valid_train": 1,
        }
    }

    max_full_anchor = (i_interval // gop_size) * gop_size
    if max_full_anchor == i_interval:
        max_full_anchor -= gop_size

    prev_anchor = 0
    anchor = gop_size
    while anchor <= max_full_anchor:
        mapping[anchor] = {
            "frame_type": "P",
            "temporal_layer": 1,
            "ref_local_1": prev_anchor,
            "ref_local_2": -1,
            "num_refs": 1,
            "valid_train": 1,
        }
        tmp = {}
        # I=0，P=1；B 从 2 起随递归加深递增（与原先 P=0、B 从 1 起相比整体 +1）
        _recursive_b_frames(prev_anchor, anchor, layer=2, out=tmp)
        for k, v in tmp.items():
            v["valid_train"] = 1
            mapping[k] = v
        prev_anchor = anchor
        anchor += gop_size

    for lpoc in range(1, i_interval):
        if lpoc not in mapping:
            mapping[lpoc] = {
                "frame_type": "B",
                "temporal_layer": -1,
                "ref_local_1": -1,
                "ref_local_2": -1,
                "num_refs": 0,
                "valid_train": 0,
            }

    for lpoc in range(i_interval):
        r = mapping[lpoc]
        records.append({"local_poc": lpoc, **r})
    df = pd.DataFrame(records).sort_values("local_poc").reset_index(drop=True)
    return df


def build_topo_order(nodes: List[int], refs: Dict[int, List[int]]) -> List[int]:
    indeg = {n: 0 for n in nodes}
    g = defaultdict(list)
    for n in nodes:
        for r in refs.get(n, []):
            if r not in indeg:
                raise RuntimeError(f"Reference node {r} for node {n} is missing in graph.")
            g[r].append(n)
            indeg[n] += 1

    q = deque(sorted([n for n in nodes if indeg[n] == 0]))
    topo = []
    while q:
        cur = q.popleft()
        topo.append(cur)
        for nxt in sorted(g[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(topo) != len(nodes):
        raise RuntimeError("Reference graph contains cycle or disconnected issue.")
    return topo


def build_graph_from_template(template_df: pd.DataFrame) -> Tuple[Dict[int, List[int]], List[int]]:
    refs = {}
    nodes = template_df["local_poc"].tolist()
    for _, row in template_df.iterrows():
        r = []
        if int(row["ref_local_1"]) >= 0:
            r.append(int(row["ref_local_1"]))
        if int(row["ref_local_2"]) >= 0:
            r.append(int(row["ref_local_2"]))
        refs[int(row["local_poc"])] = r

    topo = build_topo_order(nodes, refs)
    return refs, topo


def build_segment_topo_order(segment_df: pd.DataFrame, i_interval: int) -> List[int]:
    local_to_poc = {int(row["local_poc"]): int(row["poc"]) for _, row in segment_df.iterrows()}
    poc_to_local = {poc: local for local, poc in local_to_poc.items()}

    valid_nodes = sorted(segment_df.loc[segment_df["valid_train"] == 1, "local_poc"].astype(int).unique().tolist())
    nodes = valid_nodes
    refs: Dict[int, List[int]] = {node: [] for node in nodes}
    for _, row in segment_df.iterrows():
        local_poc = int(row["local_poc"])
        if local_poc not in refs:
            continue
        for ref_col in ("ref_poc_1", "ref_poc_2"):
            ref_poc = int(row[ref_col])
            if ref_poc < 0:
                continue
            ref_local = poc_to_local.get(ref_poc)
            if ref_local is not None and ref_local in refs:
                refs[local_poc].append(ref_local)

    topo = build_topo_order(nodes, refs) if nodes else []
    missing_nodes = [node for node in range(int(i_interval)) if node not in refs]
    return topo + missing_nodes
