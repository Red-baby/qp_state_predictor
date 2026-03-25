from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Tuple

import pandas as pd


def _make_record(
    frame_type: str,
    temporal_layer: int,
    ref_local_1: int,
    ref_local_2: int,
    num_refs: int,
    valid_train: int = 1,
) -> dict:
    return {
        "frame_type": frame_type,
        "temporal_layer": int(temporal_layer),
        "ref_local_1": int(ref_local_1),
        "ref_local_2": int(ref_local_2),
        "num_refs": int(num_refs),
        "valid_train": int(valid_train),
    }


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


def _build_full_gop(mapping: dict, left: int, right: int) -> None:
    mapping[right] = _make_record("P", 1, left, -1, 1, valid_train=1)
    tmp = {}
    _recursive_b_frames(left, right, layer=2, out=tmp)
    for local_poc, rec in tmp.items():
        rec["valid_train"] = 1
        mapping[local_poc] = rec


def _recursive_tail_layers(left: int, right: int, layer: int, out: dict, leaf_layer: int) -> None:
    if right - left < 4:
        for local_poc in range(left + 1, right):
            out[local_poc] = int(leaf_layer)
        return
    mid = (left + right) // 2
    if mid == left or mid == right:
        for local_poc in range(left + 1, right):
            out[local_poc] = int(leaf_layer)
        return
    out[mid] = int(layer)
    _recursive_tail_layers(left, mid, layer + 1, out, leaf_layer)
    _recursive_tail_layers(mid, right, layer + 1, out, leaf_layer)


def _fill_tail_refs_from_layers(mapping: dict, left_anchor: int, right_anchor: int) -> None:
    for local_poc in range(left_anchor + 1, right_anchor):
        cur = mapping.get(local_poc)
        if not cur or cur["frame_type"] != "B":
            continue
        cur_layer = int(cur["temporal_layer"])
        ref_local_1 = -1
        ref_local_2 = -1
        for cand in range(local_poc - 1, left_anchor - 1, -1):
            if int(mapping[cand]["temporal_layer"]) < cur_layer:
                ref_local_1 = cand
                break
        for cand in range(local_poc + 1, right_anchor + 1):
            if int(mapping[cand]["temporal_layer"]) < cur_layer:
                ref_local_2 = cand
                break
        if ref_local_1 < 0 or ref_local_2 < 0:
            raise RuntimeError(
                f"Failed to build tail references for local_poc={local_poc} "
                f"in shortened GOP [{left_anchor}, {right_anchor}]."
            )
        cur["ref_local_1"] = ref_local_1
        cur["ref_local_2"] = ref_local_2
        cur["num_refs"] = 2


def _build_tail_p_chain(mapping: dict, left_anchor: int, right_anchor: int) -> None:
    prev = left_anchor
    for local_poc in range(left_anchor + 1, right_anchor + 1):
        mapping[local_poc] = _make_record("P", 1, prev, -1, 1, valid_train=1)
        prev = local_poc


def _build_tail_hier_gop(mapping: dict, left_anchor: int, right_anchor: int, leaf_layer: int) -> None:
    mapping[right_anchor] = _make_record("P", 1, left_anchor, -1, 1, valid_train=1)
    layers: dict[int, int] = {}
    _recursive_tail_layers(left_anchor, right_anchor, layer=2, out=layers, leaf_layer=leaf_layer)
    for local_poc, temporal_layer in layers.items():
        mapping[local_poc] = _make_record("B", temporal_layer, -1, -1, 0, valid_train=1)
    _fill_tail_refs_from_layers(mapping, left_anchor, right_anchor)


def build_segment_local_template(
    last_local_poc: int,
    i_interval: int = 125,
    gop_size: int = 16,
    tail_hier_min: int = 4,
    tail_leaf_layer: int = 5,
) -> pd.DataFrame:
    if last_local_poc < 0:
        raise ValueError(f"last_local_poc must be >= 0, got {last_local_poc}")
    if last_local_poc >= i_interval:
        raise ValueError(f"last_local_poc must be < i_interval ({i_interval}), got {last_local_poc}")

    mapping = {0: _make_record("I", 0, -1, -1, 0, valid_train=1)}
    last_full_anchor = (last_local_poc // gop_size) * gop_size

    prev_anchor = 0
    for anchor in range(gop_size, last_full_anchor + 1, gop_size):
        _build_full_gop(mapping, prev_anchor, anchor)
        prev_anchor = anchor

    if last_full_anchor < last_local_poc:
        tail_len = int(last_local_poc - last_full_anchor)
        if tail_len < max(int(tail_hier_min), 2):
            _build_tail_p_chain(mapping, last_full_anchor, last_local_poc)
        else:
            _build_tail_hier_gop(mapping, last_full_anchor, last_local_poc, leaf_layer=tail_leaf_layer)

    records = []
    for local_poc in range(last_local_poc + 1):
        rec = mapping.get(local_poc)
        if rec is None:
            rec = _make_record("B", -1, -1, -1, 0, valid_train=0)
        records.append({"local_poc": local_poc, **rec})
    return pd.DataFrame(records).sort_values("local_poc").reset_index(drop=True)


def build_default_local_template(
    i_interval: int = 125,
    gop_size: int = 16,
    tail_hier_min: int = 4,
    tail_leaf_layer: int = 5,
) -> pd.DataFrame:
    return build_segment_local_template(
        last_local_poc=int(i_interval) - 1,
        i_interval=i_interval,
        gop_size=gop_size,
        tail_hier_min=tail_hier_min,
        tail_leaf_layer=tail_leaf_layer,
    )


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
    if "segment_last_local" in segment_df.columns and not segment_df.empty:
        max_local = int(segment_df["segment_last_local"].max())
    elif not segment_df.empty:
        max_local = int(segment_df["local_poc"].max())
    else:
        max_local = int(i_interval) - 1
    missing_nodes = [node for node in range(max_local + 1) if node not in refs]
    return topo + missing_nodes
