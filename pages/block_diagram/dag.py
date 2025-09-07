"""Minimal typed DAG representation and evaluation for block diagrams."""
from __future__ import annotations

from typing import Dict, Any, List

import control
import networkx as nx

GRAPH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["meta", "blocks", "wires", "io"],
    "properties": {
        "meta": {"type": "object"},
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "params"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "params": {"type": "object"},
                },
            },
        },
        "wires": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to"],
                "properties": {
                    "from": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                    "to":   {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                },
            },
        },
        "io": {
            "type": "object",
            "required": ["sources", "sinks"],
            "properties": {
                "sources": {"type": "array", "items": {"type": "string"}},
                "sinks":   {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


def _block_tf(block: Dict[str, Any]) -> control.TransferFunction | None:
    """Return a python-control TransferFunction for a single block.

    Only a small subset of blocks is supported. Non dynamic blocks like
    summing junctions return ``None`` and are handled structurally.
    """
    t = block["type"].lower()
    p = block.get("params", {})
    if t == "tf":
        return control.TransferFunction(p.get("num", [1]), p.get("den", [1]))
    if t == "gain":
        k = float(p.get("k", 1.0))
        return control.TransferFunction([k], [1])
    # structural blocks – Sum etc – return None
    return None


def _path_tf(start: str, end: str, G: nx.DiGraph, tfs: Dict[str, control.TransferFunction | None]) -> control.TransferFunction:
    """Multiply transfer-functions along the directed path ``start`` → ``end``."""
    path = nx.shortest_path(G, start, end)
    sys = control.TransferFunction([1], [1])
    for node in path[1:]:  # skip start; include end if it has TF
        tf = tfs.get(node)
        if tf is not None:
            sys = control.series(sys, tf)
    return sys


def evaluate_tf(graph: Dict[str, Any]) -> Dict[str, List[float]]:
    """Evaluate the overall transfer-function of a block diagram.

    Supports only:
      • pure series chains
      • parallel branches entering a single Sum block
      • a single feedback loop around that Sum block
    """
    blocks = {b["id"]: b for b in graph.get("blocks", [])}
    tfs: Dict[str, control.TransferFunction | None] = {}
    for bid, blk in blocks.items():
        tfs[bid] = _block_tf(blk)
    # IO nodes are treated as unity systems
    for node in graph["io"].get("sources", []) + graph["io"].get("sinks", []):
        tfs[node] = control.TransferFunction([1], [1])

    G = nx.DiGraph()
    for w in graph.get("wires", []):
        G.add_edge(w["from"][0], w["to"][0], to_port=w["to"][1])

    src = graph["io"]["sources"][0]
    dst = graph["io"]["sinks"][0]

    sum_blocks = [b for b in blocks.values() if b["type"].lower() == "sum"]
    if not sum_blocks:
        overall = _path_tf(src, dst, G, tfs)
    else:
        sum_blk = sum_blocks[0]
        sum_id = sum_blk["id"]
        signs = [s.strip() for s in sum_blk.get("params", {}).get("signs", "").split()]

        if nx.has_path(G, dst, sum_id):
            # single-loop feedback
            forward = _path_tf(sum_id, dst, G, tfs)
            feedback = _path_tf(dst, sum_id, G, tfs)
            source_port = feedback_port = None
            sources = set(graph["io"].get("sources", []))
            for w in graph.get("wires", []):
                if w["to"][0] == sum_id:
                    idx = int(w["to"][1][2:])  # "in0" → 0
                    if w["from"][0] in sources:
                        source_port = idx
                    elif nx.has_path(G, dst, w["from"][0]):
                        feedback_port = idx
            pre = _path_tf(src, sum_id, G, tfs)
            sign_src = 1 if signs[source_port] == "+" else -1
            sign_fb = 1 if signs[feedback_port] == "+" else -1
            pre = pre * sign_src
            loop = control.feedback(forward, feedback, sign=sign_fb)
            overall = control.series(pre, loop)
        else:
            # parallel branches into the Sum block
            branches: List[control.TransferFunction] = []
            for w in graph.get("wires", []):
                if w["to"][0] == sum_id:
                    idx = int(w["to"][1][2:])
                    branch = _path_tf(src, w["from"][0], G, tfs)
                    sign = 1 if signs[idx] == "+" else -1
                    branches.append(branch * sign)
            combined = branches[0]
            for b in branches[1:]:
                combined = control.parallel(combined, b)
            post = _path_tf(sum_id, dst, G, tfs)
            overall = control.series(combined, post)

    num, den = control.tfdata(overall)
    num = [float(c) for c in num[0][0]]
    den = [float(c) for c in den[0][0]]
    return {"num": num, "den": den}