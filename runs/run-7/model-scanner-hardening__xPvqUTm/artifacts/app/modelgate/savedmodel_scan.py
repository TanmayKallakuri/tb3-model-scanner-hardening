"""TensorFlow SavedModel scanning.

Parses ``saved_model.pb`` (or ``saved_model.pbtxt``) and, for every meta graph,
inspects every op in the top-level graph and in every library function.  The
closed op table in the threat model (policy.TF_OPS) decides which ops carry a
capability; node names, device strings, attribute values and tensor contents
carry none, so only ``node.op`` is consulted.
"""
from __future__ import annotations

import os

from modelgate import policy
from modelgate.report import Finding


def scan_saved_model_dir(path: str) -> list:
    pb = os.path.join(path, "saved_model.pb")
    pbtxt = os.path.join(path, "saved_model.pbtxt")
    from modelgate.tfproto import saved_model_pb2

    sm = saved_model_pb2.SavedModel()
    try:
        if os.path.isfile(pb):
            if os.path.getsize(pb) > 1024 * 1024 * 1024:
                return [Finding("unsupported", path, "saved_model.pb too large")]
            with open(pb, "rb") as fh:
                sm.ParseFromString(fh.read())
        elif os.path.isfile(pbtxt):
            from google.protobuf import text_format
            with open(pbtxt, "r", encoding="utf-8") as fh:
                text_format.Parse(fh.read(), sm)
        else:
            return [Finding("unsupported", path, "no saved_model.pb or .pbtxt")]
    except Exception as exc:
        return [Finding("unsupported", path, f"cannot parse SavedModel: {exc}")]

    findings = []
    seen = set()
    for mi, mg in enumerate(sm.meta_graphs):
        gd = mg.graph_def
        _scan_nodes(gd.node, f"meta_graphs[{mi}].graph_def", findings, seen)
        for fi, fn in enumerate(gd.library.function):
            fname = fn.signature.name or str(fi)
            _scan_nodes(fn.node_def,
                        f"meta_graphs[{mi}].library.function[{fname}]",
                        findings, seen)
    return findings


def _scan_nodes(nodes, where, findings, seen):
    for node in nodes:
        kind = policy.TF_OPS.get(node.op)
        if kind:
            key = (kind, node.op)
            if key in seen:
                continue
            seen.add(key)
            findings.append(Finding(kind, f"{where}.node[{node.name}]",
                                    f"TF op {node.op}"))
