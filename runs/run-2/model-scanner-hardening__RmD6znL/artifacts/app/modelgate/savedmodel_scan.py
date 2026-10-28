"""TensorFlow SavedModel scanning against the closed op table (section 4).

Every meta-graph is in scope, and within each one every node of the main
graph and every node of every library function (which is where the bodies of
If / While / Case / PartitionedCall and friends live).  An op's capability is
decided solely by its ``op`` type against the table; node names, device
strings, attribute values and tensor contents carry none.  We parse the
protobuf (binary ``saved_model.pb`` or text ``saved_model.pbtxt``) with the
vendored descriptors; we never load the graph.
"""
from __future__ import annotations

import os

from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model_dir(path: str) -> list[Finding]:
    pb = os.path.join(path, "saved_model.pb")
    pbtxt = os.path.join(path, "saved_model.pbtxt")
    try:
        from modelgate.tfproto import saved_model_pb2
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", path, f"protobuf unavailable: {exc}")]

    _MAX = 256 * 1024 * 1024
    sm = saved_model_pb2.SavedModel()
    try:
        if os.path.isfile(pb):
            if os.path.getsize(pb) > _MAX:
                return [Finding("unsupported", path, "saved_model.pb too large")]
            with open(pb, "rb") as fh:
                sm.ParseFromString(fh.read())
        elif os.path.isfile(pbtxt):
            if os.path.getsize(pbtxt) > _MAX:
                return [Finding("unsupported", path, "saved_model.pbtxt too large")]
            from google.protobuf import text_format
            with open(pbtxt, "r", encoding="utf-8") as fh:
                text_format.Parse(fh.read(), sm)
        else:
            return [Finding("unsupported", path,
                            "directory has no saved_model.pb or saved_model.pbtxt")]
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", path, f"could not parse SavedModel: {type(exc).__name__}")]

    findings: list[Finding] = []
    for gi, mg in enumerate(sm.meta_graphs):
        graph = mg.graph_def
        for node in graph.node:
            _check(node.op, node.name, f"saved_model.pb:meta_graph[{gi}]/node", findings)
        for fn in graph.library.function:
            fname = fn.signature.name
            for node in fn.node_def:
                _check(node.op, node.name,
                       f"saved_model.pb:meta_graph[{gi}]/library/{fname}/node", findings)
    return findings


def _check(op, node_name, where, findings):
    kind = TF_OPS.get(op)
    if kind:
        findings.append(Finding(kind, f"{where}[{node_name}]", f"op {op}"))
