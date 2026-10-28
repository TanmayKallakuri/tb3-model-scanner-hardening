"""TensorFlow SavedModel scanning: parse saved_model.pb and look for file ops."""
from __future__ import annotations

import os

from modelgate.policy import TF_FILE_OPS
from modelgate.report import Finding


def scan_saved_model_dir(path: str) -> list[Finding]:
    pb = os.path.join(path, "saved_model.pb")
    if not os.path.isfile(pb):
        return []
    from modelgate.tfproto import saved_model_pb2  # lazy: protobuf is slow to import

    sm = saved_model_pb2.SavedModel()
    with open(pb, "rb") as fh:
        sm.ParseFromString(fh.read())
    findings: list[Finding] = []
    if not sm.meta_graphs:
        return findings
    graph = sm.meta_graphs[0].graph_def
    for node in graph.node:
        kind = TF_FILE_OPS.get(node.op)
        if kind:
            findings.append(Finding(kind, f"saved_model.pb:node[{node.name}]", f"op {node.op}"))
    return findings
