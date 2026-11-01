"""Read TensorFlow protobufs as data, without importing TensorFlow."""
from __future__ import annotations

import os

from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model(data: bytes, location: str, *, text: bool = False) -> list[Finding]:
    from modelgate.tfproto import saved_model_pb2

    sm = saved_model_pb2.SavedModel()
    if text:
        from google.protobuf import text_format
        text_format.Parse(data.decode("utf-8"), sm)
    else:
        sm.ParseFromString(data)
    findings = []
    for i, meta in enumerate(sm.meta_graphs):
        graph = meta.graph_def
        groups = [("graph", graph.node)]
        groups.extend((f"function[{f.signature.name}]", f.node_def) for f in graph.library.function)
        for group, nodes in groups:
            for node in nodes:
                kind = TF_OPS.get(node.op)
                if kind:
                    findings.append(Finding(kind, f"{location}:meta_graph[{i}]/{group}/node[{node.name}]", f"op {node.op}"))
    return findings


def scan_saved_model_dir(path: str) -> list[Finding]:
    for name in ("saved_model.pb", "saved_model.pbtxt"):
        filename = os.path.join(path, name)
        if os.path.isfile(filename):
            with open(filename, "rb") as fh:
                return scan_saved_model(fh.read(), name, text=name.endswith("pbtxt"))
    return [Finding("unsupported", path, "directory contains no SavedModel protobuf")]
