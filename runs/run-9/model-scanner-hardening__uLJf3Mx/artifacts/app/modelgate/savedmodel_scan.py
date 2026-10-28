"""Parse SavedModel messages, never import TensorFlow or execute a graph."""
from __future__ import annotations

import os

from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model(data: bytes, location: str, *, text: bool = False) -> list[Finding]:
    from modelgate.tfproto.saved_model_pb2 import SavedModel

    sm = SavedModel()
    if text:
        from google.protobuf import text_format
        text_format.Parse(data.decode("utf-8"), sm)
    else:
        sm.ParseFromString(data)
    findings = []
    # All function definitions are in scope, including those reached only via
    # NameAttrList attributes. Scanning the library avoids reachability guesses.
    for mi, meta in enumerate(sm.meta_graphs):
        groups = [("graph", meta.graph_def.node)]
        groups.extend((f"function[{f.signature.name}]", f.node_def)
                      for f in meta.graph_def.library.function)
        for scope, nodes in groups:
            for ni, node in enumerate(nodes):
                kind = TF_OPS.get(node.op)
                if kind:
                    findings.append(Finding(kind,
                        f"{location}:meta_graphs[{mi}]/{scope}/node[{ni}:{node.name}]",
                        f"op {node.op}"))
    return findings


def scan_saved_model_dir(path: str) -> list[Finding]:
    for name in ("saved_model.pb", "saved_model.pbtxt"):
        pb = os.path.join(path, name)
        if os.path.isfile(pb):
            with open(pb, "rb") as fh:
                return scan_saved_model(fh.read(), pb, text=name.endswith("pbtxt"))
    return [Finding("unsupported", path, "directory has no SavedModel message")]
