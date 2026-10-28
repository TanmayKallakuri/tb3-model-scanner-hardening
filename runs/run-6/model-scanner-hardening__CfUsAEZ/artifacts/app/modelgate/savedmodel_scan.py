"""Read the SavedModel schema without importing TensorFlow or executing a graph."""
from __future__ import annotations

import os

from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model(data: bytes, location: str, *, text=False) -> list[Finding]:
    from modelgate.tfproto.saved_model_pb2 import SavedModel

    model = SavedModel()
    if text:
        from google.protobuf import text_format
        text_format.Parse(data.decode("utf-8"), model)
    else:
        model.ParseFromString(data)
    findings = []
    for i, meta in enumerate(model.meta_graphs):
        graph = meta.graph_def
        groups = [(f"meta_graphs[{i}]/graph", graph.node)]
        # The policy includes every library body, including bodies reached only
        # through NameAttrList attributes and bodies in non-serving metagraphs.
        groups.extend((f"meta_graphs[{i}]/function[{f.signature.name}]", f.node_def)
                      for f in graph.library.function)
        for prefix, nodes in groups:
            for j, node in enumerate(nodes):
                kind = TF_OPS.get(node.op)
                if kind:
                    findings.append(Finding(kind, f"{location}:{prefix}/node[{j}:{node.name}]", f"op {node.op}"))
    return findings


def scan_saved_model_dir(path: str) -> list[Finding]:
    for filename in ("saved_model.pb", "saved_model.pbtxt"):
        pb = os.path.join(path, filename)
        if os.path.isfile(pb):
            with open(pb, "rb") as fh:
                return scan_saved_model(fh.read(), pb, text=filename.endswith("pbtxt"))
    return [Finding("unsupported", path, "directory has no saved_model.pb or saved_model.pbtxt")]
