"""Parse SavedModel protobufs. No TensorFlow runtime is imported."""
from __future__ import annotations

import os

from modelgate.limits import Budget
from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model(data: bytes, location: str, text=False, budget=None) -> list[Finding]:
    from modelgate.tfproto.saved_model_pb2 import SavedModel

    budget = budget or Budget()
    model = SavedModel()
    if text:
        from google.protobuf import text_format
        text_format.Parse(data.decode("utf-8"), model)
    else:
        model.ParseFromString(data)
    findings = []
    # FunctionDef bodies cover attribute-only calls, cycles, unused functions,
    # and functions referenced by other functions. The policy includes all
    # bodies in every MetaGraph, regardless of tag or signature reachability.
    for i, meta in enumerate(model.meta_graphs):
        graph = meta.graph_def
        groups = [(graph.node, f"{location}:meta_graphs[{i}].graph")]
        groups.extend((f.node_def, f"{location}:meta_graphs[{i}].function[{f.signature.name}]")
                      for f in graph.library.function)
        for nodes, loc in groups:
            for node in nodes:
                budget.check()
                kind = TF_OPS.get(node.op)
                if kind:
                    findings.append(Finding(kind, f"{loc}.node[{node.name}]", f"op {node.op}"))
    return findings


def scan_saved_model_dir(path: str, budget=None) -> list[Finding]:
    budget = budget or Budget()
    for name in ("saved_model.pb", "saved_model.pbtxt"):
        source = os.path.join(path, name)
        if os.path.isfile(source):
            with open(source, "rb") as fh:
                data = budget.read(fh)
            return scan_saved_model(data, source, name.endswith("pbtxt"), budget)
    return [Finding("unsupported", path, "directory has no SavedModel protobuf")]
