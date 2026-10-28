"""TensorFlow SavedModel scanning.

The op table in the threat model is closed: a graph node is a finding iff its
``op`` is listed there, wherever it appears.  We parse ``saved_model.pb`` (or
``saved_model.pbtxt``) with vendored protobuf bindings - no TensorFlow, no
execution - and check the op of every node in every meta-graph, including every
function in each meta-graph's function library (where control-flow bodies and
called functions live).  Node names, attributes and tensor contents are never
consulted.
"""
from __future__ import annotations

import os

from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model_dir(path):
    pb = os.path.join(path, "saved_model.pb")
    pbtxt = os.path.join(path, "saved_model.pbtxt")
    try:
        from modelgate.tfproto import saved_model_pb2
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", path, f"cannot load protobuf: {exc}")]

    sm = saved_model_pb2.SavedModel()
    try:
        if os.path.isfile(pb):
            with open(pb, "rb") as fh:
                sm.ParseFromString(fh.read())
        elif os.path.isfile(pbtxt):
            from google.protobuf import text_format
            with open(pbtxt, "r", encoding="utf-8") as fh:
                text_format.Parse(fh.read(), sm)
        else:
            return [Finding("unsupported", path,
                            "directory is not a SavedModel "
                            "(no saved_model.pb or .pbtxt)")]
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", path,
                        f"could not parse SavedModel: {type(exc).__name__}: {exc}")]

    findings = []
    seen = set()
    for mi, mg in enumerate(sm.meta_graphs):
        graph = mg.graph_def
        _scan_nodes(graph.node, f"meta_graph[{mi}]", findings, seen)
        for fn in graph.library.function:
            fname = fn.signature.name
            _scan_nodes(fn.node_def, f"meta_graph[{mi}].library.{fname}",
                        findings, seen)
    return findings


def _scan_nodes(nodes, where, findings, seen):
    for node in nodes:
        kind = TF_OPS.get(node.op)
        if kind is None:
            continue
        key = (kind, where, node.name, node.op)
        if key in seen:
            continue
        seen.add(key)
        findings.append(Finding(
            kind, f"saved_model.pb:{where}:node[{node.name}]", f"op {node.op}"))
