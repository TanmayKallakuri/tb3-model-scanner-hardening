"""TensorFlow SavedModel scanning.

Parse ``saved_model.pb`` (or ``saved_model.pbtxt``) statically with the
vendored protobuf bindings -- TensorFlow itself is never imported, so no graph
is ever built or run -- and check the op of every node that is in scope: for
every ``meta_graphs[*]``, the top-level ``graph_def.node`` and every
``graph_def.library.function[*].node_def`` (which is where functions reached
through ``If``/``While``/``PartitionedCall``/... node attributes live).  An op
is judged only by the closed table in ``policy.TF_OPS``; node names, device
strings, attribute values and tensor contents carry no capability.
"""
from __future__ import annotations

import os

from modelgate.policy import TF_OPS
from modelgate.report import Finding


def _scan_nodes(nodes, where, findings):
    for node in nodes:
        kind = TF_OPS.get(node.op)
        if kind:
            findings.append(Finding(kind, f"{where}:{node.name}", f"op {node.op}"))


def _parse(data, text):
    from modelgate.tfproto import saved_model_pb2

    sm = saved_model_pb2.SavedModel()
    if text:
        from google.protobuf import text_format

        text_format.Parse(data.decode("utf-8", "replace"), sm)
    else:
        sm.ParseFromString(data)
    return sm


def _scan_sm(sm, location):
    findings: list[Finding] = []
    for gi, mg in enumerate(sm.meta_graphs):
        graph = mg.graph_def
        _scan_nodes(graph.node, f"{location}:meta_graph[{gi}].graph_def", findings)
        for fi, func in enumerate(graph.library.function):
            name = func.signature.name or str(fi)
            _scan_nodes(func.node_def, f"{location}:meta_graph[{gi}].library.{name}", findings)
    return findings


def scan_saved_model_bytes(data, location, text=False):
    """Scan a serialized SavedModel held in memory (an archive member)."""
    return _scan_sm(_parse(data, text), location)


def scan_saved_model_dir(path):
    pb = os.path.join(path, "saved_model.pb")
    pbtxt = os.path.join(path, "saved_model.pbtxt")
    if os.path.isfile(pb):
        with open(pb, "rb") as fh:
            data, text = fh.read(), False
    elif os.path.isfile(pbtxt):
        with open(pbtxt, "rb") as fh:
            data, text = fh.read(), True
    else:
        return [Finding("unsupported", path, "directory has no saved_model.pb or saved_model.pbtxt")]
    return _scan_sm(_parse(data, text), path)
