"""TensorFlow SavedModel scanning.

The directory holds ``saved_model.pb`` (binary) or ``saved_model.pbtxt``
(text).  The op table in the threat model is the whole policy: an op in the
table has the capability listed, an op not in it has none, wherever it
appears.  Only ``node.op`` is consulted -- node names, attribute values and
tensor contents carry no capability.

Every meta_graph is in scope, and within each: ``graph_def.node`` and every
``graph_def.library.function[*].node_def``.  Scanning the whole function
library is a superset of "functions referenced through node attributes", so
``If``/``While``/``PartitionedCall`` targets are covered automatically.
"""
from __future__ import annotations

import os

from modelgate.policy import TF_OPS
from modelgate.report import Finding

_MAX_PB = 256 * 1024 * 1024


def scan_saved_model_dir(path: str):
    pb = os.path.join(path, "saved_model.pb")
    pbtxt = os.path.join(path, "saved_model.pbtxt")
    if os.path.isfile(pb):
        try:
            with open(pb, "rb") as fh:
                data = fh.read(_MAX_PB + 1)
            if len(data) > _MAX_PB:
                return [Finding("unsupported", "saved_model.pb", "SavedModel proto too large to scan")]
            return _scan_binary(data)
        except Exception as exc:  # noqa: BLE001
            return [Finding("unsupported", "saved_model.pb", f"could not parse SavedModel: {exc}")]
    if os.path.isfile(pbtxt):
        try:
            with open(pbtxt, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            return _scan_text(text)
        except Exception as exc:  # noqa: BLE001
            return [Finding("unsupported", "saved_model.pbtxt", f"could not parse SavedModel: {exc}")]
    return [Finding("unsupported", path, "directory is not a SavedModel (no saved_model.pb[txt])")]


def _load_proto(parse):
    from modelgate.tfproto import saved_model_pb2
    sm = saved_model_pb2.SavedModel()
    parse(sm)
    return sm


def _scan_binary(data):
    sm = _load_proto(lambda m: m.ParseFromString(data))
    return _scan_saved_model(sm)


def _scan_text(text):
    from google.protobuf import text_format
    sm = _load_proto(lambda m: text_format.Parse(text, m))
    return _scan_saved_model(sm)


def _scan_saved_model(sm):
    findings: list[Finding] = []
    for mi, mg in enumerate(sm.meta_graphs):
        graph = mg.graph_def
        for node in graph.node:
            _check_op(node.op, f"meta_graph[{mi}].node[{node.name}]", findings)
        for fi, func in enumerate(graph.library.function):
            fname = func.signature.name or str(fi)
            for node in func.node_def:
                _check_op(node.op, f"meta_graph[{mi}].function[{fname}].node[{node.name}]", findings)
    return findings


def _check_op(op, location, findings):
    kind = TF_OPS.get(op)
    if kind:
        findings.append(Finding(kind, location, f"op {op}"))
