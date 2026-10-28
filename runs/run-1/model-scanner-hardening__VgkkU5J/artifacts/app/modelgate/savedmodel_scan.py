"""TensorFlow SavedModel scanning.

Parse ``saved_model.pb`` (or ``saved_model.pbtxt``) and, for every meta graph,
inspect every node of the main ``graph_def`` and of every library function.
An op is a finding exactly when it is in the closed capability table
(threat-model section 4); node names, attributes and tensor contents are data.
"""
from __future__ import annotations

import os

from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model_dir(path: str) -> list[Finding]:
    pb = os.path.join(path, "saved_model.pb")
    pbtxt = os.path.join(path, "saved_model.pbtxt")
    if os.path.isfile(pb):
        with open(pb, "rb") as fh:
            return scan_saved_model_bytes(fh.read(), "saved_model.pb")
    if os.path.isfile(pbtxt):
        with open(pbtxt, "rb") as fh:
            return scan_saved_model_bytes(fh.read(), "saved_model.pbtxt", text=True)
    return [Finding("unsupported", path,
                    "directory has no saved_model.pb or saved_model.pbtxt")]


def scan_saved_model_bytes(data: bytes, location: str, text: bool = False) -> list[Finding]:
    from modelgate.tfproto import saved_model_pb2
    sm = saved_model_pb2.SavedModel()
    try:
        if text:
            from google.protobuf import text_format
            text_format.Parse(data.decode("utf-8", "replace"), sm)
        else:
            sm.ParseFromString(data)
    except Exception as exc:
        return [Finding("unsupported", location, f"could not parse SavedModel: {exc}")]

    findings: list[Finding] = []
    seen = set()
    for mi, mg in enumerate(sm.meta_graphs):
        g = mg.graph_def
        for node in g.node:
            _check(node, f"{location}:meta[{mi}]", seen, findings)
        for fn in g.library.function:
            fname = fn.signature.name
            for node in fn.node_def:
                _check(node, f"{location}:meta[{mi}]/func[{fname}]", seen, findings)
    return findings


def _check(node, where, seen, findings):
    kind = TF_OPS.get(node.op)
    if not kind:
        return
    key = (where, node.op, node.name)
    if key in seen:
        return
    seen.add(key)
    findings.append(Finding(kind, f"{where}/node[{node.name}]", f"op {node.op}"))
