"""TensorFlow SavedModel: every op in every meta graph, its graph nodes and
every function of its library (which holds all functions that nodes reference
through attributes such as If/While/PartitionedCall), against the closed op
table of the policy.  Node names, attribute values and tensors are data."""
from __future__ import annotations

import os
import re

from modelgate.policy import TF_OPS
from modelgate.report import Finding


def _parse(pb: bytes | None, pbtxt: bytes | None):
    from modelgate.tfproto import saved_model_pb2

    sm = saved_model_pb2.SavedModel()
    if pb is not None:
        sm.ParseFromString(pb)
        return sm
    from google.protobuf import text_format

    text = pbtxt.decode("utf-8")
    text_format.Parse(text, sm, allow_unknown_field=True, allow_unknown_extension=True)
    return sm


_OP_RE = re.compile(rb'(?m)^\s*op\s*:\s*"((?:[^"\\]|\\.)*)"')


def scan_saved_model(pb: bytes | None, pbtxt: bytes | None, location: str) -> list:
    findings: list = []
    name = "saved_model.pb" if pb is not None else "saved_model.pbtxt"
    try:
        sm = _parse(pb, pbtxt)
    except Exception as exc:  # noqa: BLE001
        if pbtxt is not None:
            # fall back to the op fields of the text format
            for m in _OP_RE.finditer(pbtxt):
                op = m.group(1).decode("utf-8", "replace")
                kind = TF_OPS.get(op)
                if kind:
                    findings.append(Finding(kind, f"{location}/{name}", f"op {op}"))
            if findings:
                return findings
        findings.append(Finding("unsupported", f"{location}/{name}", f"cannot parse SavedModel: {exc}"))
        return findings
    for mi, mg in enumerate(sm.meta_graphs):
        prefix = f"{location}/{name}:meta_graphs[{mi}]"
        for node in mg.graph_def.node:
            kind = TF_OPS.get(node.op)
            if kind:
                findings.append(Finding(kind, f"{prefix}:node[{node.name}]", f"op {node.op}"))
        for fn in mg.graph_def.library.function:
            for node in fn.node_def:
                kind = TF_OPS.get(node.op)
                if kind:
                    findings.append(Finding(kind, f"{prefix}:function[{fn.signature.name}]:node[{node.name}]",
                                            f"op {node.op}"))
    return findings


def scan_saved_model_dir(path: str, location: str) -> list | None:
    """None when the directory holds no SavedModel."""
    pb_path = os.path.join(path, "saved_model.pb")
    txt_path = os.path.join(path, "saved_model.pbtxt")
    if os.path.isfile(pb_path):
        with open(pb_path, "rb") as fh:
            return scan_saved_model(fh.read(), None, location)
    if os.path.isfile(txt_path):
        with open(txt_path, "rb") as fh:
            return scan_saved_model(None, fh.read(), location)
    return None
