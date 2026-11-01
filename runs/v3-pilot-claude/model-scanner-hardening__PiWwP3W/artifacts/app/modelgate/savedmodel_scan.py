"""TensorFlow SavedModel scanning.

``tf.saved_model.load`` reads ``saved_model.pb`` when it exists and
``saved_model.pbtxt`` otherwise.  Every meta graph is in scope, and within
each the graph's nodes and the nodes of every library function (which is
where functions referenced through node attributes live).  The op table in
``policy.TF_OPS`` is closed: node names, attribute values and tensor
contents carry no capability.
"""
from __future__ import annotations

import os
import re

from modelgate.policy import TF_OPS


def scan_dir(ctx, path, location):
    def read(rel):
        full = os.path.join(path, rel)
        if not os.path.isfile(full):
            return None
        with open(full, "rb") as fh:
            return fh.read()
    scan_savedmodel(ctx, read, location)


def _parse(pb, pbtxt):
    from modelgate.tfproto import saved_model_pb2  # protobuf is only needed here

    sm = saved_model_pb2.SavedModel()
    if pb is not None:
        sm.ParseFromString(pb)
        return sm
    from google.protobuf import text_format
    text_format.Parse(pbtxt.decode("utf-8"), sm, allow_unknown_extension=True, allow_unknown_field=True)
    return sm


def scan_savedmodel(ctx, read, location):
    pb = read("saved_model.pb")
    pbtxt = read("saved_model.pbtxt") if pb is None else None
    if pb is None and pbtxt is None:
        ctx.report("unsupported", location, "directory holds no saved_model.pb or saved_model.pbtxt")
        return
    fname = "saved_model.pb" if pb is not None else "saved_model.pbtxt"
    try:
        sm = _parse(pb, pbtxt)
    except Exception as exc:  # noqa: BLE001 - any parse failure
        if pbtxt is not None:
            _scan_text_fallback(ctx, pbtxt, f"{location}{fname}")
            return
        ctx.report("unsupported", f"{location}{fname}", f"SavedModel protobuf cannot be parsed: {exc}")
        return
    for mi, mg in enumerate(sm.meta_graphs):
        ctx.check_time()
        where = f"{location}{fname}:meta_graphs[{mi}]"
        library = {fn.signature.name for fn in mg.graph_def.library.function}
        for node in mg.graph_def.node:
            _check(ctx, node, f"{where}:node[{node.name}]", library)
        for fn in mg.graph_def.library.function:
            fname_ = fn.signature.name
            for node in fn.node_def:
                _check(ctx, node, f"{where}:function[{fname_}]:node[{node.name}]", library)


# Ops that call the functions named in their attributes.  A name that is not
# a library function is instantiated as a primitive op of that name.
CALLER_OPS = {"If", "StatelessIf", "While", "StatelessWhile", "Case", "StatelessCase",
              "PartitionedCall", "StatefulPartitionedCall", "MapDefun", "For", "RemoteCall",
              "SymbolicGradient"}


def _check(ctx, node, location, library):
    kind = TF_OPS.get(node.op)
    if kind:
        ctx.report(kind, location, f"op {node.op}")
    if node.op in CALLER_OPS:
        for key, attr in node.attr.items():
            names = [attr.func.name] if attr.HasField("func") else [f.name for f in attr.list.func]
            for name in names:
                if name not in library and name in TF_OPS:
                    ctx.report(TF_OPS[name], location, f"attribute {key} instantiates op {name}")


_OP_RE = re.compile(rb'(?<![A-Za-z0-9_])op\s*:\s*"((?:[^"\\]|\\.)*)"')


def _scan_text_fallback(ctx, text, location):
    """Text SavedModel the structured parser rejects: every NodeDef carries ``op: "Name"``."""
    for m in _OP_RE.finditer(text):
        name = m.group(1).decode("utf-8", "replace")
        kind = TF_OPS.get(name)
        if kind:
            ctx.report(kind, location, f"op {name}")
