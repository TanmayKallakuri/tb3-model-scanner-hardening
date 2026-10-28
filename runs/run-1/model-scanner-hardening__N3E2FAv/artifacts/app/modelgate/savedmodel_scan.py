"""Parse SavedModel protobufs without importing TensorFlow or loading a model."""
from __future__ import annotations

import os

from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model(data: bytes, location: str, *, text=False) -> list[Finding]:
    from modelgate.tfproto.saved_model_pb2 import SavedModel

    model = SavedModel()
    if text:
        from google.protobuf import text_format
        text_format.Parse(data.decode('utf-8'), model)
    else:
        model.ParseFromString(data)
    findings = []
    for i, meta in enumerate(model.meta_graphs):
        graph = meta.graph_def
        groups = [(f'{location}:meta_graphs[{i}]/graph', graph.node)]
        # All bodies are in scope, including those only named in func/list.func
        # attributes or not reachable from exported signatures.
        groups.extend((f'{location}:meta_graphs[{i}]/function[{f.signature.name}]',
                       f.node_def) for f in graph.library.function)
        for prefix, nodes in groups:
            for node in nodes:
                kind = TF_OPS.get(node.op)
                if kind:
                    findings.append(Finding(kind, f'{prefix}/node[{node.name}]', f'op {node.op}'))
    return findings


def scan_saved_model_dir(path: str) -> list[Finding]:
    for name in ('saved_model.pb', 'saved_model.pbtxt'):
        filename = os.path.join(path, name)
        if os.path.isfile(filename):
            with open(filename, 'rb') as fh:
                return scan_saved_model(fh.read(), filename, text=name.endswith('pbtxt'))
    return [Finding('unsupported', path, 'directory contains no SavedModel protobuf')]
