"""Read SavedModel protobufs without importing TensorFlow or loading a model."""
from __future__ import annotations
import os
from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model(data: bytes, location: str, *, text: bool = False) -> list[Finding]:
    from modelgate.tfproto import saved_model_pb2
    model = saved_model_pb2.SavedModel()
    if text:
        from google.protobuf import text_format
        text_format.Parse(data.decode('utf-8'), model)
    else:
        model.ParseFromString(data)
    findings = []
    for mi, meta in enumerate(model.meta_graphs):
        graph = meta.graph_def
        groups = [(f'meta_graphs[{mi}].graph', graph.node)]
        # The whole library is in scope, including functions used only by
        # branch/loop/call node attributes.
        groups.extend((f'meta_graphs[{mi}].function[{f.signature.name}]', f.node_def)
                      for f in graph.library.function)
        for group, nodes in groups:
            for ni, node in enumerate(nodes):
                kind = TF_OPS.get(node.op)
                if kind:
                    findings.append(Finding(kind, f'{location}:{group}.node[{ni}:{node.name}]',
                                            f'op {node.op}'))
    return findings


def scan_saved_model_dir(path: str) -> list[Finding]:
    for name in ('saved_model.pb', 'saved_model.pbtxt'):
        pb = os.path.join(path, name)
        if os.path.isfile(pb):
            with open(pb, 'rb') as fh:
                return scan_saved_model(fh.read(), pb, text=name.endswith('pbtxt'))
    return [Finding('unsupported', path, 'directory has no SavedModel protobuf')]
