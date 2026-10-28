"""Parse the vendored protobuf schema, without importing TensorFlow."""
from __future__ import annotations
import os
from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model(data, location, text=False):
    from modelgate.tfproto import saved_model_pb2
    sm = saved_model_pb2.SavedModel()
    if text:
        from google.protobuf import text_format
        text_format.Parse(data.decode('utf-8'), sm)
    else:
        sm.ParseFromString(data)
    findings = []
    for i, meta in enumerate(sm.meta_graphs):
        graph = meta.graph_def
        groups = [(f'{location}:meta_graphs[{i}].graph', graph.node)]
        groups.extend((f'{location}:meta_graphs[{i}].function[{fn.signature.name}]', fn.node_def)
                      for fn in graph.library.function)
        # All FunctionDefs are inspected, including functions reached only by
        # NameAttrList attrs, control-flow branches, or another function.
        for path, nodes in groups:
            for node in nodes:
                kind = TF_OPS.get(node.op)
                if kind:
                    findings.append(Finding(kind, path + f'.node[{node.name}]', 'op ' + node.op))
    return findings


def scan_saved_model_dir(path):
    for filename in ('saved_model.pb', 'saved_model.pbtxt'):
        pb = os.path.join(path, filename)
        if os.path.isfile(pb):
            with open(pb, 'rb') as fh:
                return scan_saved_model(fh.read(), filename, filename.endswith('pbtxt'))
    return [Finding('unsupported', path, 'directory has no SavedModel protobuf')]
