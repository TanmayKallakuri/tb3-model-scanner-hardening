"""Parse TensorFlow's protobuf schema without importing TensorFlow."""
from __future__ import annotations

import os

from modelgate.limits import AnalysisLimit, MAX_BYTES
from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model(data, location, text=False, require_graph=False):
    from modelgate.tfproto import saved_model_pb2

    if len(data) > MAX_BYTES: raise AnalysisLimit('SavedModel graph size limit exceeded')
    sm = saved_model_pb2.SavedModel()
    if text:
        from google.protobuf import text_format
        text_format.Parse(data.decode('utf-8'), sm)
    else:
        sm.ParseFromString(data)
    if require_graph and not sm.meta_graphs: raise ValueError('not a SavedModel graph')
    findings = []
    for i, meta in enumerate(sm.meta_graphs):
        graph = meta.graph_def
        for node in graph.node:
            kind = TF_OPS.get(node.op)
            if kind:
                findings.append(Finding(kind, f'{location}:meta_graphs[{i}].node[{node.name}]', 'TensorFlow op ' + node.op))
                if len(findings) >= 100: return findings
        # Inspect every definition, including functions referenced exclusively
        # by AttrValue.func/list.func. Ordinary tensor/attribute data is inert.
        for fn in graph.library.function:
            for node in fn.node_def:
                kind = TF_OPS.get(node.op)
                if kind:
                    loc = f'{location}:meta_graphs[{i}].function[{fn.signature.name}].node[{node.name}]'
                    findings.append(Finding(kind, loc, 'TensorFlow op ' + node.op))
                    if len(findings) >= 100: return findings
    return findings


def scan_saved_model_dir(path):
    for filename in ('saved_model.pb', 'saved_model.pbtxt'):
        pb = os.path.join(path, filename)
        if os.path.isfile(pb):
            with open(pb, 'rb') as fh:
                return scan_saved_model(fh.read(MAX_BYTES + 1), pb, filename.endswith('pbtxt'))
    return [Finding('unsupported', path, 'directory has no saved_model.pb or saved_model.pbtxt')]
