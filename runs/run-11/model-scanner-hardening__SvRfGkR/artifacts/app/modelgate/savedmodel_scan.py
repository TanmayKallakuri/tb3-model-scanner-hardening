"""Read protobuf data only; never import or invoke TensorFlow."""
import os

from modelgate.policy import TF_OPS
from modelgate.report import Finding


def scan_saved_model(data, location, text=False):
    from modelgate.tfproto import saved_model_pb2
    sm = saved_model_pb2.SavedModel()
    if text:
        from google.protobuf import text_format
        text_format.Parse(data.decode("utf-8"), sm)
    else:
        sm.ParseFromString(data)
    findings = []
    for i, meta in enumerate(sm.meta_graphs):
        graph = meta.graph_def
        # All functions are in scope, including ones referenced only by a
        # node's AttrValue.func/list.func. Iterate definitions once so cycles
        # and control-flow edges need no recursive graph traversal.
        groups = [(f"meta_graph[{i}]/graph", graph.node)]
        groups.extend((f"meta_graph[{i}]/function[{f.signature.name}]", f.node_def)
                      for f in graph.library.function)
        for scope, nodes in groups:
            for node in nodes:
                kind = TF_OPS.get(node.op)
                if kind:
                    findings.append(Finding(kind, f"{location}:{scope}/node[{node.name}]", f"op {node.op}"))
                    if len(findings) >= 100:
                        return findings
    return findings


def scan_saved_model_dir(path):
    for name in ("saved_model.pb", "saved_model.pbtxt"):
        pb = os.path.join(path, name)
        if os.path.isfile(pb):
            with open(pb, "rb") as fh:
                return scan_saved_model(fh.read(), name, name.endswith("pbtxt"))
    return [Finding("unsupported", path, "directory has no SavedModel definition")]
