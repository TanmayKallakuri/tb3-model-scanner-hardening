"""TensorFlow SavedModel: the policy's closed op table.

Every ``meta_graphs[*]`` is in scope and, within each, ``graph_def.node`` and
every ``graph_def.library.function[*].node_def`` (which covers functions
referenced through ``If``/``While``/``PartitionedCall``... attributes).  Only
the ``op`` of a node carries a capability; names, devices, attributes and
tensor contents are data.
"""
from __future__ import annotations

import os
import re

OP_KINDS = {}
for _op in "PyFunc PyFuncStateless EagerPyFunc".split():
    OP_KINDS[_op] = "code-execution"
for _op in """ReadFile WriteFile MatchingFiles MatchingFilesDataset ImmutableConst WholeFileReader
        WholeFileReaderV2 TextLineReader TextLineReaderV2 FixedLengthRecordReader
        FixedLengthRecordReaderV2 TFRecordReader TFRecordReaderV2 LMDBReader LMDBDataset
        ExperimentalLMDBDataset TextLineDataset TFRecordDataset FixedLengthRecordDataset
        DatasetToTFRecord ExperimentalDatasetToTFRecord SaveDataset SaveDatasetV2 LoadDataset
        SnapshotDataset SnapshotDatasetV2 LoadAndRemapMatrix GenerateVocabRemapping Save
        SaveSlices Restore RestoreSlice""".split():
    OP_KINDS[_op] = "file-access"
for _op in """DataServiceDataset DataServiceDatasetV2 DataServiceDatasetV3 DataServiceDatasetV4
        RegisterDataset RegisterDatasetV2 DistributedSave""".split():
    OP_KINDS[_op] = "network"


def _check_graph(I, mg, mi, location):
    gd = mg.graph_def
    for node in gd.node:
        kind = OP_KINDS.get(node.op)
        if kind:
            I.finding(kind, f"op {node.op}", f"{location}:meta_graphs[{mi}].graph_def.node[{node.name}]")
    for fn in gd.library.function:
        fname = fn.signature.name
        for node in fn.node_def:
            kind = OP_KINDS.get(node.op)
            if kind:
                I.finding(kind, f"op {node.op}",
                          f"{location}:meta_graphs[{mi}].library.function[{fname}].node[{node.name}]")


_OP_RE = re.compile(r'(?m)^\s*op\s*:\s*"((?:[^"\\]|\\.)*)"')


def scan_proto_bytes(I, data: bytes, location: str, text: bool = False):
    try:
        from modelgate.tfproto import saved_model_pb2
    except Exception as exc:  # noqa: BLE001
        I.finding("unsupported", f"cannot load protobuf bindings: {exc}", location)
        return
    sm = saved_model_pb2.SavedModel()
    try:
        if text:
            from google.protobuf import text_format

            text_format.Parse(data.decode("utf-8"), sm, allow_unknown_extension=True,
                              allow_unknown_field=True)
        else:
            sm.ParseFromString(data)
    except Exception as exc:  # noqa: BLE001
        if text:
            # fall back to the node ops written in the text proto
            try:
                body = data.decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                body = ""
            for m in _OP_RE.finditer(body):
                kind = OP_KINDS.get(m.group(1))
                if kind:
                    I.finding(kind, f"op {m.group(1)}", f"{location}:offset {m.start()}")
            return
        I.finding("unsupported", f"cannot parse SavedModel protobuf: {type(exc).__name__}", location)
        return
    for mi, mg in enumerate(sm.meta_graphs):
        _check_graph(I, mg, mi, location)


def scan_saved_model_dir(I, path: str, location: str) -> bool:
    """Scan a SavedModel directory; False if it is not one."""
    pb = os.path.join(path, "saved_model.pb")
    pbtxt = os.path.join(path, "saved_model.pbtxt")
    if os.path.isfile(pb):
        with open(pb, "rb") as fh:
            scan_proto_bytes(I, fh.read(), f"{location}/saved_model.pb", text=False)
        return True
    if os.path.isfile(pbtxt):
        with open(pbtxt, "rb") as fh:
            scan_proto_bytes(I, fh.read(), f"{location}/saved_model.pbtxt", text=True)
        return True
    return False
