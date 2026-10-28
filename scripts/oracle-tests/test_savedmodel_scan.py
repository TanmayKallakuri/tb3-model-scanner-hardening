"""SavedModel walker: every meta graph, library function and op-table entry;
saver ops and node names never count."""
import os
import shutil

import pytest

from conftest import kinds
from modelgate import dispatch, policy, savedmodel_scan
from modelgate.report import Unsupported


def _protos():
    from google.protobuf import text_format
    from modelgate.tfproto import saved_model_pb2
    return text_format, saved_model_pb2


def _add_node(container, name, op, func_attrs=None, string_attr=None):
    node = container.add()
    node.name = name
    node.op = op
    for attr_name, fname in (func_attrs or {}).items():
        node.attr[attr_name].func.name = fname
    if string_attr is not None:
        node.attr["value"].tensor.string_val.append(string_attr.encode())
    return node


def build_model(top_ops=(), functions=None, extra_meta_graphs=()):
    _, pb = _protos()
    model = pb.SavedModel()
    model.saved_model_schema_version = 1
    for ops_by_graph in ((top_ops, functions or {}), *extra_meta_graphs):
        mg = model.meta_graphs.add()
        mg.meta_info_def.tags.append("serve")
        graph = mg.graph_def
        for name, op in ops_by_graph[0]:
            _add_node(graph.node, name, op)
        for fname, nodes in ops_by_graph[1].items():
            fn = graph.library.function.add()
            fn.signature.name = fname
            for name, op in nodes:
                _add_node(fn.node_def, name, op)
        # a real-shaped saver and restore library, always present
        saver = graph.library.function.add()
        saver.signature.name = "__inference__traced_save_1"
        for name, op in (("ShardedFilename", "ShardedFilename"), ("SaveV2", "SaveV2"),
                         ("MergeV2Checkpoints", "MergeV2Checkpoints"), ("Identity", "Identity")):
            _add_node(saver.node_def, name, op)
        restore = graph.library.function.add()
        restore.signature.name = "__inference__traced_restore_2"
        _add_node(restore.node_def, "RestoreV2", "RestoreV2")
        _add_node(graph.node, "saver_filename", "Placeholder")
        _add_node(graph.node, "StatefulPartitionedCall_1", "StatefulPartitionedCall", {"f": "__inference__traced_save_1"})
        _add_node(graph.node, "StatefulPartitionedCall_2", "StatefulPartitionedCall", {"f": "__inference__traced_restore_2"})
    return model


def write_dir(tmp_path, model, text=False, name="sm"):
    text_format, _ = _protos()
    d = tmp_path / name
    d.mkdir()
    (d / "variables").mkdir()
    (d / "variables" / "variables.index").write_bytes(b"")
    if text:
        (d / "saved_model.pbtxt").write_text(text_format.MessageToString(model), encoding="utf-8")
    else:
        (d / "saved_model.pb").write_bytes(model.SerializeToString())
    return str(d)


@pytest.mark.parametrize("text", [False, True])
def test_benign_model_with_saver_is_clean(tmp_path, text):
    model = build_model(top_ops=[("x", "Placeholder"), ("dense_1/MatMul", "MatMul")],
                        functions={"__inference_serve_9": [("Identity", "Identity")]})
    assert savedmodel_scan.scan_saved_model_dir(write_dir(tmp_path, model, text)) == []


@pytest.mark.parametrize("text", [False, True])
def test_top_level_readfile(tmp_path, text):
    model = build_model(top_ops=[("dense_1/BiasAdd", "ReadFile")])
    found = savedmodel_scan.scan_saved_model_dir(write_dir(tmp_path, model, text))
    assert kinds(found) == ["file-access"]
    assert found[0].location == "meta_graphs[0].node[dense_1/BiasAdd]"


def test_op_in_library_function_reached_through_call(tmp_path):
    model = build_model(top_ops=[("call", "StatefulPartitionedCall")],
                        functions={"__inference_serve_30": [("x", "Identity"), ("save/Identity_2", "WriteFile")]})
    found = savedmodel_scan.scan_saved_model_dir(write_dir(tmp_path, model))
    assert [(f.kind, f.location) for f in found] == [
        ("file-access", "meta_graphs[0].library.function[__inference_serve_30].node_def[save/Identity_2]")]


def test_op_in_cond_branch_and_while_body(tmp_path):
    model = build_model(functions={"cond_true_15": [("ReadFile", "ReadFile")], "while_body_7": [("y", "MatchingFiles")]})
    found = savedmodel_scan.scan_saved_model_dir(write_dir(tmp_path, model))
    assert sorted(f.location for f in found) == [
        "meta_graphs[0].library.function[cond_true_15].node_def[ReadFile]",
        "meta_graphs[0].library.function[while_body_7].node_def[y]",
    ]


def test_second_meta_graph(tmp_path):
    model = build_model(top_ops=[("a", "Identity")], extra_meta_graphs=[([("b", "PyFunc")], {})])
    found = savedmodel_scan.scan_saved_model_dir(write_dir(tmp_path, model))
    assert [(f.kind, f.location) for f in found] == [("code-execution", "meta_graphs[1].node[b]")]


def test_benign_second_meta_graph(tmp_path):
    model = build_model(top_ops=[("a", "Identity")], extra_meta_graphs=[([("b", "Identity")], {"cond_true_1": [("c", "Const")]})])
    assert savedmodel_scan.scan_saved_model_dir(write_dir(tmp_path, model)) == []


@pytest.mark.parametrize("op,kind", sorted(policy.TF_OPS.items()))
def test_every_table_op(tmp_path, op, kind):
    model = build_model(top_ops=[("n", op)])
    found = savedmodel_scan.scan_saved_model_dir(write_dir(tmp_path, model))
    assert [(f.kind, f.detail) for f in found] == [(kind, f"op {op}")]


def test_table_is_the_threat_model_table():
    assert len(policy.TF_OPS) == 42
    assert policy.TF_SAVER_OPS == {"SaveV2", "RestoreV2", "MergeV2Checkpoints", "ShardedFilename", "ShardedFilespec"}
    assert not set(policy.TF_SAVER_OPS) & set(policy.TF_OPS)


@pytest.mark.parametrize("name,op", [
    ("ReadFile", "Identity"), ("WriteFile", "Const"), ("PyFunc", "NoOp"), ("save/ReadFile_1", "Placeholder"),
    ("readfile", "Identity"), ("ReadFileV2", "Identity"), ("ImmutableConst_1", "Const"),
])
def test_node_names_and_near_misses_are_clean(tmp_path, name, op):
    model = build_model(top_ops=[(name, op)])
    assert savedmodel_scan.scan_saved_model_dir(write_dir(tmp_path, model)) == []


def test_saver_ops_and_string_constants_are_clean(tmp_path):
    model = build_model(top_ops=[("ShardedFilespec", "ShardedFilespec"), ("Save", "SaveV2"), ("Restore", "RestoreV2")])
    _add_node(model.meta_graphs[0].graph_def.node, "msg", "Const", string_attr="PyFunc ReadFile os.system")
    assert savedmodel_scan.scan_saved_model_dir(write_dir(tmp_path, model)) == []


def test_lowercase_or_case_variant_is_not_the_op(tmp_path):
    model = build_model(top_ops=[("a", "readfile"), ("b", "READFILE"), ("c", "Readfile")])
    assert savedmodel_scan.scan_saved_model_dir(write_dir(tmp_path, model)) == []


def test_empty_directory_is_unsupported(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(Unsupported):
        savedmodel_scan.scan_saved_model_dir(str(d))


def test_corrupt_pb_and_pbtxt_are_unsupported(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "saved_model.pb").write_bytes(b"\xff\xff\xff\xff\xff\x08")
    with pytest.raises(Unsupported):
        savedmodel_scan.scan_saved_model_dir(str(d))
    e = tmp_path / "badtxt"
    e.mkdir()
    (e / "saved_model.pbtxt").write_text("meta_graphs { this is not : a proto", encoding="utf-8")
    with pytest.raises(Unsupported):
        savedmodel_scan.scan_saved_model_dir(str(e))


def test_dispatch_reports_unsupported_directory(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    result = dispatch.scan_path(str(d))
    assert result.verdict == "malicious"
    assert kinds(result.findings) == ["unsupported"]


# --- real SavedModels -------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("benign", []),
    ("nested_readfile", ["file-access"]),
    ("writefile", ["file-access"]),
])
def test_reference_saved_models(refs, name, expected):
    result = dispatch.scan_path(os.path.join(refs, "tf-files", name))
    assert kinds(result.findings) == expected
    if name == "nested_readfile":
        assert result.findings[0].location == "meta_graphs[0].library.function[cond_true_15].node_def[cond/ReadFile]"


@pytest.mark.parametrize("name,expected", [("benign", []), ("nested_readfile", ["file-access"]), ("writefile", ["file-access"])])
def test_reference_saved_models_as_pbtxt(refs, tmp_path, name, expected):
    text_format, pb = _protos()
    src = os.path.join(refs, "tf-files", name)
    model = pb.SavedModel()
    with open(os.path.join(src, "saved_model.pb"), "rb") as f:
        model.ParseFromString(f.read())
    dst = tmp_path / name
    shutil.copytree(src, dst)
    os.remove(dst / "saved_model.pb")
    (dst / "saved_model.pbtxt").write_text(text_format.MessageToString(model), encoding="utf-8")
    result = dispatch.scan_path(str(dst))
    assert kinds(result.findings) == expected


def test_keras_saved_models_are_clean(refs):
    for name in ("flat_savedmodel", "nested_savedmodel"):
        result = dispatch.scan_path(os.path.join(refs, "keras-files", name))
        assert result.verdict == "clean", result.findings
