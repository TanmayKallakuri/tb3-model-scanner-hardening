"""Keras configs: Lambda bytecode in both forms at any nesting depth; named
functions, Activation layers and weights-only files are clean."""
import copy
import io
import json
import os
import zipfile

import pytest

from conftest import kinds
from modelgate import containers, dispatch, keras_scan
from modelgate.report import Unsupported

CODE = "4wEAAAAAAAAAAAAAAAIAAAADAAAA8wwAAACXAHwAZAF6BQAAUwApAk7nAAAAAAAAAECpAKkB2gF0cwEAAAAg+h0vd29yay9zY3JpcHRzL2tlcmFzX3NoYXBlcy5wedoIPGxhbWJkYT5yBwAAAAoAAABzCAAAAIAAoCGgY6In8wAAAAA="

LAMBDA3 = {"module": "keras.layers", "class_name": "Lambda", "config": {
    "name": "lam", "trainable": True,
    "function": {"class_name": "__lambda__", "config": {"code": CODE, "defaults": None, "closure": None}},
    "arguments": {}}, "registered_name": None}
LAMBDA2 = {"class_name": "Lambda", "config": {
    "name": "lam2", "trainable": True, "function": [CODE, None, None], "function_type": "lambda",
    "module": "__main__", "output_shape": None, "output_shape_type": "raw", "arguments": {}}}
NAMED = {"module": "keras.layers", "class_name": "Lambda", "config": {
    "name": "named_fn", "trainable": True,
    "function": {"module": "builtins", "class_name": "function", "config": "relu", "registered_name": "function"},
    "arguments": {}}, "registered_name": None}
NAMED2 = {"class_name": "Lambda", "config": {"name": "named2", "function": "relu", "function_type": "function", "module": "builtins"}}
DENSE = {"module": "keras.layers", "class_name": "Dense", "config": {"name": "dense", "units": 3, "activation": "linear"}}
ACT = {"module": "keras.layers", "class_name": "Activation", "config": {"name": "act", "activation": "relu"}}


def functional(layers, name="functional"):
    return {"module": "keras.src.models.functional", "class_name": "Functional",
            "config": {"name": name, "trainable": True, "layers": layers}}


def sequential(layers, name="sequential"):
    return {"module": "keras", "class_name": "Sequential", "config": {"name": name, "layers": layers}}


def wrapper(kind, layer):
    return {"module": "keras.layers", "class_name": kind, "config": {"name": kind.lower(), "layer": layer}}


def h5_with_config(config):
    import h5py
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.attrs["model_config"] = json.dumps(config)
        f.create_dataset("w", data=[1.0, 2.0], track_times=False)
    return buf.getvalue()


def keras_zip(config):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("metadata.json", json.dumps({"keras_version": "3.10.0"}))
        z.writestr("config.json", json.dumps(config))
        z.writestr("model.weights.h5", h5_without_config())
    return buf.getvalue()


def h5_without_config():
    import h5py
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("dense/kernel", data=[[1.0]], track_times=False)
    return buf.getvalue()


# --- config walker ----------------------------------------------------------------

@pytest.mark.parametrize("config", [
    functional([DENSE, LAMBDA3, DENSE]),
    functional([DENSE, LAMBDA2]),
    sequential([LAMBDA3]),
    functional([DENSE, functional([DENSE, LAMBDA3], "inner")]),
    functional([sequential([functional([LAMBDA2], "deep")])]),
    functional([wrapper("TimeDistributed", LAMBDA3)]),
    functional([wrapper("Bidirectional", LAMBDA2)]),
    functional([wrapper("TimeDistributed", sequential([LAMBDA3]))]),
    functional([{"class_name": "Dense", "config": {"name": "d", "activation": {"class_name": "__lambda__", "config": {"code": CODE}}}}]),
    {"class_name": "Lambda", "config": {"name": "top", "function": [CODE, None, None], "function_type": "lambda"}},
    functional([{"class_name": "Lambda", "config": {"name": "os", "function": "relu", "function_type": "function",
                                                     "output_shape": [CODE, None, None], "output_shape_type": "lambda"}}]),
])
def test_lambda_bytecode_is_code_execution(config):
    found = keras_scan.scan_keras_config(config, "cfg")
    assert kinds(found) == ["code-execution"]


@pytest.mark.parametrize("config", [
    functional([DENSE, NAMED, DENSE]),
    functional([DENSE, NAMED2]),
    functional([DENSE, ACT]),
    functional([DENSE, functional([DENSE, NAMED], "inner"), wrapper("TimeDistributed", DENSE)]),
    functional([wrapper("Bidirectional", {"class_name": "LSTM", "config": {"name": "lstm", "units": 4}})]),
    sequential([{"class_name": "Dense", "config": {"name": "Lambda", "activation": "__lambda__"}}]),
    functional([{"class_name": "Dense", "config": {"name": "d", "notes": "class_name __lambda__ os.system"}}]),
    {"class_name": "Lambda", "config": {"name": "x", "function": None}},
    [],
    {},
])
def test_named_functions_and_other_layers_are_clean(config):
    assert keras_scan.scan_keras_config(config, "cfg") == []


def test_finding_names_the_layer_path():
    found = keras_scan.scan_keras_config(functional([DENSE, functional([LAMBDA3], "inner")]), "m.h5")
    assert len(found) == 1
    assert found[0].location == "m.h5.config.layers[1].config.layers[0]:Lambda(lam)"


def test_deep_nesting_is_unsupported():
    node = LAMBDA3
    for _ in range(600):
        node = {"config": {"layers": [node]}}
    with pytest.raises(Unsupported):
        keras_scan.scan_keras_config(node, "deep")


# --- HDF5 and .keras archives -------------------------------------------------------

def test_h5_lambda_and_clean_variants():
    assert kinds(containers.scan_bytes(h5_with_config(functional([DENSE, LAMBDA3])), "m.h5")) == ["code-execution"]
    assert containers.scan_bytes(h5_with_config(functional([DENSE, NAMED])), "m.h5") == []
    assert containers.scan_bytes(h5_without_config(), "w.h5") == []


def test_h5_model_config_not_json_is_unsupported():
    import h5py
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.attrs["model_config"] = "{not json"
    with pytest.raises(Unsupported):
        containers.scan_bytes(buf.getvalue(), "bad.h5")


def test_keras_archive_lambda_and_clean_variant():
    assert kinds(containers.scan_bytes(keras_zip(functional([DENSE, LAMBDA3])), "m.keras")) == ["code-execution"]
    assert containers.scan_bytes(keras_zip(functional([DENSE, NAMED, ACT])), "m.keras") == []


# --- reference files ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["flat.keras", "flat.h5", "nested.keras", "nested.h5"])
def test_reference_keras_files_are_malicious(refs, name):
    result = dispatch.scan_path(os.path.join(refs, "keras-files", name))
    assert kinds(result.findings) == ["code-execution"], result.findings
    assert "Lambda(" in result.findings[0].location


def _replace_lambdas(node):
    if isinstance(node, dict):
        if node.get("class_name") == "Lambda":
            node["config"]["function"] = copy.deepcopy(NAMED["config"]["function"])
        for v in node.values():
            _replace_lambdas(v)
    elif isinstance(node, list):
        for v in node:
            _replace_lambdas(v)


@pytest.mark.parametrize("name", ["flat.keras", "nested.keras"])
def test_reference_keras_archive_with_named_function_is_clean(refs, name):
    with zipfile.ZipFile(os.path.join(refs, "keras-files", name)) as z:
        config = json.loads(z.read("config.json"))
        others = [(n, z.read(n)) for n in z.namelist() if n != "config.json"]
    _replace_lambdas(config)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, data in others:
            z.writestr(n, data)
        z.writestr("config.json", json.dumps(config))
    assert containers.scan_bytes(buf.getvalue(), name) == []


@pytest.mark.parametrize("name", ["flat.h5", "nested.h5"])
def test_reference_h5_with_named_function_is_clean(refs, name):
    import h5py
    with h5py.File(os.path.join(refs, "keras-files", name), "r") as f:
        raw = f.attrs["model_config"]
    config = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    _replace_lambdas(config)
    assert containers.scan_bytes(h5_with_config(config), name) == []
