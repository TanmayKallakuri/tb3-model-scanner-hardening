"""Keras HDF5 and ``.keras`` model configurations.

Policy: a ``Lambda`` whose ``function`` (or ``output_shape``) carries
serialized bytecode, and any layer whose ``activation`` is a ``__lambda__``
dict, execute attacker code when the model is rebuilt.  Named function
references, named activations and every other layer type are clean.
Layers are found wherever the JSON nests them (``layers``, a wrapper's
``layer``, a nested model's ``config``).
"""
from __future__ import annotations

import base64
import binascii
import io
import json

CODE = "code-execution"

# sub-trees of a layer entry that are not layer configurations
_SKIP_KEYS = {"inbound_nodes", "build_config", "compile_config", "input_layers", "output_layers",
              "build_input_shape", "metadata", "keras_history", "input_shape", "batch_shape",
              "dtype", "kernel_initializer", "bias_initializer", "kernel_regularizer",
              "bias_regularizer", "activity_regularizer", "kernel_constraint", "bias_constraint",
              "registered_name", "module", "name", "trainable"}
_ACTIVATION_KEYS = ("activation",)  # the policy's rule; other config keys are clean


def _code_is_bytecode(code) -> bool:
    """``python_utils.func_load``: base64 (or raw_unicode_escape) of a
    marshalled code object."""
    if not isinstance(code, str) or not code:
        return False
    raw = None
    try:
        raw = base64.decodebytes(code.encode("ascii"))
    except (UnicodeEncodeError, binascii.Error, ValueError):
        try:
            raw = code.encode("raw_unicode_escape")
        except Exception:  # noqa: BLE001
            return False
    if not raw:
        return False
    return (raw[0] & 0x7F) == ord("c")


def lambda_payload(v, legacy_list=False):
    """True if ``v`` is a serialized lambda (bytecode) as Keras writes it."""
    if isinstance(v, dict):
        if v.get("class_name") != "__lambda__":
            return False
        cfg = v.get("config")
        if not isinstance(cfg, dict):
            return False
        if "code" in cfg:
            return _code_is_bytecode(cfg.get("code"))
        val = cfg.get("value")
        if isinstance(val, (list, tuple)) and val:
            return _code_is_bytecode(val[0])
        if isinstance(val, str):
            return _code_is_bytecode(val)
        return False
    if legacy_list and isinstance(v, (list, tuple)) and len(v) == 3 and isinstance(v[0], str):
        return _code_is_bytecode(v[0])
    return False


def scan_config_obj(I, cfg, location: str):
    stack = [(cfg, "model")]
    seen = 0
    while stack:
        node, path = stack.pop()
        seen += 1
        if seen > 200000:
            break
        if isinstance(node, list):
            for i, x in enumerate(node):
                if isinstance(x, (dict, list)):
                    stack.append((x, f"{path}[{i}]"))
            continue
        if not isinstance(node, dict):
            continue
        cls = node.get("class_name")
        conf = node.get("config")
        if isinstance(cls, str) and isinstance(conf, dict):
            lname = conf.get("name", "")
            here = f"{path}<{cls}:{lname}>" if isinstance(lname, str) else f"{path}<{cls}>"
            if cls == "Lambda" or cls.endswith(">Lambda"):
                legacy = "module" not in node
                if lambda_payload(conf.get("function"), legacy_list=legacy):
                    I.finding(CODE, "Lambda layer function carries serialized bytecode", f"{location}:{here}")
                if lambda_payload(conf.get("output_shape"), legacy_list=legacy):
                    I.finding(CODE, "Lambda layer output_shape carries serialized bytecode",
                              f"{location}:{here}")
            for key in _ACTIVATION_KEYS:
                if lambda_payload(conf.get(key)):
                    I.finding(CODE, f"layer {key} is a serialized lambda (bytecode)", f"{location}:{here}")
            for k, v in conf.items():
                if k in _SKIP_KEYS or k in ("function", "output_shape") + _ACTIVATION_KEYS:
                    continue
                if isinstance(v, (dict, list)):
                    stack.append((v, f"{here}.{k}"))
            continue
        for k, v in node.items():
            if k in ("inbound_nodes", "build_config", "compile_config"):
                continue
            if isinstance(v, (dict, list)):
                stack.append((v, f"{path}.{k}"))


def scan_config_text(I, text, location: str):
    try:
        cfg = json.loads(text)
    except Exception:  # noqa: BLE001 - Keras cannot parse it either
        return
    scan_config_obj(I, cfg, location)


def scan_config_bytes(I, data: bytes, location: str):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return
    scan_config_text(I, text, location)


def scan_h5_bytes(I, data: bytes, location: str):
    import warnings

    from modelgate import h5lite

    attrs = h5lite.read_root_attrs(data)
    if attrs == h5lite.CORRUPT:
        return  # malformed: Keras cannot load it, and libhdf5 may hang on it
    if attrs != h5lite.UNSUPPORTED:
        raw = attrs.get("model_config")
        if raw is None:
            return
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                return
        scan_config_text(I, raw, f"{location}:model_config")
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import h5py
    except Exception:  # noqa: BLE001
        I.finding("unsupported", "h5py is not available to read the HDF5 file", location)
        return
    try:
        with h5py.File(io.BytesIO(data), "r") as fh:
            raw = fh.attrs.get("model_config")
    except Exception:  # noqa: BLE001 - not an HDF5 file Keras can open
        return
    if raw is None:
        return
    try:
        import numpy as np

        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
            if isinstance(raw, list) and raw:
                raw = raw[0]
    except Exception:  # noqa: BLE001
        pass
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return
    if not isinstance(raw, str):
        return
    scan_config_text(I, raw, f"{location}:model_config")
