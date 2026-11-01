"""Keras model configurations (HDF5 ``model_config`` and ``.keras`` config.json).

A ``__lambda__`` dict carries marshalled bytecode that Keras turns into a
function (python_utils.func_load) when it rebuilds the model: in a Lambda
layer's ``function`` / ``output_shape``, and in any layer's ``activation``
(plus the other keys Keras deserializes through the same path).  Named
function references, named activations and all other layers are clean.
"""
from __future__ import annotations

import io
import json

from modelgate.policy import KERAS_GENERIC_KEYS, KERAS_GENERIC_SUFFIXES, KERAS_LAMBDA_KEYS
from modelgate.report import Finding


def _is_lambda(v) -> bool:
    if not isinstance(v, dict) or v.get("class_name") != "__lambda__":
        return False
    cfg = v.get("config")
    return isinstance(cfg, dict) and ("code" in cfg or "value" in cfg)


def _contains_lambda(v, depth=0) -> bool:
    if depth > 64:
        return False
    if _is_lambda(v):
        return True
    if isinstance(v, dict):
        return any(_contains_lambda(x, depth + 1) for x in v.values())
    if isinstance(v, list):
        return any(_contains_lambda(x, depth + 1) for x in v)
    return False


def _generic_key(key: str) -> bool:
    return key in KERAS_GENERIC_KEYS or key.endswith(KERAS_GENERIC_SUFFIXES)


def scan_config(cfg, location: str, findings: list):
    seen = set()

    def add(where, detail):
        if (where, detail) not in seen:
            seen.add((where, detail))
            findings.append(Finding("code-execution", where, detail))

    def walk(node, path, depth):
        if depth > 200:
            return
        if isinstance(node, dict):
            cn = node.get("class_name")
            cfgd = node.get("config")
            here = path
            if isinstance(cn, str) and isinstance(cfgd, dict):
                lname = cfgd.get("name")
                here = f"{path}/{lname}" if isinstance(lname, str) else f"{path}/{cn}"
                if cn == "Lambda":
                    for key in KERAS_LAMBDA_KEYS:
                        if _is_lambda(cfgd.get(key)):
                            add(f"{location}:{here}", f"Lambda layer {key} carries serialized bytecode (__lambda__)")
                    fn = cfgd.get("function")
                    if ("module" not in node and isinstance(fn, list) and fn and isinstance(fn[0], str)
                            and len(fn[0]) >= 8):
                        # legacy (Keras 2) config: model_from_config turns [code, defaults, closure]
                        # into a __lambda__ and func_load materialises it
                        add(f"{location}:{here}", "legacy Lambda layer function carries serialized bytecode")
                    if _contains_lambda(cfgd.get("arguments")):
                        add(f"{location}:{here}", "Lambda layer arguments carry serialized bytecode (__lambda__)")
                for key, val in cfgd.items():
                    if isinstance(key, str) and _generic_key(key) and _is_lambda(val):
                        add(f"{location}:{here}", f"layer {cn} {key} is a __lambda__ with serialized bytecode")
            for k, v in node.items():
                walk(v, f"{here}/{k}" if not (isinstance(cn, str) and isinstance(cfgd, dict)) else here, depth + 1)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]", depth + 1)

    walk(cfg, "", 0)
    # compile / training configuration: losses and metrics are deserialized too
    for key in ("compile_config", "training_config"):
        sub = cfg.get(key) if isinstance(cfg, dict) else None
        if isinstance(sub, dict):
            scan_training(sub, f"{location}:{key}", findings)


def scan_training(cfg, location, findings):
    for key in ("loss", "metrics", "weighted_metrics", "optimizer", "optimizer_config", "loss_weights"):
        if _contains_lambda(cfg.get(key)):
            findings.append(Finding("code-execution", location,
                                    f"{key} carries serialized bytecode (__lambda__)"))


def _decode_json(raw, what, location):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if hasattr(raw, "item") and not isinstance(raw, str):
        try:
            raw = raw.item()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
        except Exception:
            pass
    if not isinstance(raw, str):
        raise ValueError(f"{what} is not a string")
    return json.loads(raw)


def _root_attrs_h5py(data: bytes):
    import h5py

    with h5py.File(io.BytesIO(data), "r") as fh:
        return fh.attrs.get("model_config"), fh.attrs.get("training_config")


def scan_h5_bytes(data: bytes, location: str) -> list:
    """Keras HDF5 file: the root attribute ``model_config`` holds the model."""
    from modelgate import h5attrs

    findings: list = []
    try:
        attrs = h5attrs.root_attributes(data)
        raw, training = attrs.get("model_config"), attrs.get("training_config")
    except h5attrs.H5Unsupported:
        raw, training = _root_attrs_h5py(data)
    except h5attrs.H5Corrupt as exc:
        findings.append(Finding("unsupported", location, f"malformed HDF5 file: {exc}"))
        return findings
    if raw is not None:
        try:
            cfg = _decode_json(raw, "model_config", location)
        except ValueError as exc:
            findings.append(Finding("unsupported", location, f"model_config is not valid JSON: {exc}"))
            return findings
        scan_config(cfg, location, findings)
    if training is not None:
        try:
            tcfg = _decode_json(training, "training_config", location)
            if isinstance(tcfg, dict):
                scan_training(tcfg, f"{location}:training_config", findings)
        except ValueError:
            pass
    return findings


def scan_keras_config_bytes(raw: bytes, location: str) -> list:
    findings: list = []
    try:
        cfg = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        findings.append(Finding("unsupported", location, f"config.json is not valid JSON: {exc}"))
        return findings
    scan_config(cfg, location, findings)
    return findings
