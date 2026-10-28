"""Keras scanning: a ``Lambda`` layer whose function carries serialized
bytecode executes attacker code when the model is loaded with unsafe
deserialization.  Named-function Lambdas, ``Activation`` layers and every
other layer type are clean.  Layers are found wherever the JSON nests them."""
from __future__ import annotations

import json

from modelgate.report import Finding


def scan_h5(path: str, location: str) -> list[Finding]:
    import h5py
    try:
        with h5py.File(path, "r") as fh:
            raw = fh.attrs.get("model_config")
    except Exception as exc:
        return [Finding("unsupported", location, f"could not open HDF5: {exc}")]
    if raw is None:
        return []
    return _scan_config(raw, location)


def scan_keras_v3(config_bytes: bytes, location: str) -> list[Finding]:
    return _scan_config(config_bytes, location)


def _scan_config(raw, location: str) -> list[Finding]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        config = json.loads(raw)
    except Exception as exc:
        return [Finding("unsupported", location, f"bad Keras config JSON: {exc}")]
    findings: list[Finding] = []
    seen = set()
    _walk(config, location, findings, seen, 0)
    return findings


def _is_bytecode_function(fn) -> bool:
    """True when a Lambda's serialized function carries raw bytecode."""
    if not isinstance(fn, dict):
        return False
    if fn.get("class_name") == "__lambda__":
        return True
    cfg = fn.get("config")
    if isinstance(cfg, dict) and "code" in cfg:
        # a function spec whose config holds a (base64) code object
        return True
    return False


def _walk(node, loc, findings, seen, depth):
    if depth > 200:
        return
    if isinstance(node, dict):
        cls = node.get("class_name")
        if cls == "Lambda":
            cfg = node.get("config", {})
            fn = cfg.get("function") if isinstance(cfg, dict) else None
            name = cfg.get("name", "lambda") if isinstance(cfg, dict) else "lambda"
            if _is_bytecode_function(fn):
                key = ("lambda", name)
                if key not in seen:
                    seen.add(key)
                    findings.append(Finding(
                        "code-execution", f"{loc}:Lambda[{name}]",
                        "Lambda layer function carries serialized bytecode"))
        # Keras 2 style: a Lambda whose function is a marshalled code list
        if cls == "Lambda" and node.get("config", {}).get("function_type") == "lambda":
            name = node.get("config", {}).get("name", "lambda")
            key = ("lambda2", name)
            if key not in seen:
                seen.add(key)
                findings.append(Finding(
                    "code-execution", f"{loc}:Lambda[{name}]",
                    "Lambda layer carries a marshalled function"))
        # a serialized bytecode function anywhere (defensive)
        if cls == "__lambda__":
            key = ("rawlambda", id(node))
            if key not in seen:
                seen.add(key)
                findings.append(Finding(
                    "code-execution", f"{loc}:__lambda__",
                    "serialized lambda bytecode in config"))
        for v in node.values():
            _walk(v, loc, findings, seen, depth + 1)
    elif isinstance(node, list):
        for v in node:
            _walk(v, loc, findings, seen, depth + 1)
