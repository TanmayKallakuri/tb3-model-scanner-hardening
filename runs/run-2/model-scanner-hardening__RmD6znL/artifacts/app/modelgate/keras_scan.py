"""Keras scanning: a Lambda layer whose function carries serialized bytecode.

Per the policy (threat model section 5) that is the *only* malicious Keras
construct: a named-function Lambda, an Activation layer, and every other layer
type are clean, whatever they are named.  Layers are found wherever the JSON
nests them -- the top-level ``layers``, a wrapper's ``layer``, and the
``config`` of a nested model -- so we walk the whole configuration tree.
We only parse JSON; the bytecode itself is never decoded or run.
"""
from __future__ import annotations

import io
import json

from modelgate.report import Finding


def scan_h5_bytes(data: bytes, location: str) -> list[Finding]:
    try:
        import h5py
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"h5py unavailable: {exc}")]
    try:
        with h5py.File(io.BytesIO(data), "r") as fh:
            raw = fh.attrs.get("model_config")
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"could not open HDF5: {type(exc).__name__}")]
    if raw is None:
        return []  # weights-only HDF5: no model configuration, nothing to load as code
    config = _load_json(raw)
    if config is None:
        return [Finding("unsupported", location, "model_config is not valid JSON")]
    findings: list[Finding] = []
    _walk(config, f"{location}:model_config", findings)
    return findings


def scan_keras_config(blob: bytes, location: str) -> list[Finding]:
    config = _load_json(blob)
    if config is None:
        return [Finding("unsupported", location, "config.json is not valid JSON")]
    findings: list[Finding] = []
    _walk(config, location, findings)
    return findings


def _load_json(raw):
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        elif not isinstance(raw, str):
            raw = bytes(raw).decode("utf-8", "replace")
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def _walk(obj, path, findings):
    if isinstance(obj, dict):
        if obj.get("class_name") == "Lambda" and _lambda_has_bytecode(obj):
            findings.append(Finding("code-execution", path,
                                    "Lambda layer carrying serialized bytecode"))
        for key, val in obj.items():
            _walk(val, f"{path}/{key}", findings)
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            _walk(val, f"{path}[{i}]", findings)


def _lambda_has_bytecode(layer) -> bool:
    cfg = layer.get("config")
    fn = cfg.get("function") if isinstance(cfg, dict) else None
    return _is_bytecode_function(fn)


def _is_bytecode_function(fn) -> bool:
    if not isinstance(fn, dict):
        return False
    if fn.get("class_name") == "__lambda__":
        return True
    inner = fn.get("config")
    if isinstance(inner, dict) and isinstance(inner.get("code"), str):
        return True
    return False
