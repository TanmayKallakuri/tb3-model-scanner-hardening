"""Keras scanning: serialized-lambda bytecode in the model configuration.

A Keras model config (the HDF5 root attribute ``model_config``, or ``config.json``
inside a ``.keras`` archive) is clean unless it carries a serialized lambda.
Keras writes those as a dict ``{"class_name": "__lambda__", "config": {...}}``
holding base64 marshalled bytecode, used for a ``Lambda`` layer's ``function``
or ``output_shape`` and for any layer's ``activation``.  Deserialising one with
unsafe mode materialises and runs attacker bytecode, so any ``__lambda__`` dict
anywhere in the config is (A) code execution.  Named function references
(``class_name == "function"``) and named activations are clean.
"""
from __future__ import annotations

import io
import json

from modelgate.report import Finding

_MAX_NODES = 200_000


def _h5py():
    # Never let HDF5 take a file lock: the scanner's sandbox treats any write
    # as a failed scan, and locking a read-only artifact is unnecessary.
    import os
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import h5py
    return h5py


def scan_h5(path: str, location: str) -> list:
    h5py = _h5py()
    with h5py.File(path, "r") as fh:
        return _from_attr(fh, location)


def scan_h5_bytes(data: bytes, location: str) -> list:
    h5py = _h5py()
    with h5py.File(io.BytesIO(data), "r") as fh:
        return _from_attr(fh, location)


def _from_attr(fh, location: str) -> list:
    raw = fh.attrs.get("model_config")
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    elif not isinstance(raw, str):
        raw = bytes(raw).decode("utf-8", "replace") if hasattr(raw, "__bytes__") else str(raw)
    try:
        config = json.loads(raw)
    except Exception as exc:
        return [Finding("unsupported", location, f"bad model_config JSON: {exc}")]
    return _walk(config, location)


def scan_config_json(data: bytes, location: str) -> list:
    try:
        config = json.loads(data.decode("utf-8", "replace"))
    except Exception as exc:
        return [Finding("unsupported", location, f"bad config.json: {exc}")]
    return _walk(config, location)


def _walk(config, location: str) -> list:
    findings = []
    seen = []
    counter = [0]

    def rec(node, path):
        counter[0] += 1
        if counter[0] > _MAX_NODES:
            return
        if isinstance(node, dict):
            # Keras 3 serialises a lambda as {"class_name": "__lambda__", ...};
            # Keras 2 marks a bytecode lambda with "function_type": "lambda".
            # Both materialise attacker bytecode under unsafe deserialization.
            if node.get("class_name") == "__lambda__" or node.get("function_type") == "lambda":
                if path not in seen:
                    seen.append(path)
                    findings.append(Finding(
                        "code-execution", f"{location}:{path}",
                        "Keras serialized lambda bytecode"))
            for k, v in node.items():
                rec(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                rec(v, f"{path}[{i}]")

    rec(config, "config")
    return findings
