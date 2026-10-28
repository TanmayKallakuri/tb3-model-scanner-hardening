"""Keras scanning.

Both the HDF5 ``model_config`` attribute and the ``config.json`` of a ``.keras``
archive hold the same model JSON.  The only layer that executes code when the
model is loaded with unsafe deserialization is a ``Lambda`` whose ``function``
carries serialized bytecode (``class_name == "__lambda__"`` with a base64
``code`` string).  A Lambda that names an existing function, an ``Activation``
layer, and every other layer are clean (threat model section 5).  Layers nest
at any depth -- in ``layers``, in a wrapper's ``layer``, in a nested model's
``config`` -- so the whole JSON tree is walked.
"""
from __future__ import annotations

import io
import json

from modelgate.report import Finding


def _walk_for_lambda(node, path, out):
    if isinstance(node, dict):
        cls = node.get("class_name")
        if cls == "__lambda__" and _has_code(node.get("config")):
            out.append((path, "Lambda carrying serialized bytecode"))
        elif cls == "Lambda":
            fn = node.get("config", {}).get("function") if isinstance(node.get("config"), dict) else None
            if _is_bytecode_function(fn):
                out.append((f"{path}.function", "Lambda carrying serialized bytecode"))
        for key, value in node.items():
            _walk_for_lambda(value, f"{path}.{key}", out)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _walk_for_lambda(value, f"{path}[{i}]", out)


def _has_code(config):
    return isinstance(config, dict) and isinstance(config.get("code"), str)


def _is_bytecode_function(fn):
    if not isinstance(fn, dict):
        return False
    if fn.get("class_name") == "__lambda__" and _has_code(fn.get("config")):
        return True
    return _has_code(fn.get("config"))


def scan_model_json(raw, location):
    """Scan a model-config JSON document (str/bytes) for bytecode Lambdas."""
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", "surrogatepass")
    try:
        config = json.loads(raw)
    except ValueError as exc:
        return [Finding("unsupported", location, f"invalid model JSON: {exc}")]
    hits: list[tuple[str, str]] = []
    _walk_for_lambda(config, location, hits)
    return [Finding("code-execution", where, detail) for where, detail in hits]


def scan_h5_bytes(data, location):
    """Read the model_config attribute of an HDF5 file held in memory.

    Only root attributes are touched; datasets and any external links are never
    followed, so the scan cannot be tricked into reading another file.
    """
    import h5py  # lazy: only needed for HDF5 artifacts

    with h5py.File(io.BytesIO(data), "r") as fh:
        raw = fh.attrs.get("model_config")
    if raw is None:
        return []           # weights-only HDF5: nothing executes on load
    return scan_model_json(raw, f"{location}:model_config")
