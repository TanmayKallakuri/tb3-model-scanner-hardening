"""Keras model-config scanning (HDF5 ``model_config`` and ``.keras`` config).

Under unsafe deserialization a ``Lambda`` layer whose ``function`` or
``output_shape`` carries serialized Python bytecode, and any layer whose
``activation`` is such a lambda, executes that bytecode when the model is
rebuilt and called.  Keras writes a serialized lambda as a ``__lambda__``
object (``{"class_name": "__lambda__", "config": {"code"/"value": ...}}``) or,
in the older HDF5 form, as a ``Lambda`` with ``function_type == "lambda"`` and
a ``[code, defaults, closure]`` function list.  A named function or activation
reference carries no such bytecode and is clean, as is every other layer.
"""
from __future__ import annotations

import io
import json

from modelgate.report import Finding


def _harden_platform():
    """Stop importing h5py (via numpy/platform) from shelling out to ``uname``.

    ``platform.processor()`` runs ``uname -p`` in a subprocess the first time it
    is used, which the registry would count as a failed scan.  Neutralise that
    single choke point so no process is ever started.
    """
    try:
        import platform
        if hasattr(platform, "_Processor"):
            platform._Processor.get = staticmethod(lambda: "")
        platform.processor = lambda: ""
    except Exception:  # noqa: BLE001
        pass


def scan_h5_bytes(data, location):
    _harden_platform()
    try:
        import h5py
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"h5py unavailable: {exc}")]
    try:
        with h5py.File(io.BytesIO(data), "r") as fh:
            raw = fh.attrs.get("model_config")
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"bad HDF5 file: {exc}")]
    if raw is None:
        return []  # weights-only HDF5: no model graph to deserialize
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    return _scan_config_text(raw, location)


def scan_keras_config_bytes(raw, location):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    return _scan_config_text(raw, location)


def _scan_config_text(text, location):
    try:
        config = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"bad model_config JSON: {exc}")]
    findings = []
    _walk(config, location, findings)
    # De-duplicate identical findings.
    seen = set()
    out = []
    for f in findings:
        key = (f.kind, f.detail)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _walk(node, location, findings):
    if isinstance(node, dict):
        if node.get("class_name") == "__lambda__":
            findings.append(Finding(
                "code-execution", location,
                "Keras __lambda__ serialized bytecode"))
        if node.get("class_name") == "Lambda":
            _check_lambda_layer(node.get("config"), location, findings)
        for value in node.values():
            _walk(value, location, findings)
    elif isinstance(node, list):
        for value in node:
            _walk(value, location, findings)


def _check_lambda_layer(config, location, findings):
    if not isinstance(config, dict):
        return
    # Older HDF5 form: function serialized as [code, defaults, closure].
    if config.get("function_type") == "lambda":
        findings.append(Finding(
            "code-execution", location,
            "Keras Lambda layer with serialized lambda bytecode"))
    for key in ("function", "output_shape"):
        val = config.get(key)
        if _is_serialized_bytecode(val):
            findings.append(Finding(
                "code-execution", location,
                f"Keras Lambda {key} carries serialized bytecode"))


def _is_serialized_bytecode(val):
    if isinstance(val, dict):
        if val.get("class_name") == "__lambda__":
            return True
        cfg = val.get("config")
        if isinstance(cfg, dict) and ("code" in cfg or "value" in cfg):
            return True
    if isinstance(val, list) and val and isinstance(val[0], str) and len(val[0]) > 16:
        # [code_b64, defaults, closure]
        return True
    return False
