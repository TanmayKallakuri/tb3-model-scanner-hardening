"""Keras model scanning.

Both the HDF5 (`model_config` root attribute) and the v3 archive
(`config.json`) carry the same model JSON.  The only executable payload the
policy recognises is a serialized lambda: a ``Lambda`` layer whose
``function`` or ``output_shape`` is a ``__lambda__`` dict (base64 bytecode),
and any layer whose ``activation`` is a ``__lambda__`` dict.  Named functions,
named activations and every other layer type are clean.

Layers are found wherever the JSON nests them: ``layers``, the ``layer`` of a
wrapper, and the ``config`` of a nested model.
"""
from __future__ import annotations

import io
import json

from modelgate.report import Finding


def scan_h5_bytes(data, location, fail_closed=False):
    import h5py  # lazy
    try:
        with h5py.File(io.BytesIO(data), "r") as fh:
            raw = fh.attrs.get("model_config")
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"could not read HDF5: {exc}")] if fail_closed else []
    if raw is None:
        return []  # a weights-only or non-Keras HDF5 file: nothing to run
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        config = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"invalid model_config JSON: {exc}")] if fail_closed else []
    findings: list[Finding] = []
    _walk(config, f"{location}:model_config", findings)
    return findings


def scan_keras_v3(zf, location, fail_closed=False):
    names = zf.namelist()
    target = "config.json" if "config.json" in names else None
    if target is None:
        for n in names:
            if n.endswith("config.json"):
                target = n
                break
    if target is None:
        return []
    try:
        raw = zf.read(target)
        config = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"invalid {target}: {exc}")] if fail_closed else []
    findings: list[Finding] = []
    _walk(config, f"{location}!{target}", findings)
    return findings


def _is_lambda_dict(obj) -> bool:
    return isinstance(obj, dict) and obj.get("class_name") == "__lambda__"


def _walk(node, loc, findings, seen=None):
    if seen is None:
        seen = set()
    if id(node) in seen:
        return
    if isinstance(node, dict):
        seen.add(id(node))
        cls = node.get("class_name")
        cfg = node.get("config") if isinstance(node.get("config"), dict) else {}

        if cls == "Lambda":
            fn = cfg.get("function")
            if _is_lambda_dict(fn):
                findings.append(Finding("code-execution", f"{loc}:Lambda.function",
                                        "Lambda function carries serialized bytecode"))
            os_ = cfg.get("output_shape")
            if _is_lambda_dict(os_):
                findings.append(Finding("code-execution", f"{loc}:Lambda.output_shape",
                                        "Lambda output_shape carries serialized bytecode"))

        # any layer whose activation is a __lambda__ dict
        act = cfg.get("activation")
        if _is_lambda_dict(act):
            findings.append(Finding("code-execution", f"{loc}:{cls or 'layer'}.activation",
                                    "activation carries serialized bytecode"))

        for key, val in node.items():
            _walk(val, f"{loc}.{key}", findings, seen)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _walk(item, f"{loc}[{i}]", findings, seen)
