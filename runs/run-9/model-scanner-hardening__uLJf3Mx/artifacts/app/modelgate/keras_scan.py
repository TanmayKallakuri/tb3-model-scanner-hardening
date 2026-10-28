"""Inspect Keras configuration data, without importing Keras or its objects."""
from __future__ import annotations

import io
import json

from modelgate.report import Finding


def _bytecode(value):
    if not isinstance(value, dict) or value.get("class_name") != "__lambda__":
        return False
    cfg = value.get("config")
    return isinstance(cfg, dict) and ("code" in cfg or "value" in cfg)


def scan_config(config, location: str) -> list[Finding]:
    if isinstance(config, (bytes, str)):
        config = json.loads(config)
    findings = []
    todo = [(config, location)]
    while todo:
        obj, loc = todo.pop()
        if isinstance(obj, list):
            todo.extend((v, f"{loc}[{i}]") for i, v in enumerate(obj))
        elif isinstance(obj, dict):
            cfg = obj.get("config")
            if isinstance(cfg, dict) and "class_name" in obj:
                fields = ["activation"]
                if obj["class_name"] == "Lambda":
                    fields += ["function", "output_shape"]
                for field in fields:
                    v = cfg.get(field)
                    # Keras 2 used a [code, defaults, closure] triple with a
                    # separate function_type marker instead of __lambda__.
                    legacy = (obj["class_name"] == "Lambda" and
                              field in ("function", "output_shape") and
                              cfg.get(field + "_type") == "lambda" and
                              isinstance(v, (list, tuple)) and len(v) == 3)
                    if _bytecode(v) or legacy:
                        findings.append(Finding("code-execution", f"{loc}.config.{field}",
                                                "serialized Python bytecode"))
            todo.extend((v, f"{loc}.{k}") for k, v in obj.items()
                        if isinstance(v, (dict, list)))
    return findings


def scan_h5(source, location: str) -> list[Finding]:
    import h5py

    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with h5py.File(source, "r") as fh:
        raw = fh.attrs.get("model_config")
    return [] if raw is None else scan_config(raw, location + ":model_config")
