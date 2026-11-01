"""Keras configuration inspection. No Keras deserializers are used."""
from __future__ import annotations

import io
import json

from modelgate.report import Finding


def _bytecode(value):
    return isinstance(value, dict) and value.get("class_name") == "__lambda__"


def scan_config(config, location: str) -> list[Finding]:
    if isinstance(config, (str, bytes)):
        config = json.loads(config)
    findings = []
    pending = [(config, location)]
    while pending:
        value, path = pending.pop()
        if isinstance(value, dict):
            body = value.get("config")
            if isinstance(body, dict) and isinstance(value.get("class_name"), str):
                for field in ("function", "output_shape") if value["class_name"] == "Lambda" else ():
                    if _bytecode(body.get(field)):
                        findings.append(Finding("code-execution", f"{path}/config/{field}", "serialized Lambda bytecode"))
                if _bytecode(body.get("activation")):
                    findings.append(Finding("code-execution", f"{path}/config/activation", "serialized activation bytecode"))
            pending.extend((child, f"{path}/{key}") for key, child in value.items() if isinstance(child, (dict, list)))
        elif isinstance(value, list):
            pending.extend((child, f"{path}[{i}]") for i, child in enumerate(value) if isinstance(child, (dict, list)))
    return findings


def scan_h5(source, location: str) -> list[Finding]:
    import h5py

    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with h5py.File(source, "r") as fh:
        raw = fh.attrs.get("model_config")
        if raw is None:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return scan_config(json.loads(raw), location)
