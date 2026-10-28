"""Read Keras configuration as data, without importing Keras or bytecode."""
from __future__ import annotations

import io
import json
from modelgate.report import Finding


def scan_config(config, location: str) -> list[Finding]:
    if isinstance(config, (str, bytes)):
        config = json.loads(config)
    findings = []
    pending = [(config, location)]
    while pending:
        value, path = pending.pop()
        if isinstance(value, dict):
            if value.get("class_name") == "Lambda":
                cfg = value.get("config", {})
                fn = cfg.get("function") if isinstance(cfg, dict) else None
                if (isinstance(fn, dict) and fn.get("class_name") == "__lambda__"
                        and isinstance(fn.get("config"), dict) and "code" in fn["config"]):
                    findings.append(Finding("code-execution", path + ".config.function", "Lambda function carries serialized Python bytecode"))
                # Keras 2's HDF5 form: [code, defaults, closure].
                elif isinstance(fn, (list, tuple)) and cfg.get("function_type") == "lambda" and fn:
                    findings.append(Finding("code-execution", path + ".config.function", "Lambda function carries serialized Python bytecode"))
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    pending.append((child, path + "." + key))
        elif isinstance(value, list):
            pending.extend((child, f"{path}[{i}]") for i, child in enumerate(value) if isinstance(child, (dict, list)))
    return findings


def scan_h5(source, location: str) -> list[Finding]:
    # h5py's native extension indexes platform.uname(), which on Python 3.12
    # eagerly evaluates its lazy processor field and otherwise spawns uname -p.
    # Supply that informational field from the in-process OS query first.
    import os
    import platform
    info = platform.uname()
    info.__dict__["processor"] = os.uname().machine
    import h5py  # Only the metadata reader; no model deserialization.
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with h5py.File(source, "r") as fh:
        raw = fh.attrs.get("model_config")
    return [] if raw is None else scan_config(raw, location + ":model_config")
