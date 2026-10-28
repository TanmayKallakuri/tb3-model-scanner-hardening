"""Read Keras configuration as data, without importing Keras or loading code."""
from __future__ import annotations

import io
import json
import os
import platform

from modelgate.limits import Budget
from modelgate.report import Finding


def scan_config(config, location: str, budget=None) -> list[Finding]:
    budget = budget or Budget()
    if isinstance(config, (bytes, str)):
        config = json.loads(config)
    findings = []
    pending = [(config, location)]
    while pending:
        value, loc = pending.pop()
        budget.check()
        if isinstance(value, dict):
            if value.get("class_name") == "Lambda":
                cfg = value.get("config", {})
                fn = cfg.get("function") if isinstance(cfg, dict) else None
                if (isinstance(fn, dict) and fn.get("class_name") == "__lambda__"
                        and isinstance(fn.get("config"), dict) and fn["config"].get("code")):
                    findings.append(Finding("code-execution", loc + ".config.function", "Lambda function carries serialized Python bytecode"))
                # Legacy Keras 2 encodes the same bytecode as [code, defaults,
                # closure], paired with function_type = lambda.
                elif (isinstance(fn, (list, tuple)) and fn and fn[0]
                      and cfg.get("function_type") == "lambda"):
                    findings.append(Finding("code-execution", loc + ".config.function", "Lambda function carries serialized Python bytecode"))
            pending.extend((v, f"{loc}.{k}") for k, v in value.items() if isinstance(v, (dict, list)))
        elif isinstance(value, list):
            pending.extend((v, f"{loc}[{i}]") for i, v in enumerate(value) if isinstance(v, (dict, list)))
    return findings


def scan_h5(source, location: str, budget=None) -> list[Finding]:
    # h5py's compiled modules index platform.uname(). CPython computes its
    # lazy `processor` field by launching `uname -p`, which is forbidden in a
    # scan. Supply the kernel's machine string before that optional probe.
    # This affects only platform metadata; artifact parsing stays read-only.
    platform.uname().__dict__.setdefault("processor", os.uname().machine)
    import h5py

    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with h5py.File(source, "r") as fh:
        raw = fh.attrs.get("model_config")
    return [] if raw is None else scan_config(raw, location + ":model_config", budget)
