"""Inspect serialized Keras configuration, never deserialize a Keras object."""
from __future__ import annotations

import json

from modelgate.report import Finding


def _bytecode(value):
    if not isinstance(value, dict) or value.get("class_name") != "__lambda__":
        return False
    config = value.get("config")
    return isinstance(config, dict) and ("code" in config or "value" in config)


def scan_config(config, location: str) -> list[Finding]:
    if isinstance(config, (str, bytes)):
        config = json.loads(config)
    findings = []
    todo = [(config, location)]
    while todo:
        value, path = todo.pop()
        if isinstance(value, list):
            todo.extend((v, f"{path}[{i}]") for i, v in enumerate(value))
        elif isinstance(value, dict):
            cfg = value.get("config")
            if isinstance(cfg, dict) and isinstance(value.get("class_name"), str):
                fields = ["activation"]
                if value["class_name"] == "Lambda":
                    fields.extend(("function", "output_shape"))
                for field in fields:
                    if _bytecode(cfg.get(field)):
                        findings.append(Finding("code-execution", f"{path}/config/{field}",
                                                f"{value['class_name']} {field} carries serialized bytecode"))
                    # Legacy HDF5 saves Lambda functions as a (code, defaults,
                    # closure) triple and a separate function_type marker.
                    elif (value["class_name"] == "Lambda" and field in ("function", "output_shape")
                          and cfg.get(field + "_type") == "lambda"
                          and isinstance(cfg.get(field), (list, tuple))):
                        findings.append(Finding("code-execution", f"{path}/config/{field}",
                                                f"Lambda {field} carries serialized bytecode"))
            todo.extend((v, f"{path}/{k}") for k, v in value.items() if isinstance(v, (dict, list)))
    return findings


def scan_h5(source, location: str) -> list[Finding]:
    from modelgate.parser_env import prepare
    prepare()
    import h5py

    # Read only the root attribute: no datasets, external links or model loaders.
    with h5py.File(source, "r") as fh:
        raw = fh.attrs.get("model_config")
    return [] if raw is None else scan_config(raw, location + ":model_config")
