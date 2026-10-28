"""Keras HDF5 scanning: Lambda layers in the model configuration."""
from __future__ import annotations

import json

from modelgate.report import Finding


def scan_h5(path: str, location: str) -> list[Finding]:
    import h5py  # lazy: only needed for Keras files

    with h5py.File(path, "r") as fh:
        raw = fh.attrs.get("model_config")
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    config = json.loads(raw)
    findings: list[Finding] = []
    for i, layer in enumerate(config.get("config", {}).get("layers", [])):
        if layer.get("class_name") == "Lambda":
            findings.append(Finding("code-execution", f"{location}:layers[{i}]", "Lambda layer"))
    return findings
