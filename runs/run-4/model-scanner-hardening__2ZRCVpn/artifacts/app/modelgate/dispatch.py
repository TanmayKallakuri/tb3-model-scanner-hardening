"""Top-level dispatch: choose SavedModel (a directory) or a file artifact.

The path argument is either a directory (a TensorFlow SavedModel) or a file.
File formats and containers are decided by content, not extension, inside
``container.analyze_artifact``.
"""
from __future__ import annotations

import os

from modelgate import container, savedmodel_scan
from modelgate.report import Finding, Result

MAX_FILE_BYTES = 1024 * 1024 * 1024


def scan_path(path: str) -> Result:
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    if not os.path.exists(path):
        return Result(path, [Finding("unsupported", path, "path does not exist")])
    try:
        size = os.path.getsize(path)
        if size > MAX_FILE_BYTES:
            return Result(path, [Finding("unsupported", path,
                                         "artifact larger than size limit")])
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return Result(path, [Finding("unsupported", path, f"cannot read: {exc}")])

    name = os.path.basename(path.rstrip("/")) or path
    findings = container.analyze_artifact(data, name, nest=0, top_level=True)
    return Result(path, findings)
