"""Top-level dispatch: choose a scanner by the artifact, build the Result.

A directory is a TensorFlow SavedModel.  A file's format is decided by its
content (see :mod:`modelgate.container`), not its extension.  An artifact the
scanner cannot analyse still receives a verdict: the policy is fail-closed, so
it is reported ``malicious`` with a single ``unsupported`` finding.
"""
from __future__ import annotations

import os

from modelgate import container, savedmodel_scan
from modelgate.report import Finding, Result

_MAX_FILE = 2 * 1024 * 1024 * 1024  # 2 GiB safety ceiling


def scan_path(path: str) -> Result:
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    if not os.path.exists(path):
        return Result(path, [Finding("unsupported", path, "artifact path does not exist")])
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    if size > _MAX_FILE:
        return Result(path, [Finding("unsupported", path, "artifact too large to scan")])
    with open(path, "rb") as fh:
        data = fh.read()
    findings = container.analyse_blob(data, os.path.basename(path), depth=0, toplevel=True)
    return Result(path, findings)
