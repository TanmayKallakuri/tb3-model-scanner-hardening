"""Route an artifact to the right analyser and build its result.

A directory is a TensorFlow SavedModel; everything else is a file whose format
is decided by its content (``containers.scan_bytes``), not its extension.  Any
exception raised while scanning is turned by the CLI into a fail-closed
``unsupported`` verdict, so every artifact receives a verdict.
"""
from __future__ import annotations

import os

from modelgate import containers, savedmodel_scan
from modelgate.report import Result

_MAX_READ = 512 * 1024 * 1024


def scan_path(path: str) -> Result:
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    with open(path, "rb") as fh:
        data = fh.read(_MAX_READ)
    name = os.path.basename(path.rstrip("/")) or path
    return Result(path, containers.scan_bytes(data, name, depth=0, budget={"ops": 0}))
