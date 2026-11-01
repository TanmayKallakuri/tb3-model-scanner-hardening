"""Scan one artifact path and build its result."""
from __future__ import annotations

import os

from modelgate import formats, savedmodel_scan
from modelgate.interp import Interp
from modelgate.report import Finding, Result
from modelgate.symbolic import Exhausted, LoadError

MAX_FILE = 2 * 1024 * 1024 * 1024


def _read(path):
    with open(path, "rb") as fh:
        return fh.read(MAX_FILE)


def scan_path(path: str, deadline=None) -> Result:
    from modelgate import containers

    containers.TOTAL_OUT[0] = 512 * 1024 * 1024
    I = Interp(deadline=deadline)
    loc = os.path.basename(os.path.normpath(path)) or path
    try:
        if os.path.isdir(path):
            if not savedmodel_scan.scan_saved_model_dir(I, path, loc):
                _scan_dir(I, path, loc)
        else:
            formats.analyze_blob(I, _read(path), loc)
    except Exhausted as exc:
        I.finding("unsupported", f"analysis incomplete: {exc}", loc)
    except RecursionError:
        I.finding("unsupported", "analysis incomplete: structure nested too deeply", loc)
    except LoadError:
        pass  # the artifact's own loader fails at this point
    if not I.findings and I.internal_errors:
        # the emulator could not follow part of the artifact: fail closed
        I.finding("unsupported", f"analysis incomplete: {I.internal_errors[0]}", loc)
    findings = [Finding(kind, _text(location), _text(detail)) for kind, location, detail in I.findings]
    return Result(path, findings)


def _text(s) -> str:
    """Valid Unicode for the report (payload strings may hold surrogates)."""
    s = str(s)
    if len(s) > 500:
        s = s[:497] + "..."
    return s.encode("utf-8", "backslashreplace").decode("utf-8")


def _scan_dir(I, path, loc):
    """A directory that is not a SavedModel: every file by its content (a
    SavedModel found below is scanned as one)."""
    for root, dirs, files in os.walk(path):
        dirs.sort()
        rel = os.path.relpath(root, path)
        if savedmodel_scan.scan_saved_model_dir(I, root, f"{loc}/{rel}"):
            dirs[:] = []
            continue
        for name in sorted(files):
            full = os.path.join(root, name)
            if os.path.islink(full) or not os.path.isfile(full):
                continue
            formats.analyze_blob(I, _read(full), f"{loc}/{os.path.relpath(full, path)}")
