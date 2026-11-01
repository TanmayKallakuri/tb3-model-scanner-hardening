"""Scan one artifact path and build its result."""
from __future__ import annotations

import os
import time

from modelgate.report import Finding, Result

TIME_BUDGET = 40.0  # seconds of analysis per artifact (the registry allows 60)
MAX_FILE = 2 << 30


def _finalise(path, findings):
    real = [f for f in findings if f.kind != "unsupported"]
    if real:
        seen = set()
        out = []
        for f in real:
            key = (f.kind, f.location, f.detail)
            if key not in seen:
                seen.add(key)
                out.append(f)
        return Result(path, out)
    unsupported = [f for f in findings if f.kind == "unsupported"]
    if unsupported:
        first = unsupported[0]
        detail = first.detail
        if len(unsupported) > 1:
            detail += f" (and {len(unsupported) - 1} more)"
        return Result(path, [Finding("unsupported", first.location, detail)])
    return Result(path, [])


def scan_path(path: str) -> Result:
    from modelgate.formats import Scanner
    from modelgate.semantics import Budget

    deadline = time.monotonic() + TIME_BUDGET
    scanner = Scanner(deadline)
    name = os.path.basename(os.path.normpath(path)) or path
    try:
        if os.path.isdir(path):
            scanner.scan_dir(path, name)
        elif os.path.isfile(path):
            size = os.path.getsize(path)
            if size > MAX_FILE:
                scanner.unsupported(name, f"artifact too large to analyse ({size} bytes)")
            else:
                with open(path, "rb") as fh:
                    data = fh.read()
                scanner.scan_bytes(data, name, name, 0)
        else:
            scanner.unsupported(name, "artifact is not a file or directory")
    except Budget as exc:
        scanner.unsupported(name, f"analysis did not finish: {exc}")
    except RecursionError:
        scanner.unsupported(name, "artifact structure nested too deeply to analyse")
    except MemoryError:
        scanner.unsupported(name, "artifact too large to analyse in memory")
    except Exception as exc:  # noqa: BLE001 - a crash must still yield a verdict
        scanner.unsupported(name, f"scanner error: {type(exc).__name__}: {exc}")
    return _finalise(path, scanner.findings)
