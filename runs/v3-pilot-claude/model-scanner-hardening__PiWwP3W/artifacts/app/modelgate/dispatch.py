"""Scan one artifact path and build its result."""
from __future__ import annotations

import os
import time

from modelgate.report import Finding, Result

# The registry allows 60 s per artifact; stop analysing well before that and
# fail closed.
TIME_BUDGET = 40.0
MAX_FILE = 2 * 1024 * 1024 * 1024


def scan_path(path: str, budget: float = TIME_BUDGET) -> Result:
    from modelgate import loaders, savedmodel_scan
    from modelgate.pickle_vm import Context, PickleStop, ScanAbort

    ctx = Context(time.monotonic() + budget)
    name = os.path.basename(path.rstrip("/")) or path
    try:
        if os.path.isdir(path):
            savedmodel_scan.scan_dir(ctx, path, name + "/")
        elif not os.path.isfile(path):
            return Result(path, [Finding("unsupported", path, "path is neither a file nor a directory")])
        else:
            size = os.path.getsize(path)
            if size > MAX_FILE:
                return Result(path, [Finding("unsupported", name, f"file too large to analyse ({size} bytes)")])
            with open(path, "rb") as fh:
                data = fh.read()
            if not data:
                return Result(path, [Finding("unsupported", name, "empty file")])
            loaders.scan_blob(ctx, data, name, name, 0, top=True)
    except PickleStop as exc:
        ctx.note(f"{name}: loading stops: {exc}")  # the loader raises; what ran before is recorded
    except ScanAbort as exc:
        ctx.report("unsupported", name, f"analysis incomplete: {exc}")
    except RecursionError:
        ctx.report("unsupported", name, "analysis incomplete: structure nested too deeply")
    except MemoryError:
        ctx.report("unsupported", name, "analysis incomplete: out of memory")
    return Result(path, _ordered(ctx.findings))


def _ordered(findings):
    """Capability findings first; drop 'unsupported' when a capability was found."""
    real = [f for f in findings if f.kind != "unsupported"]
    return real if real else list(findings)
