#!/usr/bin/env python3
"""ModelGate command line.

    python3 -I /app/modelgate/cli.py scan --output <report.json> <path> [<path> ...]

Writes one JSON report (see /app/docs/report-schema.md) and exits 0 once it
is written; exits 2 on a usage error.  ``-I`` keeps the script directory off
sys.path, so the package directory's parent is added explicitly below.
"""
import argparse
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelgate import __version__  # noqa: E402
from modelgate import dispatch  # noqa: E402
from modelgate.report import Finding, Result, to_json  # noqa: E402

# Keep a comfortable margin under the registry's 60s-per-artifact limit: if a
# scan runs away, abort it and fail closed rather than let the registry kill
# the process before any report is written.
_PER_ARTIFACT_SECONDS = 45


class _ScanTimeout(Exception):
    pass


def _deadline(_signum, _frame):
    raise _ScanTimeout()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="modelgate", description="static scanner for model artifacts")
    parser.add_argument("--version", action="version", version=f"modelgate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="scan one or more artifacts")
    scan.add_argument("--output", required=True, help="where to write the JSON report")
    scan.add_argument("paths", nargs="+", help="artifact files, or SavedModel directories")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _deadline)

    results = []
    for path in args.paths:
        try:
            if has_alarm:
                signal.alarm(_PER_ARTIFACT_SECONDS)
            results.append(dispatch.scan_path(path))
        except _ScanTimeout:
            results.append(Result(path, [Finding("unsupported", path, "scan exceeded time budget")]))
        except Exception as exc:  # noqa: BLE001 - a crash must still yield a verdict
            results.append(Result(path, [Finding("unsupported", path, f"scanner error: {type(exc).__name__}: {exc}")]))
        finally:
            if has_alarm:
                signal.alarm(0)

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(to_json(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
