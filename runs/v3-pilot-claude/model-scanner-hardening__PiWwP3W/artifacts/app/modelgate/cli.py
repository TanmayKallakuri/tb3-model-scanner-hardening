#!/usr/bin/env python3
"""ModelGate command line.

    python3 -I /app/modelgate/cli.py scan --output <report.json> <path> [<path> ...]

Writes one JSON report (see /app/docs/report-schema.md) and exits 0 once it
is written; exits 2 on a usage error.  ``-I`` keeps the script directory off
sys.path, so the package directory's parent is added explicitly below.
"""
import sys

# The scanner must not write any file but its report: no bytecode caches.
sys.dont_write_bytecode = True
sys.setrecursionlimit(20000)

import argparse  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelgate import __version__  # noqa: E402
from modelgate.report import Finding, Result, to_json  # noqa: E402


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

    budget = 40.0 / max(1, len(args.paths)) if len(args.paths) > 1 else 40.0
    results = []
    _arm_watchdog(args.output, args.paths, results)
    for path in args.paths:
        try:
            from modelgate import dispatch
            results.append(dispatch.scan_path(path, max(budget, 5.0)))
        except BaseException as exc:  # noqa: BLE001 - a crash must still yield a verdict
            if isinstance(exc, KeyboardInterrupt):
                raise
            results.append(Result(path, [Finding("unsupported", path, f"scanner error: {type(exc).__name__}: {exc}")]))
    _write_report(args.output, results)
    return 0


def _write_report(output, results):
    with open(output, "w", encoding="utf-8") as fh:
        fh.write(to_json(results))


WATCHDOG_SECONDS = 55  # the registry allows 60 s per artifact


def _arm_watchdog(output, paths, results):
    """If analysis overruns, report the unfinished paths as unsupported and exit 0."""
    import signal

    def expire(signum, frame):
        done = list(results)
        for path in paths[len(done):]:
            done.append(Result(path, [Finding("unsupported", path, "analysis did not finish within the time limit")]))
        try:
            _write_report(output, done)
        finally:
            os._exit(0)

    try:
        signal.signal(signal.SIGALRM, expire)
        signal.alarm(WATCHDOG_SECONDS)
    except (AttributeError, ValueError, OSError):
        pass  # no SIGALRM on this platform


if __name__ == "__main__":
    sys.exit(main())
