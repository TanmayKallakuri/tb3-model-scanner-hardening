#!/usr/bin/env python3
"""ModelGate command line.

    python3 -I /app/modelgate/cli.py scan --output <report.json> <path> [<path> ...]

Writes one JSON report (see /app/docs/report-schema.md) and exits 0 once it
is written; exits 2 on a usage error.  ``-I`` keeps the script directory off
sys.path, so the package directory's parent is added explicitly below.
"""
import sys

# The registry treats any file the scanner writes other than its report as a
# failed scan.  Importing anything would otherwise drop .pyc caches (into
# modelgate/__pycache__, and into any dependency lacking them), so bytecode
# writing is disabled first thing -- before argparse or the package is
# imported.  Recompiling the few small modules from source each run costs only
# milliseconds.
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import os  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelgate import __version__  # noqa: E402
from modelgate import dispatch  # noqa: E402
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

    results = []
    for path in args.paths:
        try:
            results.append(dispatch.scan_path(path))
        except Exception as exc:  # noqa: BLE001 - a crash must still yield a verdict
            results.append(Result(path, [Finding("unsupported", path, f"scanner error: {type(exc).__name__}: {exc}")]))
    try:
        document = to_json(results)
    except Exception:  # noqa: BLE001 - serialising must never leave no report
        document = to_json([Result(p, [Finding("unsupported", p, "report serialisation failed")])
                            for p in args.paths])
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(document)
    return 0


if __name__ == "__main__":
    sys.exit(main())
