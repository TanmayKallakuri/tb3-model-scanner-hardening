#!/usr/bin/env python3
"""ModelGate command line.

    python3 -I /app/modelgate/cli.py scan --output <report.json> <path> [<path> ...]

Writes one JSON report (see /app/docs/report-schema.md) and exits 0 once it
is written; exits 2 on a usage error.  ``-I`` keeps the script directory off
sys.path, so the package directory's parent is added explicitly below.

The scan never runs anything from the artifact and writes no file other
than the report (bytecode caching is disabled before anything is imported).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

from modelgate import __version__  # noqa: E402

TIME_BUDGET = 48.0  # seconds for all paths of one invocation (registry limit: 60)
HARD_LIMIT = 56.0  # a native call that never returns: exit, leaving the fail-closed report


class _Timeout(BaseException):
    """Raised by SIGALRM; not an Exception, so nothing swallows it."""


def _on_alarm(signum, frame):  # pragma: no cover - only on pathological input
    raise _Timeout()


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

    from modelgate.report import Finding, Result, to_json

    started = time.monotonic()
    results = []

    def write(final_results):
        with open(args.output, "w", encoding="utf-8", errors="surrogateescape") as fh:
            fh.write(to_json(final_results))

    def provisional(done):
        # fail-closed placeholder for the paths not scanned yet, so that a
        # report exists even if a native library never returns
        pending = [Result(p, [Finding("unsupported", p, "scan did not complete")])
                   for p in args.paths[len(done):]]
        write(done + pending)

    provisional([])
    try:
        import faulthandler

        faulthandler.dump_traceback_later(HARD_LIMIT, exit=True)
    except Exception:  # noqa: BLE001
        faulthandler = None
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 12000))
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
    except (ValueError, AttributeError, OSError):
        pass
    for i, path in enumerate(args.paths):
        remaining = TIME_BUDGET - (time.monotonic() - started)
        share = max(1.0, remaining / max(1, len(args.paths) - i))
        deadline = time.monotonic() + share * 0.9
        try:
            signal.setitimer(signal.ITIMER_REAL, share)
        except (ValueError, AttributeError, OSError):
            pass
        try:
            from modelgate import dispatch

            results.append(dispatch.scan_path(path, deadline=deadline))
        except _Timeout:
            results.append(Result(path, [Finding("unsupported", path, "scan did not finish within the time limit")]))
        except BaseException as exc:  # noqa: BLE001 - a crash must still yield a verdict
            if isinstance(exc, KeyboardInterrupt):
                raise
            msg = f"scanner error: {type(exc).__name__}: {exc}"[:300].encode("utf-8", "backslashreplace").decode()
            results.append(Result(path, [Finding("unsupported", path, msg)]))
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except (ValueError, AttributeError, OSError):
                pass
        if i + 1 < len(args.paths):
            provisional(results)
    write(results)
    if faulthandler is not None:
        faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == "__main__":
    sys.exit(main())
