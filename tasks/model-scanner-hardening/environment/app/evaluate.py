#!/usr/bin/env python3
"""Run the ModelGate scanner over /app/samples the way the registry does and
compare its verdicts with /app/samples/labels.json.

Usage: python3 /app/evaluate.py [--samples DIR] [--scanner /app/modelgate/cli.py]

The scanner is invoked as a separate process per artifact, with the same
command line the registry uses, so this script never imports the scanner.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

KINDS = {"code-execution", "process", "file-access", "network", "unsupported"}


def run_scanner(cli, artifact, timeout=60):
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, "report.json")
        cmd = [sys.executable, "-I", cli, "scan", "--output", report, artifact]
        started = time.monotonic()
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "timeout", time.monotonic() - started
        seconds = time.monotonic() - started
        if proc.returncode != 0:
            return None, f"exit {proc.returncode}: {proc.stderr.strip()[-200:]}", seconds
        try:
            with open(report, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as exc:
            return None, f"unreadable report: {exc}", seconds
        return doc, None, seconds


def check_schema(doc, artifact):
    if not isinstance(doc, dict) or doc.get("scanner") != "modelgate":
        return "top-level object with scanner == modelgate expected"
    results = doc.get("results")
    if not isinstance(results, list) or len(results) != 1:
        return "results must hold exactly one entry"
    entry = results[0]
    if not isinstance(entry, dict) or set(entry) != {"path", "verdict", "findings"}:
        return "entry must have exactly path, verdict, findings"
    if entry["path"] != artifact:
        return "path must repeat the argument exactly"
    if entry["verdict"] not in ("malicious", "clean"):
        return "verdict must be malicious or clean"
    findings = entry["findings"]
    if not isinstance(findings, list):
        return "findings must be a list"
    if (entry["verdict"] == "malicious") != (len(findings) > 0):
        return "findings must be non-empty iff verdict is malicious"
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != {"kind", "location", "detail"}:
            return "each finding needs exactly kind, location, detail"
        if not all(isinstance(finding[k], str) for k in ("kind", "location", "detail")):
            return "finding fields must be strings"
        if finding["kind"] not in KINDS:
            return f"unknown finding kind {finding['kind']!r}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", default="/app/samples")
    parser.add_argument("--scanner", default="/app/modelgate/cli.py")
    args = parser.parse_args()

    with open(os.path.join(args.samples, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)

    rows = []
    for name in sorted(labels):
        expected = labels[name]
        artifact = os.path.join(args.samples, name)
        doc, error, seconds = run_scanner(args.scanner, artifact)
        if error is None:
            error = check_schema(doc, artifact)
        if error is None:
            entry = doc["results"][0]
            verdict = entry["verdict"]
            kinds = sorted({f["kind"] for f in entry["findings"]})
        else:
            verdict = "invalid"
            kinds = []
        rows.append((name, expected, verdict, kinds, error, seconds))

    width = max(len(r[0]) for r in rows)
    print(f"{'artifact':<{width}}  expected   reported   ok  seconds  findings")
    for name, expected, verdict, kinds, error, seconds in rows:
        ok = "yes" if verdict == expected else "NO"
        extra = ",".join(kinds) if error is None else f"ERROR: {error}"
        print(f"{name:<{width}}  {expected:<9}  {verdict:<9}  {ok:<3} {seconds:7.2f}  {extra}")

    malicious = [r for r in rows if r[1] == "malicious"]
    clean = [r for r in rows if r[1] == "clean"]
    caught = sum(1 for r in malicious if r[2] == "malicious")
    false_pos = sum(1 for r in clean if r[2] != "clean")
    invalid = sum(1 for r in rows if r[4] is not None)
    print()
    print(f"recall: {caught}/{len(malicious)} malicious artifacts reported malicious")
    print(f"false positives: {false_pos}/{len(clean)} clean artifacts not reported clean")
    print(f"invalid reports: {invalid}")
    print(f"slowest scan: {max(r[5] for r in rows):.2f} s")
    return 0 if caught == len(malicious) and false_pos == 0 and invalid == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
