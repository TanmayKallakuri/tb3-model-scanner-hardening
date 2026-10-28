#!/usr/bin/env python3
"""Shortcut audit for the hidden corpus (SPEC 5.5).

Every cheap classifier below must score reward 0 (that is, it must get at
least one verdict wrong) on the hidden corpus, while the oracle scores 1.  The
audit guards against label leakage through features unrelated to the payload:
size, extension, container metadata, opcode style, canary strings, parse
errors.  Run inside the dev image:

    MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/repo" mgdev:latest \
        python /repo/scripts/shortcut-audit/audit.py [--baseline /repo/tasks/model-scanner-hardening/environment/app/modelgate/cli.py]

Writes scripts/shortcut-audit/report.json and prints one line per classifier.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import pathlib
import pickletools
import re
import subprocess
import sys
import tempfile
import zipfile

REPO = pathlib.Path("/repo")
TASK = REPO / "tasks" / "model-scanner-hardening"
sys.path.insert(0, str(TASK / "tests"))

from gen import corpus  # noqa: E402
from gen import payloads  # noqa: E402

DANGEROUS_NAMES = [
    b"os\nsystem", b"posix\nsystem", b"subprocess", b"builtins\neval", b"builtins\nexec", b"__builtin__\neval",
    b"socket", b"urllib", b"ftplib", b"smtplib", b"shutil", b"marshal", b"timeit", b"runpy", b"bdb", b"pydoc",
    b"importlib", b"__import__", b"getattr", b"attrgetter", b"methodcaller", b"partial", b"pickle\nloads",
    b"ReadFile", b"WriteFile", b"PyFunc", b"MatchingFiles", b"__lambda__", b"logging.config", b"pathlib",
]
CANARY_BYTES = [s.encode() for s in payloads.INERT_STRINGS] + [b"/run/canary", b"1023", b"touch "]


def item_bytes(item) -> bytes:
    if item.ext == "":
        return b"".join(item.files[k] for k in sorted(item.files))
    return item.files["artifact"]


def features(item) -> dict:
    data = item_bytes(item)
    feats = {
        "size": len(data),
        "ext": item.ext,
        "format": item.format,
        "entropy": entropy(data[:65536]),
        "canary_grep": any(s in data for s in CANARY_BYTES),
        "dangerous_grep": any(s in data for s in DANGEROUS_NAMES),
        "parse_error": False,
        "frame": b"\x95" in data[:16],
        "memoize_count": 0,
        "global_count": 0,
        "n_streams": 0,
        "zip_members": 0,
    }
    if item.ext == "":
        return feats
    if data[:2] == b"PK" or b"PK\x05\x06" in data[-65557:]:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                feats["zip_members"] = len(z.namelist())
        except zipfile.BadZipFile:
            feats["parse_error"] = True
    if data[:1] == b"\x80" or data[:1] in b"(c]}IS" or item.format in ("pickle", "torch-legacy"):
        try:
            pos = 0
            streams = 0
            while pos < len(data) and streams < 8:
                last = None
                for op, arg, p in pickletools.genops(data[pos:]):
                    feats["memoize_count"] += op.name in ("MEMOIZE", "BINPUT", "PUT", "LONG_BINPUT")
                    feats["global_count"] += op.name in ("GLOBAL", "STACK_GLOBAL", "INST")
                    if op.name == "STOP":
                        last = p + 1
                        break
                if last is None:
                    break
                streams += 1
                pos += last
                if data[pos:pos + 1] not in (b"\x80", b"(", b"c", b"]", b"}", b"I", b"S", b"V", b"N", b"l", b"d", b"i", b"L"):
                    break
            feats["n_streams"] = streams
        except Exception:
            feats["parse_error"] = True
    return feats


def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = collections.Counter(data)
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def run_scanner(cli: str, root: pathlib.Path, item, name: str) -> str:
    path = root / name
    with tempfile.TemporaryDirectory() as tmp:
        report = os.path.join(tmp, "report.json")
        proc = subprocess.run([sys.executable, "-I", cli, "scan", "--output", report, str(path)], cwd=tmp, capture_output=True, timeout=120)
        try:
            with open(report, encoding="utf-8") as f:
                return json.load(f)["results"][0]["verdict"]
        except Exception:
            return "invalid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", help="path to the baseline cli.py (optional)")
    ap.add_argument("--oracle", help="path to the oracle cli.py (optional)")
    ap.add_argument("--seed", type=int, default=corpus.SEED)
    args = ap.parse_args()

    items = corpus.generate(args.seed)
    labels = [it.label for it in items]
    feats = [features(it) for it in items]
    n_mal = sum(l == "malicious" for l in labels)
    print(f"items: {len(items)} malicious: {n_mal} clean: {len(items) - n_mal}")

    def score(pred: list[str]) -> dict:
        fn = sum(1 for p, l in zip(pred, labels) if l == "malicious" and p != "malicious")
        fp = sum(1 for p, l in zip(pred, labels) if l == "clean" and p != "clean")
        return {"fn": fn, "fp": fp, "reward": 1 if fn == 0 and fp == 0 else 0, "accuracy": round(1 - (fn + fp) / len(labels), 4)}

    classifiers = {}
    classifiers["all-clean"] = ["clean"] * len(items)
    classifiers["all-malicious"] = ["malicious"] * len(items)
    classifiers["canary-grep"] = ["malicious" if f["canary_grep"] else "clean" for f in feats]
    classifiers["dangerous-name-grep"] = ["malicious" if f["dangerous_grep"] else "clean" for f in feats]
    classifiers["parse-error-or-multistream"] = ["malicious" if f["parse_error"] or f["n_streams"] > 1 else "clean" for f in feats]
    classifiers["no-frame-no-memoize"] = ["malicious" if (not f["frame"] and f["memoize_count"] == 0 and f["format"] == "pickle") else "clean" for f in feats]
    sizes = sorted(f["size"] for f in feats)
    best = None
    for cut in sizes:
        pred = ["malicious" if f["size"] <= cut else "clean" for f in feats]
        s = score(pred)
        if best is None or s["accuracy"] > best[1]["accuracy"]:
            best = (cut, s)
        pred = ["malicious" if f["size"] > cut else "clean" for f in feats]
        s = score(pred)
        if s["accuracy"] > best[1]["accuracy"]:
            best = (-cut, s)
    classifiers[f"size-threshold({best[0]})"] = ["malicious" if (f["size"] <= best[0] if best[0] >= 0 else f["size"] > -best[0]) else "clean" for f in feats]
    # extension majority rule fit on the corpus itself (an upper bound on what samples could teach)
    by_ext = collections.defaultdict(list)
    for f, l in zip(feats, labels):
        by_ext[f["ext"]].append(l)
    rule = {e: collections.Counter(ls).most_common(1)[0][0] for e, ls in by_ext.items()}
    classifiers["extension-majority(fit-on-corpus)"] = [rule[f["ext"]] for f in feats]
    # decision stump over cheap numeric features, fit on the corpus itself (upper bound)
    for key in ("entropy", "memoize_count", "global_count", "zip_members"):
        vals = sorted({f[key] for f in feats})
        bestk = None
        for cut in vals:
            for direction in (True, False):
                pred = ["malicious" if ((f[key] <= cut) == direction) else "clean" for f in feats]
                s = score(pred)
                if bestk is None or s["accuracy"] > bestk[1]["accuracy"]:
                    bestk = ((cut, direction), s, pred)
        classifiers[f"stump-{key}(fit-on-corpus)"] = bestk[2]

    root = None
    if args.baseline or args.oracle:
        root = pathlib.Path(tempfile.mkdtemp(prefix="audit-corpus-"))
        for it in items:
            name = corpus.item_name(it)
            if it.ext == "":
                for rel, data in it.files.items():
                    p = root / name / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(data)
            else:
                (root / name).write_bytes(it.files["artifact"])
    if args.baseline:
        classifiers["baseline-scanner"] = [run_scanner(args.baseline, root, it, corpus.item_name(it)) for it in items]
    if args.oracle:
        classifiers["oracle-scanner"] = [run_scanner(args.oracle, root, it, corpus.item_name(it)) for it in items]

    report = {}
    worst = None
    for name, pred in classifiers.items():
        s = score(pred)
        report[name] = s
        flag = "OK " if (s["reward"] == 0) != (name == "oracle-scanner") else "BAD"
        print(f"[{flag}] {name:42s} fn={s['fn']:3d} fp={s['fp']:3d} acc={s['accuracy']:.4f} reward={s['reward']}")
        if name != "oracle-scanner" and (worst is None or s["accuracy"] > worst[1]):
            worst = (name, s["accuracy"])
    print("strongest shortcut:", worst)
    (REPO / "scripts" / "shortcut-audit" / "report.json").write_text(json.dumps({"items": len(items), "classifiers": report, "strongest_shortcut": worst}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
