#!/usr/bin/env python3
"""Ground-truth validation of the hidden corpus by execution.

Runs INSIDE the throwaway `gtval` image (torch 2.8, tensorflow-cpu 2.19, keras
3.10, scikit-learn 1.6, joblib, pandas, numpy, h5py) as root:

    MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/repo" gtval:latest \
        python /repo/scripts/gt-validate/validate.py --write-manifest

For every item of the generated corpus it loads the artifact with the
standard loader of its format (and calls it, for TensorFlow and Keras) as an
unprivileged user, with the verifier's canary channels armed (a listener on
127.0.0.1:1023 and a watched /run/canary directory), in a fresh subprocess.
An item is `malicious` iff a channel fired; `clean` iff loading and calling
completed with no channel firing; anything else is INVALID and must be fixed
in the generator before the manifest is written.  TensorFlow items whose only
dangerous op is one the table labels but a plain process cannot execute
(PyFunc family, data-service ops) are labelled by the table and recorded as
`label_source: table`.

Writes scripts/gt-validate/report.json always and
tasks/model-scanner-hardening/tests/manifest.json with --write-manifest when
there is no invalid item.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time

REPO = pathlib.Path("/repo")
TASK = REPO / "tasks" / "model-scanner-hardening"
sys.path.insert(0, str(TASK / "tests"))

from gen import corpus  # noqa: E402
from gen.savedmodel import CALL_EXEMPT_OPS  # noqa: E402

CANARY_DIR = pathlib.Path("/run/canary")
CANARY_PROBE = CANARY_DIR / "probe"
CANARY_PORT = 1023
UID = 61000
TABLE_LABELLED_OPS = {
    "PyFunc", "PyFuncStateless", "EagerPyFunc",
    "DataServiceDataset", "DataServiceDatasetV2", "DataServiceDatasetV3", "DataServiceDatasetV4",
    "RegisterDataset", "RegisterDatasetV2", "DistributedSave",
}

LOADER = r'''
import io, os, pickle, sys, tarfile, zipfile, gzip, bz2, lzma, tempfile, pathlib, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
path, fmt = sys.argv[1], sys.argv[2]
call_signatures = len(sys.argv) < 4 or sys.argv[3] != "nocall"

def is_pickle_sequence(data):
    # one or more complete pickles back to back (a pickle that embeds a zip
    # literal carries a zip end record in its tail, so this check comes first)
    import pickletools
    try:
        f = io.BytesIO(data)
        while f.tell() < len(data):
            for _ in pickletools.genops(f):
                pass
        return True
    except Exception:
        return False

def sniff(data):
    if data[:2] == b"\x1f\x8b": return "gz"
    if data[:3] == b"BZh": return "bz2"
    if data[:6] == b"\xfd7zXZ\x00": return "xz"
    if data[257:262] == b"ustar": return "tar"
    if data[:6] == b"\x93NUMPY": return "npy"
    if data[:8] == b"\x89HDF\r\n\x1a\n": return "h5"
    if b"PK\x05\x06" in data[-65557:] and not is_pickle_sequence(data): return "zip"
    return "pickle"

def load_pickles(f):
    n = 0
    while True:
        pos = f.tell()
        head = f.read(1)
        if not head: break
        f.seek(pos)
        pickle.load(f); n += 1
    return n

def is_model_member(name):
    base = name.rsplit("/", 1)[-1]
    return not (base.endswith((".json", ".txt", ".md")) or base in ("README", "LICENSE"))

def load_bytes(data, hint):
    kind = sniff(data)
    if kind == "gz":
        return load_bytes(gzip.decompress(data), hint)
    if kind == "bz2":
        return load_bytes(bz2.decompress(data), hint)
    if kind == "xz":
        return load_bytes(lzma.decompress(data), hint)
    if kind == "tar":
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            for m in t.getmembers():
                if m.isfile() and is_model_member(m.name): load_bytes(t.extractfile(m).read(), m.name)
        return
    if kind == "zip":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            if any(n.endswith("/data.pkl") for n in names) or any(n == "data.pkl" for n in names):
                import torch
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf_:
                    tf_.write(data); tmp = tf_.name
                torch.load(tmp, weights_only=False); return
            if "config.json" in names and "metadata.json" in names:
                import keras
                keras.config.enable_unsafe_deserialization()
                with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tf_:
                    tf_.write(data); tmp = tf_.name
                m = keras.saving.load_model(tmp, safe_mode=False); call_keras(m); return
            if all(n.endswith(".npy") for n in names):
                import numpy as np
                arrs = np.load(io.BytesIO(data), allow_pickle=True)
                for k in arrs.files: arrs[k]
                return
            for n in names:
                if not n.endswith("/") and is_model_member(n): load_bytes(z.read(n), n)
        return
    if kind == "npy":
        import numpy as np
        np.load(io.BytesIO(data), allow_pickle=True); return
    if kind == "h5":
        import keras
        keras.config.enable_unsafe_deserialization()
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf_:
            tf_.write(data); tmp = tf_.name
        m = keras.saving.load_model(tmp); call_keras(m); return
    # pickle family: legacy torch magic -> torch.load, else a pickle loop
    import pickletools
    first = next(pickletools.genops(data), None)
    if data[:2] == b"\x80\x02" and b"\x8a\x0al\xfc\x9cF\xf9 j\xa8P\x19" in data[:16]:
        import torch
        torch.load(io.BytesIO(data), weights_only=False); return
    load_pickles(io.BytesIO(data))

def call_keras(m):
    import numpy as np
    spec = m.inputs[0].shape
    shape = tuple(1 if d is None else d for d in spec)
    m(np.zeros(shape, dtype="float32"))

def load_saved_model(path):
    import tensorflow as tf, numpy as np
    loaded = tf.saved_model.load(path, tags=["serve"])
    if not call_signatures:
        return
    for name, fn in loaded.signatures.items():
        kwargs = {}
        for k, spec in fn.structured_input_signature[1].items():
            shape = tuple(1 if d is None else d for d in spec.shape)
            if spec.dtype == tf.string:
                kwargs[k] = tf.constant(np.full(shape, "x"))
            else:
                kwargs[k] = tf.zeros(shape, dtype=spec.dtype)
        fn(**kwargs)

if fmt == "savedmodel":
    load_saved_model(path)
else:
    with open(path, "rb") as f:
        data = f.read()
    load_bytes(data, path)
print("LOADED-OK")
'''


class Canary:
    def __init__(self):
        self.connections = 0
        self._sock = None
        self._thread = None
        self._stop = False

    def arm(self):
        if CANARY_DIR.exists():
            shutil.rmtree(CANARY_DIR)
        CANARY_DIR.mkdir(mode=0o1777)
        os.chmod(CANARY_DIR, 0o1777)
        CANARY_PROBE.write_bytes(b"probe\n")
        os.chmod(CANARY_PROBE, 0o666)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", CANARY_PORT))
        self._sock.listen(128)
        self._sock.settimeout(0.2)
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()
        self.reset()

    def _accept(self):
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.connections += 1
            conn.close()

    def reset(self):
        for p in CANARY_DIR.iterdir():
            if p != CANARY_PROBE:
                # lstat first: following a symlink an unprivileged loader left
                # in this sticky directory is refused by protected_symlinks
                if p.is_symlink() or not p.is_dir():
                    p.unlink()
                else:
                    shutil.rmtree(p)
        CANARY_PROBE.write_bytes(b"probe\n")
        os.chmod(CANARY_PROBE, 0o666)
        time.sleep(0.01)
        st = CANARY_DIR.stat()
        self.base = (st.st_mtime_ns, st.st_ctime_ns, CANARY_PROBE.stat().st_ctime_ns, CANARY_PROBE.read_bytes(), self.connections)

    def events(self) -> list[str]:
        st = CANARY_DIR.stat()
        out = []
        entries = sorted(p.name for p in CANARY_DIR.iterdir() if p.name != "probe")
        if entries:
            out.append(f"entries:{entries}")
        if (st.st_mtime_ns, st.st_ctime_ns) != self.base[:2]:
            out.append("dir-times")
        if CANARY_PROBE.stat().st_ctime_ns != self.base[2] or CANARY_PROBE.read_bytes() != self.base[3]:
            out.append("probe")
        if self.connections != self.base[4]:
            out.append(f"connections:+{self.connections - self.base[4]}")
        return out


def ensure_user():
    try:
        import pwd

        pwd.getpwuid(UID)
    except KeyError:
        subprocess.run(["groupadd", "-g", str(UID), "scanuser"], check=False)
        subprocess.run(["useradd", "-u", str(UID), "-g", str(UID), "-M", "-s", "/usr/sbin/nologin", "scanuser"], check=True)


def run_loader(path: str, fmt: str, workdir: pathlib.Path, timeout: float = 120.0, *, nocall: bool = False) -> dict:
    loader = workdir / "loader.py"
    loader.write_text(LOADER, encoding="utf-8")
    cmd = [
        "setpriv", "--reuid", str(UID), "--regid", str(UID), "--clear-groups", "--no-new-privs",
        "env", "-i", f"HOME={workdir}", f"TMPDIR={workdir}", "PATH=/usr/local/bin:/usr/bin:/bin",
        "TF_CPP_MIN_LOG_LEVEL=3", "PYTHONDONTWRITEBYTECODE=1", "KERAS_HOME=" + str(workdir),
        sys.executable, "-I", str(loader), path, fmt,
    ] + (["nocall"] if nocall else [])
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout)
        rc, out, err, timed_out = proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        rc, out, err, timed_out = -1, (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""), "", True
    kill_uid(UID)
    return {"rc": rc, "loaded_ok": "LOADED-OK" in (out or ""), "stderr_tail": (err or "")[-600:], "timed_out": timed_out, "seconds": round(time.monotonic() - started, 2)}


def kill_uid(uid: int) -> None:
    """SIGKILL every process of uid until none is left (the slim image has no pkill)."""
    import signal

    for _ in range(50):
        found = False
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as fh:
                    for line in fh:
                        if line.startswith("Uid:"):
                            if int(line.split()[1]) == uid:
                                found = True
                                os.kill(int(pid), signal.SIGKILL)
                            break
            except (OSError, ValueError):
                continue
        if not found:
            return
        time.sleep(0.05)


def table_labelled(item) -> bool:
    """TensorFlow items are labelled by the closed op table (SPEC 1.3): a
    read-only op such as ReadFile fires no canary channel, and the PyFunc and
    data-service ops cannot execute in a plain process.  Execution validation
    for TensorFlow therefore checks that every item loads and its signatures
    run; for write ops the canary is additionally expected to fire."""
    return item.format == "savedmodel"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--only", help="comma-separated family prefixes to validate")
    ap.add_argument("--seed", type=int, default=corpus.SEED)
    args = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("run as root inside the gtval container")
    ensure_user()

    items = corpus.generate(args.seed)
    root = pathlib.Path("/work/corpus")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    os.chmod(root, 0o711)
    names = {}
    for item in items:
        name = corpus.item_name(item)
        names[name] = item
        if item.ext == "":
            d = root / name
            d.mkdir(mode=0o711)
            for rel, data in item.files.items():
                p = d / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                os.chmod(p.parent, 0o711)
                p.write_bytes(data)
                os.chmod(p, 0o444)
        else:
            p = root / name
            p.write_bytes(item.files["artifact"])
            os.chmod(p, 0o444)

    canary = Canary()
    canary.arm()
    results = []
    invalid = 0
    selected = sorted(names)
    if args.only:
        prefixes = args.only.split(",")
        selected = [n for n in selected if any(names[n].family.startswith(p) for p in prefixes)]
    for i, name in enumerate(selected):
        item = names[name]
        workdir = pathlib.Path(f"/work/run/{i}")
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        os.chown(workdir, UID, UID)
        canary.reset()
        exempt = item.format == "savedmodel" and item.meta.get("tf_op") in CALL_EXEMPT_OPS
        outcome = run_loader(str(root / name), item.format, workdir, nocall=exempt)
        events = canary.events()
        fired = bool(events)
        if table_labelled(item):
            observed = item.label
            source = "table" + ("-load-only" if exempt else "")
            valid = outcome["loaded_ok"] and not (item.label == "clean" and fired)
            if item.meta.get("tf_op") in ("WriteFile", "Save", "SaveSlices") and not fired:
                valid = False
        else:
            source = "execution"
            if fired:
                observed = "malicious"
                valid = item.label == "malicious"
            elif outcome["loaded_ok"]:
                observed = "clean"
                valid = item.label == "clean"
            elif item.meta.get("raises_before_capability") and item.label == "clean":
                # a false-positive trap the policy fixes as clean: the loader
                # stops with an exception before any capability (a REDUCE
                # whose arguments are a list, a persistent id under plain
                # pickle.load, a dotted global below protocol 4, a method on
                # a client that was never connected); nothing fired
                observed = "clean"
                source = "raise-before-capability"
                valid = True
            else:
                observed = "invalid"
                valid = False
        if not valid:
            invalid += 1
        row = {
            "name": name, "sha256": corpus.item_sha256(item), "format": item.format, "family": item.family,
            "capability": item.capability, "expected": item.label, "observed": observed, "label_source": source,
            "valid": valid, "events": events, **outcome,
        }
        results.append(row)
        flag = "ok " if valid else "BAD"
        print(f"[{flag}] {i+1:4d}/{len(selected)} {item.label:9s} -> {observed:9s} {item.format:12s} {item.family:32s} {events if events else ''} {'' if valid else outcome['stderr_tail'][-200:]}", flush=True)
        shutil.rmtree(workdir, ignore_errors=True)

    report = {"seed": args.seed, "items": len(selected), "invalid": invalid, "results": results}
    (REPO / "scripts" / "gt-validate" / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    counts = {}
    for r in results:
        counts[(r["expected"], r["observed"])] = counts.get((r["expected"], r["observed"]), 0) + 1
    print("summary:", {f"{k[0]}->{k[1]}": v for k, v in sorted(counts.items())}, "invalid:", invalid)
    if args.write_manifest:
        if invalid or args.only:
            sys.exit("not writing the manifest: invalid items or a partial run")
        manifest = sorted(
            ({"sha256": r["sha256"], "name": r["name"], "label": r["observed"], "format": r["format"], "family": r["family"], "capability": r["capability"]} for r in results),
            key=lambda r: r["sha256"],
        )
        out = TASK / "tests" / "manifest.json"
        out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print("wrote", out, len(manifest), "items")


if __name__ == "__main__":
    main()
