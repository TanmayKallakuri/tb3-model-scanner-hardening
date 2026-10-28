#!/usr/bin/env python3
"""Prove, with the real frameworks, that every SavedModel placement and every
Keras injection point the generator uses produces a loadable artifact whose
injected construct executes.  Runs inside the gtval image:

    MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/repo" gtval:latest \
        python /repo/scripts/gt-validate/check_tfkeras.py

The injected TensorFlow op is WriteFile (observable: it creates a file) or
ReadFile (observable through the data-path wiring); the Keras lambda writes a
marker file when the layer is called.  Everything stays under /tmp of the
throwaway container.
"""
import json
import os
import pathlib
import random
import shutil
import sys
import tempfile
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/repo/tasks/model-scanner-hardening/tests")
from gen import keras_files, savedmodel  # noqa: E402

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

# the policy loader: Keras with unsafe deserialization enabled (h5 has no safe_mode argument)
keras.config.enable_unsafe_deserialization()

MARK = pathlib.Path("/tmp/gtcheck")
results = []


def write_dir(files: dict, root: pathlib.Path):
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def call_signatures(loaded):
    for name, fn in loaded.signatures.items():
        kwargs = {}
        for k, spec in fn.structured_input_signature[1].items():
            shape = tuple(1 if d is None else d for d in spec.shape)
            kwargs[k] = tf.constant(np.full(shape, "x")) if spec.dtype == tf.string else tf.zeros(shape, spec.dtype)
        fn(**kwargs)


def check_tf(label, fixture, op, placement, benign, wiring, second, pbtxt):
    rng = random.Random(7)
    shutil.rmtree(MARK, ignore_errors=True)
    MARK.mkdir()
    path = str(MARK / "wf") if op == "WriteFile" else str(MARK / "probe")
    (MARK / "probe").write_text("probe\n")
    try:
        files, info = savedmodel.build(fixture, op=op, placement=placement, path=path, rng=rng, benign=benign, wiring=wiring, second_meta_graph=second, pbtxt=pbtxt, inert_strings=[b"os.system", b"ReadFile"] if benign else None)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "m"
            write_dir(files, root)
            loaded = tf.saved_model.load(str(root), tags=["serve"])
            exempt = op in savedmodel.CALL_EXEMPT_OPS
            if not exempt:
                call_signatures(loaded)
        fired = (MARK / "wf").exists()
        ok = (op != "WriteFile") or benign or fired
        if benign and fired:
            ok = False
        results.append((label, "OK" if ok else "FAIL", f"fired={fired} info={info.get('function')}/{info.get('node')} exempt={exempt}"))
    except Exception as exc:  # noqa: BLE001
        results.append((label, "FAIL", f"{type(exc).__name__}: {str(exc)[:300]}"))


def check_keras(label, fixture, ext, where, form, benign):
    shutil.rmtree(MARK, ignore_errors=True)
    MARK.mkdir()
    # the corpus lambdas write under /run/canary; this check points at /tmp/gtcheck instead
    src = "lambda x: (open('/tmp/gtcheck/marker', 'w'), x)[1]"
    import base64, marshal

    code = base64.b64encode(marshal.dumps(compile(src, "<lambda>", "eval").co_consts[0])).decode("ascii")
    try:
        if benign:
            layer = keras_files.named_lambda_layer("lam_b", "relu") if form == "keras3" else keras_files.activation_layer("act_b", "relu")
        else:
            layer = keras_files.lambda_layer("lam", code)

        def mutate(cfg):
            return keras_files.insert_layer(cfg, layer, where=where)

        data = keras_files.build(fixture, ext, mutate)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m" + ext)
            with open(p, "wb") as fh:
                fh.write(data)
            model = keras.saving.load_model(p, safe_mode=False) if ext == ".keras" else keras.saving.load_model(p)
            shape = tuple(1 if d is None else d for d in model.inputs[0].shape)
            model(np.zeros(shape, dtype="float32"))
        fired = (MARK / "marker").exists()
        ok = fired != benign
        results.append((label, "OK" if ok else "FAIL", f"fired={fired}"))
    except Exception as exc:  # noqa: BLE001
        results.append((label, "FAIL", f"{type(exc).__name__}: {str(exc)[:300]}"))


def main():
    for fixture in savedmodel.FIXTURE_NAMES:
        for placement in savedmodel.PLACEMENTS:
            if placement == "while_body" and fixture != "loop":
                continue
            for benign in (False, True):
                check_tf(f"tf {fixture} {placement} WriteFile control benign={benign}", fixture, "WriteFile", placement, benign, "control", False, False)
            check_tf(f"tf {fixture} {placement} ReadFile data", fixture, "ReadFile", placement, False, "data", False, False)
    check_tf("tf matmul serving WriteFile second-meta-graph", "matmul", "WriteFile", "serving", False, "control", True, False)
    check_tf("tf cond serving WriteFile pbtxt", "cond", "WriteFile", "serving", False, "control", False, True)
    check_tf("tf matmul skeleton benign", "matmul", None, "serving", True, "control", False, False)
    for op in savedmodel.OP_SPECS:
        if op in ("WriteFile", "ReadFile"):
            continue
        check_tf(f"tf two_signatures serving {op}", "two_signatures", op, "serving", False, "control", False, False)
    for fixture, ext, where in [("functional", ".keras", "top"), ("functional", ".h5", "top"), ("nested", ".keras", "nested"), ("nested", ".h5", "nested"), ("wrappers", ".keras", "wrapper"), ("wrappers", ".h5", "wrapper"), ("conv", ".keras", "top"), ("conv", ".h5", "top")]:
        check_keras(f"keras {fixture}{ext} {where} lambda", fixture, ext, where, "keras3", False)
        check_keras(f"keras {fixture}{ext} {where} benign", fixture, ext, where, "keras3", True)
    bad = 0
    for label, status, detail in results:
        print(f"[{status}] {label}: {detail}")
        bad += status != "OK"
    print(f"{len(results)} checks, {bad} failures")
    json.dump(results, open("/repo/scripts/gt-validate/check_tfkeras_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
